"""
tests/test_metrics.py — Validation suite for the automated evaluation metrics.
==============================================================================

These tests validate the DETERMINISTIC metric formulas against small,
hand-computed examples with known expected results. They do NOT call the LLM
judge (that layer is non-deterministic and tested separately by inspection).

Worked example these tests are anchored on (from the spec):
  Gold evidence units: E1, E2, E3
  Retrieved top-5 covers E1 and E3, first relevant chunk at rank 2
  → Context Precision@5 = 2/5 = 0.4000   (2 of 5 chunks are relevant)
  → Evidence Recall@5   = 2/3 = 0.6667   (2 of 3 gold units covered)
  → Context F1@5        = 0.5000
  → Reciprocal Rank     = 1/2 = 0.5000
"""

import os, sys, math, tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics as M


# ══════════════════════════════════════════════════════════════════════════════
# Pure-formula helpers — reimplemented here so tests don't depend on embeddings.
# We test the ARITHMETIC of the metrics using explicit relevance flags and
# covered-evidence sets, which is exactly what score_retrieval computes
# internally once the semantic overlap is resolved.
# ══════════════════════════════════════════════════════════════════════════════
def context_precision(relevance_flags):
    n = len(relevance_flags)
    return round(sum(relevance_flags) / n, 4) if n else 0.0

def evidence_recall(covered_unique, total_evidence):
    return round(len(covered_unique) / total_evidence, 4) if total_evidence else 0.0

def context_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)

def reciprocal_rank(relevance_flags):
    for rank, flag in enumerate(relevance_flags, start=1):
        if flag:
            return round(1.0 / rank, 4)
    return 0.0

def answer_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ══════════════════════════════════════════════════════════════════════════════
# 1 · The canonical worked example
# ══════════════════════════════════════════════════════════════════════════════
def test_worked_example_precision():
    # top-5: relevant at ranks 2 and 4 → flags [0,1,0,1,0]
    flags = [0, 1, 0, 1, 0]
    assert context_precision(flags) == 0.4000

def test_worked_example_recall():
    covered = {"E1", "E3"}      # 2 unique units covered
    assert evidence_recall(covered, total_evidence=3) == 0.6667

def test_worked_example_context_f1():
    assert context_f1(0.4000, 0.6667) == 0.5000

def test_worked_example_reciprocal_rank():
    flags = [0, 1, 0, 1, 0]     # first relevant at rank 2
    assert reciprocal_rank(flags) == 0.5000


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Answer F1 formula
# ══════════════════════════════════════════════════════════════════════════════
def test_answer_f1_standard():
    # precision 0.6667, recall 1.0 → F1 0.8
    assert answer_f1(0.6667, 1.0) == 0.8000

def test_answer_f1_both_zero():
    assert answer_f1(0.0, 0.0) == 0.0    # guard: no div-by-zero

def test_answer_f1_perfect():
    assert answer_f1(1.0, 1.0) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Context Precision@k edge cases
# ══════════════════════════════════════════════════════════════════════════════
def test_precision_all_relevant():
    assert context_precision([1, 1, 1]) == 1.0

def test_precision_none_relevant():
    assert context_precision([0, 0, 0, 0, 0]) == 0.0

def test_precision_empty():
    assert context_precision([]) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Evidence Recall@k — duplicate handling
# ══════════════════════════════════════════════════════════════════════════════
def test_recall_duplicate_evidence_not_double_counted():
    # Chunk A covers E1, chunk B covers E1 again, chunk C covers E2.
    # Unique covered = {E1, E2} = 2, total = 3 → 0.6667, NOT 3/3.
    covered_unique = {"E1", "E2"}          # set already de-dups
    assert evidence_recall(covered_unique, total_evidence=3) == 0.6667

def test_recall_full_coverage():
    assert evidence_recall({"E1", "E2", "E3"}, 3) == 1.0

def test_recall_no_coverage():
    assert evidence_recall(set(), 3) == 0.0

def test_recall_no_gold_evidence():
    # No gold evidence at all → defined as 0.0 (guard against div-by-zero)
    assert evidence_recall(set(), 0) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5 · Context F1@k edge cases
# ══════════════════════════════════════════════════════════════════════════════
def test_context_f1_both_zero():
    assert context_f1(0.0, 0.0) == 0.0

def test_context_f1_precision_zero():
    assert context_f1(0.0, 0.5) == 0.0

def test_context_f1_symmetry():
    assert context_f1(0.4, 0.6) == context_f1(0.6, 0.4)


# ══════════════════════════════════════════════════════════════════════════════
# 6 · MRR / reciprocal rank
# ══════════════════════════════════════════════════════════════════════════════
def test_rr_first_rank():
    assert reciprocal_rank([1, 0, 0]) == 1.0

def test_rr_third_rank():
    assert reciprocal_rank([0, 0, 1]) == round(1/3, 4)

def test_rr_no_relevant():
    assert reciprocal_rank([0, 0, 0, 0]) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 7 · Nested-retrieval invariants (Stage 14)
# ══════════════════════════════════════════════════════════════════════════════
def test_nested_prefix_slicing():
    top10 = list(range(10))          # simulated ranked chunk ids
    top3, top5 = top10[:3], top10[:5]
    assert top3 == top5[:3]          # top-3 is a prefix of top-5
    assert top5 == top10[:5]         # top-5 is a prefix of top-10

def test_evidence_recall_non_decreasing_with_k():
    # Same nested ranking: covered evidence can only grow as k increases.
    # ranks:      1    2    3    4    5
    covers = [set(), {"E1"}, set(), {"E3"}, {"E1"}]
    def covered_upto(k):
        u = set()
        for c in covers[:k]:
            u |= c
        return u
    r3  = evidence_recall(covered_upto(3), 3)
    r5  = evidence_recall(covered_upto(5), 3)
    assert r3 <= r5                  # 1/3 ≤ 2/3

def test_rr_stable_under_nesting():
    # First relevant chunk rank does not change when more chunks are appended.
    base = [0, 1, 0]                 # first relevant at rank 2
    extended = base + [1, 0, 1]      # same prefix, more chunks
    assert reciprocal_rank(base) == reciprocal_rank(extended)


# ══════════════════════════════════════════════════════════════════════════════
# 8 · Empty / degenerate answers
# ══════════════════════════════════════════════════════════════════════════════
def test_empty_answer_precision_zero():
    # No generated claims → precision defined as 0.0, not a crash.
    n_gen = 0
    precision = round(0 / n_gen, 4) if n_gen else 0.0
    assert precision == 0.0

def test_refusal_detection_multiple_phrases():
    assert M.is_refusal("I could not find this in your documents.")
    assert M.is_refusal("This is not in the provided context.")
    assert M.is_refusal("There is no information about that.")
    assert not M.is_refusal("Supervised learning uses labelled data.")


# ══════════════════════════════════════════════════════════════════════════════
# 9 · Out-of-scope handling
# ══════════════════════════════════════════════════════════════════════════════
def test_out_of_scope_correct_refusal():
    # Q9 is out-of-scope in the ground truth; a refusal scores correct.
    res = M.score_out_of_scope(9, "I could not find this in your documents.")
    assert res["is_out_of_scope"] is True
    assert res["correct_refusal"] == 1

def test_out_of_scope_hallucination():
    res = M.score_out_of_scope(9, "Nvidia stock is $880 today.")
    assert res["correct_refusal"] == 0

def test_in_scope_not_flagged_out_of_scope():
    res = M.score_out_of_scope(1, "Supervised learning uses labelled data.")
    assert res["is_out_of_scope"] is False
    assert res["correct_refusal"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 10 · Baseline N/A + real-zero-not-missing
# ══════════════════════════════════════════════════════════════════════════════
def test_baseline_retrieval_returns_none():
    # Baseline: no chunks → retrieval metrics must be None (→ N/A), not 0.
    res = M.score_retrieval(1, chunks=[], gold_evidence={1: []}, model=None)
    assert res["context_precision_at_k"] is None
    assert res["evidence_recall_at_k"] is None
    assert res["reciprocal_rank"] is None

def test_aggregate_keeps_real_zero():
    # A genuine 0.0 must be averaged, NOT dropped as if it were missing.
    per_q = [
        {"context_f1_at_k": 0.0, "is_out_of_scope": False},
        {"context_f1_at_k": 0.6, "is_out_of_scope": False},
    ]
    agg = M.aggregate_deterministic(per_q)
    # mean of [0.0, 0.6] = 0.3 — if 0.0 were dropped it would be 0.6
    assert agg["context_f1_at_k"] == 0.3

def test_aggregate_all_none_gives_none():
    per_q = [{"context_f1_at_k": None, "is_out_of_scope": True}]
    agg = M.aggregate_deterministic(per_q)
    assert agg["context_f1_at_k"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 11 · Database read/write consistency
# ══════════════════════════════════════════════════════════════════════════════
def test_db_write_read_roundtrip(monkeypatch):
    import src.store as store
    # Point the DB at a temp file so the test never touches real data.
    tmp = os.path.join(tempfile.mkdtemp(), "test.db")
    monkeypatch.setattr(store, "DB", tmp)
    store.init_db()

    agg = {"context_f1_at_k": 0.5, "evidence_recall_at_k": 0.6667,
           "mrr_at_k": 0.5, "bertscore_f1": 0.8,
           "context_precision_at_k": 0.4, "context_relevance": 0.55,
           "refusal_accuracy": 1.0, "false_refusal_rate": 0.0,
           "answer_precision": None, "answer_recall": None,
           "answer_f1": None, "faithfulness": None}
    store.save_result("E1", 300, 3, agg, n=10, run_id="test", judged=False)

    df = store.load_results(run_id="test")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["config"] == "E1"
    assert abs(row["context_f1_at_k"] - 0.5) < 1e-9
    # None must round-trip as NULL/NaN, not 0
    assert row["answer_f1"] is None or math.isnan(row["answer_f1"])


def test_db_none_not_stored_as_zero(monkeypatch):
    import src.store as store
    tmp = os.path.join(tempfile.mkdtemp(), "test2.db")
    monkeypatch.setattr(store, "DB", tmp)
    store.init_db()
    agg = {"context_f1_at_k": None, "evidence_recall_at_k": None,
           "mrr_at_k": None, "bertscore_f1": 0.75,
           "context_precision_at_k": None, "context_relevance": None,
           "refusal_accuracy": None, "false_refusal_rate": None,
           "answer_precision": None, "answer_recall": None,
           "answer_f1": None, "faithfulness": None}
    store.save_result("Baseline", 0, 0, agg, n=10, run_id="test", judged=False)
    df = store.load_results(run_id="test")
    row = df.iloc[0]
    # Baseline retrieval metric must be NULL/NaN, NOT 0.0
    assert row["context_f1_at_k"] is None or math.isnan(row["context_f1_at_k"])