"""
metrics.py — Deterministic RAG evaluation, three RQ constructs.
===============================================================

All metrics deterministic (embeddings only, no LLM judge) — fast and
exactly reproducible. Three constructs, mapped to the research question:

  Accuracy             → Answer Accuracy      = cos(answer, reference)
  Contextual relevance → Context Relevance    = mean cos(question, chunks)
                         Context Precision@k  = relevant chunks / k
                         Evidence Recall@k    = silver evidence covered / total
                         Context F1@k         = harmonic mean of the two
  Faithfulness         → Faithfulness         = mean over answer sentences of
                                                max cos(sentence, chunk)
                                                (sentence-level grounding)

Silver evidence (auto-derived): a corpus chunk is a silver-evidence unit if
its similarity to the reference answer ≥ threshold, frozen at the smallest
chunk size. NOT human-annotated — a documented limitation.
"""

import re
import numpy as np
from src.config import (GROUND_TRUTH, REFUSAL_PATTERNS,
                        RELEVANCE_THRESHOLD, GOLD_EVIDENCE_THRESHOLD)
from src.retriever import get_embedding_model


def _cos(a, b) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d else 0.0


def _clip01(x) -> float:
    return round(min(max(x, 0.0), 1.0), 4)


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in REFUSAL_PATTERNS)


def _f1(p, r):
    if p is None or r is None:
        return None
    return 0.0 if (p + r) == 0 else _clip01(2 * p * r / (p + r))


def _sentences(text: str) -> list:
    """Split answer into sentences for sentence-level faithfulness."""
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in parts if len(s.strip()) > 10]


# ── Silver evidence ───────────────────────────────────────────────────────────
def build_gold_evidence(all_chunks: list) -> dict:
    """Derive silver evidence per in-scope question from reference answers.
    Build ONCE at smallest chunk size; freeze for all configs."""
    model = get_embedding_model()
    cvecs = model.encode([c["text"] for c in all_chunks],
                         batch_size=128, convert_to_numpy=True)
    gold = {}
    for qid, gt in GROUND_TRUTH.items():
        if gt["out_of_scope"]:
            gold[qid] = []
            continue
        e_ref = model.encode([gt["reference"]])[0]
        units, seen = [], set()
        for c, v in zip(all_chunks, cvecs):
            if _cos(e_ref, v) >= GOLD_EVIDENCE_THRESHOLD:
                pg = c["page_number"]
                if pg not in seen:
                    seen.add(pg)
                    units.append({"page": pg, "text": c["text"],
                                  "eid": f"S{qid}_{len(units)+1}", "vec": v})
        if not units:
            sims = [_cos(e_ref, v) for v in cvecs]
            b = int(np.argmax(sims)); c = all_chunks[b]
            units = [{"page": c["page_number"], "text": c["text"],
                      "eid": f"S{qid}_1", "vec": cvecs[b]}]
        gold[qid] = units
    return gold


# ── The one scoring function: all three constructs ───────────────────────────
def score_question(qid, question, answer, chunks, gold_evidence,
                   is_baseline=False, chunk_vecs=None):
    """Compute Accuracy, Contextual Relevance (3 sub-metrics), Faithfulness.
    Deterministic. Returns None for retrieval/faithfulness on baseline."""
    model = get_embedding_model()
    gt = GROUND_TRUTH[qid]
    e_ans = model.encode([answer])[0]
    e_ref = model.encode([gt["reference"]])[0]

    # ── ACCURACY ──────────────────────────────────────────────────────────
    if gt["out_of_scope"]:
        answer_accuracy = 1.0 if is_refusal(answer) else 0.0
    else:
        answer_accuracy = _clip01(_cos(e_ans, e_ref))

    # baseline: no retrieval-based metrics
    if is_baseline or not chunks or gt["out_of_scope"]:
        return {"answer_accuracy": answer_accuracy,
                "context_relevance": None, "context_precision_at_k": None,
                "evidence_recall_at_k": None, "context_f1_at_k": None,
                "faithfulness": None,
                "is_out_of_scope": gt["out_of_scope"],
                "correct_refusal": (1 if is_refusal(answer) else 0)
                                   if gt["out_of_scope"] else None}

    if chunk_vecs is None:
        chunk_vecs = model.encode([c["text"] for c in chunks],
                                  batch_size=64, convert_to_numpy=True)
    e_q = model.encode([question])[0]

    # ── CONTEXTUAL RELEVANCE ──────────────────────────────────────────────
    context_relevance = _clip01(float(np.mean([_cos(e_q, v) for v in chunk_vecs])))

    units = gold_evidence.get(qid, [])
    total = len(units)
    covered, flags, first_rel = set(), [], None
    for rank, (c, cv) in enumerate(zip(chunks, chunk_vecs), 1):
        hit = []
        for u in units:
            if (c["page_number"] == u["page"] and _cos(cv, u["vec"]) >= 0.5) \
               or _cos(cv, u["vec"]) >= 0.8:
                hit.append(u["eid"])
        flags.append(1 if hit else 0)
        if hit:
            covered.update(hit)
            if first_rel is None:
                first_rel = rank
    precision = _clip01(sum(flags) / len(chunks))
    recall = _clip01(len(covered) / total) if total else 0.0
    context_f1 = _f1(precision, recall)

    # ── FAITHFULNESS (deterministic, sentence-level grounding) ────────────
    # Each answer sentence should be supported by SOME retrieved chunk.
    # faithfulness = mean over sentences of max similarity to any chunk.
    sents = _sentences(answer)
    if sents:
        svecs = model.encode(sents, batch_size=32, convert_to_numpy=True)
        per_sent = [max(_cos(sv, cv) for cv in chunk_vecs) for sv in svecs]
        faithfulness = _clip01(float(np.mean(per_sent)))
    else:
        faithfulness = _clip01(_cos(e_ans, chunk_vecs.mean(axis=0)))

    return {"answer_accuracy": answer_accuracy,
            "context_relevance": context_relevance,
            "context_precision_at_k": precision,
            "evidence_recall_at_k": recall,
            "context_f1_at_k": context_f1,
            "faithfulness": faithfulness,
            "chunk_relevance_flags": flags,
            "is_out_of_scope": False, "correct_refusal": None}


def aggregate(per_question: list) -> dict:
    def mean_of(k):
        vals = [q[k] for q in per_question if q.get(k) is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    return {
        "answer_accuracy":        mean_of("answer_accuracy"),
        "context_relevance":      mean_of("context_relevance"),
        "context_precision_at_k": mean_of("context_precision_at_k"),
        "evidence_recall_at_k":   mean_of("evidence_recall_at_k"),
        "context_f1_at_k":        mean_of("context_f1_at_k"),
        "faithfulness":           mean_of("faithfulness"),
        "n_questions":            len(per_question),
    }