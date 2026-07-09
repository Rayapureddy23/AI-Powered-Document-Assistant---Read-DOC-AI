"""
metrics.py — Fully automated evaluation metrics.
=================================================
All metrics are deterministic (embeddings + fixed threshold). No LLM judge,
and no hand-labelled page numbers: retrieval relevance is derived
semantically — a retrieved chunk is RELEVANT if its embedding similarity
to the expert reference answer meets RELEVANCE_THRESHOLD.

  RQ construct            Metric
  ─────────────────────   ────────────────────────────────────────────
  Accuracy             →  answer_accuracy  = cos(E(answer), E(reference))
                          recall_at_k      = hit@k (≥1 relevant chunk retrieved)
  Contextual relevance →  context_relevance = mean cos(E(question), E(chunk_i))
                          precision_at_k   = relevant retrieved / k
                          mrr              = 1 / rank of first relevant chunk
  Faithfulness         →  faithfulness     = cos(E(answer), mean E(chunks))
                          (baseline ≡ 0.0 — no context to be faithful to)
"""

import numpy as np
from src.config import GROUND_TRUTH, REFUSAL, RELEVANCE_THRESHOLD
from src.retriever import get_embedding_model


def _cos(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _clip01(x: float) -> float:
    return round(min(max(x, 0.0), 1.0), 4)


def score_question(qid: int, question: str, answer: str, chunks: list,
                   is_baseline: bool = False) -> dict:
    model = get_embedding_model()
    gt    = GROUND_TRUTH[qid]

    e_answer    = model.encode([answer])[0]
    e_reference = model.encode([gt["reference"]])[0]
    e_question  = model.encode([question])[0]
    ctx_vecs    = model.encode([c["text"] for c in chunks]) if chunks else None

    # ── Accuracy ──────────────────────────────────────────────────────────
    out_of_scope = not gt["pages"] and gt["reference"] == REFUSAL
    if out_of_scope:
        refused = "could not find" in answer.lower()
        answer_accuracy = 1.0 if refused else 0.0
    else:
        answer_accuracy = _clip01(_cos(e_answer, e_reference))

    # ── Faithfulness ──────────────────────────────────────────────────────
    if is_baseline or ctx_vecs is None:
        faithfulness = 0.0
    else:
        faithfulness = _clip01(_cos(e_answer, ctx_vecs.mean(axis=0)))

    # ── Context relevance ─────────────────────────────────────────────────
    if is_baseline or ctx_vecs is None:
        context_relevance = 0.0
    else:
        context_relevance = _clip01(float(np.mean(
            [_cos(e_question, v) for v in ctx_vecs])))

    # ── Retrieval metrics — semantic relevance, no page labels ────────────
    if not is_baseline and ctx_vecs is not None and not out_of_scope:
        sims = [_cos(e_reference, v) for v in ctx_vecs]
        relevant_flags = [s >= RELEVANCE_THRESHOLD for s in sims]
        n_relevant = sum(relevant_flags)
# ── Retrieval metrics — semantic relevance, no page labels ────────────
    if not is_baseline and ctx_vecs is not None and not out_of_scope:
        sims = [_cos(e_reference, v) for v in ctx_vecs]
        relevant_flags = [s >= RELEVANCE_THRESHOLD for s in sims]
        n_relevant = sum(relevant_flags)

        # ---- TEMPORARY DEBUG (remove later) ----
        print(f"  Q{qid} k={len(sims)} sims={[round(s,3) for s in sims]}")
        print(f"  Q{qid} threshold={RELEVANCE_THRESHOLD} n_relevant={n_relevant}")
        # ---- END DEBUG ----
        precision = _clip01(sum(relevant_flags) / len(relevant_flags))
        recall    = 1.0 if any(relevant_flags) else 0.0          # hit@k
        mrr       = 0.0
        for rank, flag in enumerate(relevant_flags, start=1):
            if flag:
                mrr = _clip01(1.0 / rank)
                break
    else:
        precision = recall = mrr = None

    return {
        "answer_accuracy":   answer_accuracy,
        "faithfulness":      faithfulness,
        "context_relevance": context_relevance,
        "precision_at_k":    precision,
        "recall_at_k":       recall,
        "mrr":               mrr,
    }


def aggregate(per_question: list) -> dict:
    def mean_of(key):
        vals = [q[key] for q in per_question if q.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    acc, faith, ctx = mean_of("answer_accuracy"), mean_of("faithfulness"), mean_of("context_relevance")
    prec, rec, mrr  = mean_of("precision_at_k"), mean_of("recall_at_k"), mean_of("mrr")

    accuracy_construct  = round((acc + rec) / 2, 4) if rec else acc
    relevance_construct = round((ctx + prec + mrr) / 3, 4) if prec else ctx
    overall             = round((accuracy_construct + relevance_construct + faith) / 3, 4)

    return {"answer_accuracy": acc, "faithfulness": faith, "context_relevance": ctx,
            "precision_at_k": prec, "recall_at_k": rec, "mrr": mrr,
            "rq_accuracy": accuracy_construct, "rq_relevance": relevance_construct,
            "rq_faithfulness": faith, "overall": overall}

