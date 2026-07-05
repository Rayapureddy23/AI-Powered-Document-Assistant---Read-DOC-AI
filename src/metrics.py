"""
metrics.py — Fully automated evaluation metrics.
=================================================
Every metric is computed deterministically from embeddings and
ground-truth page labels. NO LLM judge — this removes the entire
class of failures encountered with LLM-as-judge approaches
(parse errors, timeouts, rate limits, NaN scores) and makes every
score exactly reproducible.

Mapping to the research question:

  RQ construct            Metric(s)
  ─────────────────────   ─────────────────────────────────────────────
  Accuracy             →  answer_accuracy   = cos( E(answer), E(reference) )
                          recall_at_k       = |Relevant ∩ Retrieved| / |Relevant|
  Contextual relevance →  context_relevance = mean cos( E(question), E(chunk_i) )
                          precision_at_k    = |Relevant ∩ Retrieved| / |Retrieved|
                          mrr               = 1 / rank of first relevant chunk
  Faithfulness         →  faithfulness      = cos( E(answer), mean E(chunks) )
                          (baseline ≡ 0.0 — no context exists to be faithful to)

Academic grounding:
  - Embedding cosine similarity is the same operation RAGAS uses inside
    Answer Relevancy (Es et al., 2023, arXiv:2309.15217).
  - Precision@k / Recall@k / MRR are classical IR metrics
    (Manning, Raghavan & Schütze, Introduction to Information Retrieval, 2008).
"""

import numpy as np
from src.config import GROUND_TRUTH, REFUSAL
from src.retriever import get_embedding_model


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _clip01(x: float) -> float:
    return round(min(max(x, 0.0), 1.0), 4)


def score_question(qid: int, question: str, answer: str, chunks: list,
                   is_baseline: bool = False) -> dict:
    """All metrics for one question. Deterministic; ~50 ms on CPU."""
    model = get_embedding_model()
    gt    = GROUND_TRUTH[qid]

    e_answer    = model.encode([answer])[0]
    e_reference = model.encode([gt["reference"]])[0]
    e_question  = model.encode([question])[0]

    # ── Accuracy: semantic similarity to expert reference answer ─────────────
    # Out-of-scope handling: correct refusal must score 1.0, hallucination 0.0.
    if not gt["pages"]:
        refused = REFUSAL.lower()[:30] in answer.lower() or "could not find" in answer.lower()
        answer_accuracy = 1.0 if refused else 0.0
    else:
        answer_accuracy = _clip01(_cos(e_answer, e_reference))

    # ── Faithfulness: grounding of the answer in retrieved context ───────────
    if is_baseline or not chunks:
        faithfulness = 0.0                      # no context ⇒ nothing to be faithful to
    else:
        ctx_vecs     = model.encode([c["text"] for c in chunks])
        faithfulness = _clip01(_cos(e_answer, ctx_vecs.mean(axis=0)))

    # ── Contextual relevance: are retrieved chunks about the question? ───────
    if is_baseline or not chunks:
        context_relevance = 0.0
    else:
        sims = [_cos(e_question, v) for v in model.encode([c["text"] for c in chunks])]
        context_relevance = _clip01(float(np.mean(sims)))

    # ── Retrieval metrics vs ground-truth pages ──────────────────────────────
    if gt["pages"] and chunks and not is_baseline:
        retrieved = {str(c["page_number"]) for c in chunks}
        relevant  = {str(p) for p in gt["pages"]}
        overlap   = retrieved & relevant
        precision = _clip01(len(overlap) / len(retrieved)) if retrieved else 0.0
        recall    = _clip01(len(overlap) / len(relevant))  if relevant  else 0.0
        mrr = 0.0
        for rank, c in enumerate(chunks, start=1):
            if str(c["page_number"]) in relevant:
                mrr = _clip01(1.0 / rank)
                break
    else:
        precision = recall = mrr = None  # undefined for baseline / out-of-scope

    return {
        "answer_accuracy":   answer_accuracy,
        "faithfulness":      faithfulness,
        "context_relevance": context_relevance,
        "precision_at_k":    precision,
        "recall_at_k":       recall,
        "mrr":               mrr,
    }


def aggregate(per_question: list) -> dict:
    """Mean of every metric across questions (skipping None values)."""
    def mean_of(key):
        vals = [q[key] for q in per_question if q.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    acc   = mean_of("answer_accuracy")
    faith = mean_of("faithfulness")
    ctx   = mean_of("context_relevance")
    prec  = mean_of("precision_at_k")
    rec   = mean_of("recall_at_k")
    mrr   = mean_of("mrr")

    # RQ construct scores
    accuracy_construct  = round((acc + rec) / 2, 4) if rec else acc
    relevance_construct = round((ctx + prec + mrr) / 3, 4) if prec else ctx
    overall             = round((accuracy_construct + relevance_construct + faith) / 3, 4)

    return {
        "answer_accuracy":     acc,
        "faithfulness":        faith,
        "context_relevance":   ctx,
        "precision_at_k":      prec,
        "recall_at_k":         rec,
        "mrr":                 mrr,
        "rq_accuracy":         accuracy_construct,
        "rq_relevance":        relevance_construct,
        "rq_faithfulness":     faith,
        "overall":             overall,
    }
