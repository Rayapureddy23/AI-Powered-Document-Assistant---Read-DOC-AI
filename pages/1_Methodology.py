"""
1_Methodology.py — Research question, design, and metric definitions.
======================================================================

Reflects the final deterministic three-construct evaluation:
  Accuracy             → Answer Accuracy
  Contextual relevance → Context F1@k (Precision@k + Evidence Recall@k)
  Faithfulness         → sentence-level grounding
"""

import os, sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import apply_theme
from src.config import EXPERIMENTS, QUESTIONS, OLLAMA_MODEL, EMBEDDING_MODEL

st.set_page_config(page_title="Methodology — ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("Methodology",
            "Research question, 3×3 factorial design, and deterministic metric definitions.")

# ── Research question ─────────────────────────────────────────────────────────
st.header("Research question")
st.info('**"How does varying chunk size and retrieval depth in a Retrieval-Augmented '
        'Generation pipeline affect the accuracy, contextual relevance, and '
        'faithfulness of answers generated from unstructured documents?"**')
st.caption("The question names three constructs. Each maps to one dedicated, "
           "deterministic metric.")

# ── Design ────────────────────────────────────────────────────────────────────
st.header("Experimental design — 3 × 3 factorial")
c1, c2, c3 = st.columns(3)
c1.metric("Configurations", "9 + baseline")
c2.metric("Questions per config", "10")
c3.metric("Constructs", "3")

st.dataframe(
    pd.DataFrame([{"Experiment": e, "Chunk size (chars)": c, "Retrieval depth k": k}
                  for e, c, k in EXPERIMENTS]),
    hide_index=True, use_container_width=True)

st.markdown(
    "The **baseline** runs the same 10 questions with zero document context — "
    "the model answers from training knowledge alone. Retrieval-dependent metrics "
    "(contextual relevance, faithfulness) are undefined for the baseline and shown "
    "as N/A, making it both a control condition and a validity check.")

# ── Test set ──────────────────────────────────────────────────────────────────
st.header("Fixed test set")
st.dataframe(
    pd.DataFrame([{"Q": q["id"], "Category": q["cat"], "Question": q["text"]}
                  for q in QUESTIONS]),
    hide_index=True, use_container_width=True)
st.caption("Q9–Q10 are out-of-scope by design: the correct behaviour is refusal.")

# ── Metric definitions ────────────────────────────────────────────────────────
st.header("Evaluation metrics — deterministic, three constructs")
st.markdown(
    "All metrics are computed **deterministically** from sentence-transformer "
    "embeddings. No LLM judge is used, so re-running the evaluation on the same "
    "answers always yields identical scores — the results are exactly reproducible.")

st.subheader("Foundation — cosine similarity")
st.latex(r"\cos(A,B)=\frac{A\cdot B}{\lVert A\rVert\,\lVert B\rVert}"
         r"=\frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\sqrt{\sum_i b_i^2}}\in[0,1]")
st.caption("E(x) below denotes the 384-dimensional all-MiniLM-L6-v2 embedding of text x.")

# Construct 1
st.subheader("Construct 1 · Accuracy → Answer Accuracy")
st.latex(r"\text{Answer Accuracy}=\cos\bigl(E(\text{answer}),\,E(\text{reference})\bigr)")
st.markdown("Semantic similarity between the generated answer and an expert-written "
            "reference answer. Captures correctness independent of exact wording. "
            "For out-of-scope questions: 1.0 for a correct refusal, 0.0 for a "
            "hallucinated answer. *(This is a similarity-based measure; claim-level "
            "factual verification is identified as future work.)*")

# Construct 2
st.subheader("Construct 2 · Contextual relevance → Context F1@k")
st.markdown("Contextual relevance combines two sub-metrics computed against "
            "**silver evidence** — corpus chunks whose similarity to the reference "
            "answer exceeds a threshold, derived once at the smallest chunk size and "
            "frozen across all configurations.")
st.latex(r"\text{Context Precision@}k=\frac{\text{relevant retrieved chunks}}{k}")
st.latex(r"\text{Evidence Recall@}k=\frac{\text{unique silver evidence covered}}{\text{total silver evidence}}")
st.latex(r"\text{Context F1@}k=\frac{2\cdot P\cdot R}{P+R}\quad(\text{macro-averaged over questions})")
st.markdown("Precision measures retrieval purity (it falls as k grows — more chunks, "
            "more noise); Evidence Recall measures coverage (it rises with k — more "
            "chunks, more evidence found). Context F1@k balances the two and is the "
            "primary contextual-relevance metric.")

# Construct 3
st.subheader("Construct 3 · Faithfulness")
st.latex(r"\text{Faithfulness}=\frac{1}{|S|}\sum_{s\in S}"
         r"\max_{c\in\text{chunks}}\cos\bigl(E(s),E(c)\bigr)")
st.markdown("Sentence-level grounding: each sentence *s* in the answer is matched to "
            "its most similar retrieved chunk, and the scores are averaged. A faithful "
            "answer stays close to the retrieved context; a hallucinated one drifts "
            "away. **Baseline = N/A** — with no retrieved context, grounding is "
            "undefined. Computed deterministically (no LLM judge).")

# Mapping
st.subheader("Mapping to the research question")
st.dataframe(pd.DataFrame([
    {"RQ construct": "Accuracy",
     "Metric": "Answer Accuracy"},
    {"RQ construct": "Contextual relevance",
     "Metric": "Context F1@k  (Precision@k + Evidence Recall@k)"},
    {"RQ construct": "Faithfulness",
     "Metric": "Faithfulness (sentence-level grounding)"},
]), hide_index=True, use_container_width=True)

# ── Worked example ────────────────────────────────────────────────────────────
st.header("Worked example — Context metrics (k = 5)")
st.code("""Silver evidence units for the question : {S1, S2, S3}   (total = 3)
Retrieved top-5, evidence covered per chunk:
  rank 1: —          not relevant
  rank 2: S1         relevant
  rank 3: —          not relevant
  rank 4: S3         relevant
  rank 5: S1         relevant (duplicate — not double-counted)

relevant chunks = 3 of 5      unique evidence covered = {S1, S3} = 2

Context Precision@5 = 3 / 5 = 0.6000
Evidence Recall@5   = 2 / 3 = 0.6667
Context F1@5        = 2·0.6·0.6667 / (0.6+0.6667) = 0.6316""")

# ── Justification ─────────────────────────────────────────────────────────────
st.header("Why deterministic metrics (no LLM judge)")
st.markdown(
    "- **Reproducibility** — identical inputs always produce identical scores; an "
    "LLM judge introduces stochastic variation and parse failures.\n"
    "- **Zero external dependency** — no API quotas, keys, or rate limits can "
    "invalidate an experimental run.\n"
    "- **Academic grounding** — embedding cosine similarity is the same operation "
    "used inside RAGAS answer/context relevance (Es et al., 2023, arXiv:2309.15217); "
    "Precision@k and Recall@k are standard information-retrieval metrics "
    "(Manning et al., 2008).\n"
    "- **Construct validity** — each metric maps one-to-one onto a construct named "
    "in the research question.")

# ── Limitations ───────────────────────────────────────────────────────────────
st.header("Documented limitations")
st.markdown(
    "- **Silver evidence** is model-derived (from reference-answer similarity), not "
    "human-annotated, so contextual-relevance metrics measure retrieval consistency "
    "with that signal rather than agreement with human judgment.\n"
    "- **Answer Accuracy** is an embedding-similarity measure, not claim-level "
    "factual verification.\n"
    "- The **test set** of 10 questions (2 out-of-scope) makes findings indicative "
    "rather than conclusive.")

# ── Configuration ─────────────────────────────────────────────────────────────
st.header("System configuration (held constant)")
st.dataframe(pd.DataFrame([
    {"Component": "Answer LLM",      "Value": f"{OLLAMA_MODEL} via Ollama (local), temperature 0.1"},
    {"Component": "Embedding model", "Value": f"{EMBEDDING_MODEL} (384-dim)"},
    {"Component": "Vector index",    "Value": "FAISS IndexFlatL2 (exact search)"},
    {"Component": "Retrieval",       "Value": "top-10 once, sliced to k (nested comparison)"},
    {"Component": "Chunk overlap",   "Value": "100 characters"},
    {"Component": "Variables",       "Value": "chunk size ∈ {300, 600, 1000} · k ∈ {3, 5, 10}"},
]), hide_index=True, use_container_width=True)