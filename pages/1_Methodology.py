"""
1_Methodology.py — Research question, experimental design, metric definitions.
"""

import os, sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import apply_theme
from src.config import EXPERIMENTS, QUESTIONS, OLLAMA_MODEL, EMBEDDING_MODEL

st.set_page_config(page_title="Methodology — ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("Methodology",
            "Research question, 3×3 factorial design, and automated metric definitions.")

# ── Research question ─────────────────────────────────────────────────────────
st.header("Research question")
st.info('**"How does varying chunk size and retrieval depth in a Retrieval-Augmented '
        'Generation pipeline affect the accuracy, contextual relevance, and '
        'faithfulness of answers generated from unstructured documents?"**')

# ── Design ────────────────────────────────────────────────────────────────────
st.header("Experimental design — 3 × 3 factorial")
c1, c2, c3 = st.columns(3)
c1.metric("Configurations", "9 + baseline")
c2.metric("Questions per config", "10")
c3.metric("Metrics", "6 automated")

st.dataframe(
    pd.DataFrame([{"Experiment": e, "Chunk size (chars)": c, "Retrieval depth k": k}
                  for e, c, k in EXPERIMENTS]),
    hide_index=True, use_container_width=True)

st.markdown(
    "The **baseline** condition runs the same 10 questions with zero document "
    "context — the LLM answers from training data alone. Faithfulness is 0.0 "
    "by definition (no context exists to be faithful to), making the baseline "
    "both a control condition and a built-in validity check.")

# ── Test set ──────────────────────────────────────────────────────────────────
st.header("Fixed test set")
st.dataframe(
    pd.DataFrame([{"Q": q["id"], "Category": q["cat"], "Question": q["text"]}
                  for q in QUESTIONS]),
    hide_index=True, use_container_width=True)
st.caption("Q9–Q10 are out-of-scope by design: the correct behaviour is refusal. "
           "A correct refusal scores accuracy 1.0; a hallucinated answer scores 0.0.")

# ── Metric definitions ────────────────────────────────────────────────────────
st.header("Automated metrics — definitions and formulas")
st.markdown(
    "All metrics are computed **deterministically** from sentence-transformer "
    "embeddings and ground-truth page labels. No LLM judge is used, which makes "
    "every score exactly reproducible — re-running the evaluation on the same "
    "answers always yields identical numbers.")

st.subheader("Foundation — cosine similarity")
st.latex(r"\cos(A,B)=\frac{A\cdot B}{\lVert A\rVert\,\lVert B\rVert}"
         r"=\frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\sqrt{\sum_i b_i^2}}\in[0,1]")
st.caption("E(x) below denotes the 384-dimensional all-MiniLM-L6-v2 embedding of text x.")

st.subheader("1 · Answer Accuracy")
st.latex(r"\text{Accuracy}=\cos\bigl(E(\text{answer}),\,E(\text{reference})\bigr)")
st.markdown("Semantic similarity between the generated answer and an expert-written "
            "reference answer. Captures correctness independent of exact wording. "
            "For out-of-scope questions: 1.0 for a correct refusal, 0.0 for a "
            "hallucinated answer.")

st.subheader("2 · Faithfulness")
st.latex(r"\text{Faithfulness}=\cos\Bigl(E(\text{answer}),\,"
         r"\tfrac{1}{k}\textstyle\sum_{i=1}^{k}E(\text{chunk}_i)\Bigr)")
st.markdown("Grounding of the answer in the retrieved context. A faithful answer "
            "stays semantically close to the chunks it was generated from; a "
            "hallucinated answer drifts away. **Baseline ≡ 0.0** — with no retrieved "
            "context, grounding is undefined and set to zero by construction.")

st.subheader("3 · Context Relevance")
st.latex(r"\text{CtxRel}=\tfrac{1}{k}\textstyle\sum_{i=1}^{k}"
         r"\cos\bigl(E(\text{question}),E(\text{chunk}_i)\bigr)")
st.markdown("Are the retrieved chunks actually about the question? Directly measures "
            "retrieval quality at the semantic level.")

st.subheader("4 · Precision@k / 5 · Recall@k / 6 · MRR")
st.latex(r"P@k=\frac{|\text{Relevant}\cap\text{Retrieved}|}{|\text{Retrieved}|}"
         r"\qquad R@k=\frac{|\text{Relevant}\cap\text{Retrieved}|}{|\text{Relevant}|}"
         r"\qquad \text{MRR}=\frac{1}{\text{rank of first relevant chunk}}")
st.markdown("Classical information-retrieval metrics computed against ground-truth "
            "page labels (the pages known to contain each answer). "
            "*Relevant* = ground-truth pages; *Retrieved* = pages of the k returned chunks.")

st.subheader("Mapping to the research question")
st.dataframe(pd.DataFrame([
    {"RQ construct": "Accuracy",
     "Computed as": "mean(Answer Accuracy, Recall@k)"},
    {"RQ construct": "Contextual relevance",
     "Computed as": "mean(Context Relevance, Precision@k, MRR)"},
    {"RQ construct": "Faithfulness",
     "Computed as": "Faithfulness (direct)"},
    {"RQ construct": "Overall",
     "Computed as": "mean of the three construct scores"},
]), hide_index=True, use_container_width=True)

# ── Worked example ────────────────────────────────────────────────────────────
st.header("Worked example (Precision@k, k = 5)")
st.code("""Ground-truth pages : {40, 139}
Retrieved pages    : {37, 38, 40, 139}          (after de-duplication)
Overlap            : {40, 139}                  → |overlap| = 2

Precision@5 = 2 / 4 = 0.50      half of retrieved chunks were relevant
Recall@5    = 2 / 2 = 1.00      every relevant page was found
MRR         = 1 / 1 = 1.00      first retrieved chunk was already relevant""")

# ── Methodological justification ──────────────────────────────────────────────
st.header("Why deterministic metrics (no LLM judge)")
st.markdown(
    "- **Reproducibility** — identical inputs always produce identical scores; "
    "an LLM judge introduces stochastic variation and parse failures.\n"
    "- **Zero external dependency** — no API quotas, keys, or rate limits can "
    "invalidate an experimental run.\n"
    "- **Academic grounding** — embedding cosine similarity is the same operation "
    "used inside RAGAS Answer Relevancy (Es et al., 2023, arXiv:2309.15217); "
    "Precision@k, Recall@k and MRR are standard IR metrics "
    "(Manning et al., *Introduction to Information Retrieval*, 2008).\n"
    "- **Construct validity** — each metric maps one-to-one onto a construct "
    "named in the research question.")

st.header("System configuration (held constant)")
st.dataframe(pd.DataFrame([
    {"Component": "Answer LLM",       "Value": f"{OLLAMA_MODEL} via Ollama (local), temperature 0.1"},
    {"Component": "Embedding model",  "Value": f"{EMBEDDING_MODEL} (384-dim)"},
    {"Component": "Vector index",     "Value": "FAISS IndexFlatL2 (exact search)"},
    {"Component": "Chunk overlap",    "Value": "100 characters"},
    {"Component": "Variables",        "Value": "chunk size ∈ {300, 600, 1000} · k ∈ {3, 5, 10}"},
]), hide_index=True, use_container_width=True)
