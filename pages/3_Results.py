"""
3_Results.py — Three-construct results (Accuracy · Relevance · Faithfulness).
==============================================================================

One table per research-question construct. No unnecessary metrics.
Baseline retrieval + faithfulness shown as N/A (undefined without context).
"""

import os, sys
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import apply_theme
from src.store import load_results, load_answers
from src.config import EXPERIMENTS, CHUNK_SIZES, K_VALUES

st.set_page_config(page_title="Results — ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("Results", "Three constructs: Accuracy · Contextual Relevance · Faithfulness.")

PURPLE = "#5C2D91"
NA = "N/A"


def fmt(v):
    return NA if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


df = load_results()
if df.empty:
    st.info("No results yet. On the *Run Experiments* page: build silver evidence, "
            "then click ▶▶ RUN ALL.")
    st.stop()

# Order Baseline → E1..E9
order = ["Baseline"] + [e for e, _, _ in EXPERIMENTS]
df["_o"] = df["config"].apply(lambda c: order.index(c) if c in order else 99)
df = df.sort_values("_o").drop(columns="_o").reset_index(drop=True)
rag = df[df["config"] != "Baseline"].copy()

missing = sorted({e for e, _, _ in EXPERIMENTS} - set(rag["config"].tolist()))
if missing:
    st.warning(f"⚠ Missing: {', '.join(missing)}. Run all configs for a complete grid.")


def show(cols_map):
    d = df[["config"] + list(cols_map.keys())].copy()
    d = d.rename(columns={"config": "Config", **cols_map})
    for c in cols_map.values():
        d[c] = d[c].apply(fmt)
    st.dataframe(d, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1 · ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 · Accuracy")
st.caption("Answer Accuracy = semantic similarity between the generated answer "
           "and the expert reference answer (deterministic).")
show({"answer_accuracy": "Answer Accuracy"})

# ══════════════════════════════════════════════════════════════════════════════
# 2 · CONTEXTUAL RELEVANCE
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 · Contextual relevance")
st.caption("Context F1@k = macro-averaged harmonic mean of Context Precision@k "
           "(relevant chunks / k) and Evidence Recall@k (silver evidence covered / "
           "total). Baseline = N/A (no retrieval).")
show({
    "context_f1_at_k":        "Context F1@k",
    "context_precision_at_k": "Context Precision@k",
    "evidence_recall_at_k":   "Evidence Recall@k",
    "context_relevance":      "Context Relevance",
})

# ══════════════════════════════════════════════════════════════════════════════
# 3 · FAITHFULNESS
# ══════════════════════════════════════════════════════════════════════════════
st.header("3 · Faithfulness")
st.caption("Faithfulness = mean over answer sentences of the maximum similarity to "
           "any retrieved chunk (deterministic sentence-level grounding). "
           "Baseline = N/A.")
show({"faithfulness": "Faithfulness"})

# ══════════════════════════════════════════════════════════════════════════════
# 4 · Heatmaps — one per construct
# ══════════════════════════════════════════════════════════════════════════════
st.header("4 · Parameter heatmaps (3 × 3)")

def heatmap(col, title):
    if col not in rag.columns:
        return
    z, txt = [], []
    for cs in CHUNK_SIZES:
        rz, rt = [], []
        for k in K_VALUES:
            m = rag[(rag["chunk_size"] == cs) & (rag["top_k"] == k)]
            v = None if m.empty else m[col].iloc[0]
            v = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            rz.append(v if v is not None else np.nan)
            rt.append(fmt(v))
        z.append(rz); txt.append(rt)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"k={k}" for k in K_VALUES], y=[f"{c}c" for c in CHUNK_SIZES],
        text=txt, texttemplate="%{text}", colorscale="Purples", showscale=True))
    fig.update_layout(title=title, height=260, margin=dict(t=40, b=20),
                      font=dict(family="Inter"), paper_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1: heatmap("answer_accuracy", "Accuracy")
with c2: heatmap("context_f1_at_k", "Contextual relevance (Context F1@k)")
c3, _ = st.columns(2)
with c3: heatmap("faithfulness", "Faithfulness")

# ══════════════════════════════════════════════════════════════════════════════
# 5 · Best observed configuration
# ══════════════════════════════════════════════════════════════════════════════
st.header("5 · Summary")
if not rag.empty and "context_f1_at_k" in rag.columns:
    valid = rag.dropna(subset=["context_f1_at_k"])
    if not valid.empty:
        ranked = valid.sort_values("context_f1_at_k", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        st.success(f"**Best observed configuration:** {best['config']} "
                   f"(chunk {int(best['chunk_size'])} chars, k={int(best['top_k'])}) — "
                   f"Context F1@k = {best['context_f1_at_k']:.4f}, "
                   f"Accuracy = {best['answer_accuracy']:.4f}, "
        )
# ══════════════════════════════════════════════════════════════════════════════
# 6 · Export
# ══════════════════════════════════════════════════════════════════════════════
st.header("6 · Export")
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download results CSV", data=load_results().to_csv(index=False),
        file_name="readdocai_results.csv", mime="text/csv", use_container_width=True)
with c2:
    st.download_button("Download per-question CSV", data=load_answers().to_csv(index=False),
        file_name="readdocai_answers.csv", mime="text/csv", use_container_width=True)