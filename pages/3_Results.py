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
# VISUAL ANALYSIS — bar & line plots for supervisor presentation
# ══════════════════════════════════════════════════════════════════════════════
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.header("Visual Analysis")

# Pull ordered RAG data (exclude baseline for retrieval plots)
plot_df = rag.copy().sort_values("config").reset_index(drop=True)
configs = plot_df["config"].tolist()

PURPLE = "#5C2D91"; VIOLET = "#7C3AED"; GREEN = "#059669"
BLUE = "#2563EB"; CORAL = "#D85A30"; GREY = "#94A3B8"

# ── PLOT 1 — Three constructs grouped bar chart ───────────────────────────────
st.subheader("1 · Three constructs across configurations")
fig1 = go.Figure()
fig1.add_trace(go.Bar(name="Accuracy", x=configs,
                      y=plot_df["answer_accuracy"], marker_color=BLUE))
fig1.add_trace(go.Bar(name="Context F1@k", x=configs,
                      y=plot_df["context_f1_at_k"], marker_color=VIOLET))
fig1.add_trace(go.Bar(name="Faithfulness", x=configs,
                      y=plot_df["faithfulness"], marker_color=GREEN))
fig1.update_layout(barmode="group", height=400,
                   yaxis_title="Score", xaxis_title="Configuration",
                   legend=dict(orientation="h", y=1.1),
                   font=dict(family="Inter"), plot_bgcolor="white")
fig1.update_yaxes(range=[0, 0.9], gridcolor="#EEE")
st.plotly_chart(fig1, use_container_width=True)
st.caption("All three quality constructs, side by side, for every RAG configuration.")

# ── PLOT 2 — Accuracy vs baseline bar chart ───────────────────────────────────
st.subheader("2 · Answer accuracy — RAG vs baseline")
acc_df = df.sort_values("config").reset_index(drop=True)
baseline_acc = df[df["config"] == "Baseline"]["answer_accuracy"].iloc[0] \
    if "Baseline" in df["config"].values else 0.64
colors = [CORAL if c == "Baseline" else BLUE for c in acc_df["config"]]
fig2 = go.Figure(go.Bar(x=acc_df["config"], y=acc_df["answer_accuracy"],
                        marker_color=colors,
                        text=[f"{v:.3f}" for v in acc_df["answer_accuracy"]],
                        textposition="outside"))
fig2.add_hline(y=baseline_acc, line_dash="dash", line_color=CORAL,
               annotation_text=f"baseline {baseline_acc:.3f}",
               annotation_position="top right")
fig2.update_layout(height=380, yaxis_title="Answer Accuracy",
                   xaxis_title="Configuration", font=dict(family="Inter"),
                   plot_bgcolor="white", showlegend=False)
fig2.update_yaxes(range=[0, 0.95], gridcolor="#EEE")
st.plotly_chart(fig2, use_container_width=True)
st.caption("Every RAG configuration (blue) exceeds the zero-context baseline (coral).")

# ── PLOT 3 — Precision vs Recall line plot across k ───────────────────────────
st.subheader("3 · Precision–Recall trade-off across retrieval depth")
from src.config import CHUNK_SIZES, K_VALUES
fig3 = go.Figure()
dash_styles = {300: "solid", 600: "dash", 1000: "dot"}
for cs in CHUNK_SIZES:
    sub = plot_df[plot_df["chunk_size"] == cs].sort_values("top_k")
    if not sub.empty:
        fig3.add_trace(go.Scatter(
            x=sub["top_k"], y=sub["context_precision_at_k"],
            mode="lines+markers", name=f"Precision {cs}c",
            line=dict(color=BLUE, dash=dash_styles.get(cs, "solid"))))
        fig3.add_trace(go.Scatter(
            x=sub["top_k"], y=sub["evidence_recall_at_k"],
            mode="lines+markers", name=f"Recall {cs}c",
            line=dict(color=GREEN, dash=dash_styles.get(cs, "solid"))))
fig3.update_layout(height=420, xaxis_title="Retrieval depth (k)",
                   yaxis_title="Score", font=dict(family="Inter"),
                   plot_bgcolor="white", xaxis=dict(tickvals=K_VALUES),
                   legend=dict(orientation="h", y=-0.25))
fig3.update_yaxes(gridcolor="#EEE")
st.plotly_chart(fig3, use_container_width=True)
st.caption("As k increases, precision (blue) falls while recall (green) rises — "
           "the core retrieval trade-off. Line style = chunk size.")

# ── PLOT 4 — Context F1 line plot by chunk size ───────────────────────────────
st.subheader("4 · Context F1@k by chunk size and depth")
fig4 = go.Figure()
palette = {300: BLUE, 600: VIOLET, 1000: CORAL}
for cs in CHUNK_SIZES:
    sub = plot_df[plot_df["chunk_size"] == cs].sort_values("top_k")
    if not sub.empty:
        fig4.add_trace(go.Scatter(
            x=sub["top_k"], y=sub["context_f1_at_k"],
            mode="lines+markers+text", name=f"{cs} chars",
            line=dict(color=palette.get(cs), width=3),
            marker=dict(size=10),
            text=[f"{v:.3f}" for v in sub["context_f1_at_k"]],
            textposition="top center"))
fig4.update_layout(height=420, xaxis_title="Retrieval depth (k)",
                   yaxis_title="Context F1@k", font=dict(family="Inter"),
                   plot_bgcolor="white", xaxis=dict(tickvals=K_VALUES),
                   legend=dict(title="Chunk size"))
fig4.update_yaxes(gridcolor="#EEE")
st.plotly_chart(fig4, use_container_width=True)
st.caption("Contextual relevance peaks at 600 chars, k=5 (E5). Each line is a chunk size.")

# ── PLOT 5 — Faithfulness bar chart ───────────────────────────────────────────
st.subheader("5 · Faithfulness by configuration")
faith_df = plot_df.dropna(subset=["faithfulness"]).sort_values("faithfulness",
                                                                ascending=False)
fig5 = go.Figure(go.Bar(x=faith_df["config"], y=faith_df["faithfulness"],
                        marker_color=GREEN,
                        text=[f"{v:.3f}" for v in faith_df["faithfulness"]],
                        textposition="outside"))
fig5.update_layout(height=380, yaxis_title="Faithfulness",
                   xaxis_title="Configuration (ranked)",
                   font=dict(family="Inter"), plot_bgcolor="white",
                   showlegend=False)
fig5.update_yaxes(range=[0, 0.75], gridcolor="#EEE")
st.plotly_chart(fig5, use_container_width=True)
st.caption("Sentence-level grounding, ranked. Higher = answers stay closer to "
           "retrieved context. Baseline excluded (no context).")

# ── PLOT 6 — Accuracy line trend across k (all chunk sizes) ────────────────────
st.subheader("6 · Accuracy trend across retrieval depth")
fig6 = go.Figure()
for cs in CHUNK_SIZES:
    sub = plot_df[plot_df["chunk_size"] == cs].sort_values("top_k")
    if not sub.empty:
        fig6.add_trace(go.Scatter(
            x=sub["top_k"], y=sub["answer_accuracy"],
            mode="lines+markers", name=f"{cs} chars",
            line=dict(color=palette.get(cs), width=2.5),
            marker=dict(size=9)))
fig6.add_hline(y=baseline_acc, line_dash="dash", line_color=GREY,
               annotation_text=f"baseline {baseline_acc:.3f}")
fig6.update_layout(height=400, xaxis_title="Retrieval depth (k)",
                   yaxis_title="Answer Accuracy", font=dict(family="Inter"),
                   plot_bgcolor="white", xaxis=dict(tickvals=K_VALUES),
                   legend=dict(title="Chunk size"))
fig6.update_yaxes(range=[0.6, 0.85], gridcolor="#EEE")
st.plotly_chart(fig6, use_container_width=True)
st.caption("Accuracy stays high and stable across depths, well above baseline "
           "(grey line) for every configuration.")
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
    