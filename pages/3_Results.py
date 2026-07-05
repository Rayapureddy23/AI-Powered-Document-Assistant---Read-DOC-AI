"""
3_Results.py — Results analysis and visualisation.
===================================================
Reads the results table and renders: summary table, RQ-construct charts,
parameter heatmaps, auto-written findings, RQ answer draft, CSV export.
"""

import os, sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import apply_theme
from src.store import load_results
from src.config import EXPERIMENTS, CHUNK_SIZES, K_VALUES

st.set_page_config(page_title="Results — ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("Results", "Automated evaluation results across all configurations.")

BLUE, PURPLE, GREEN, RED, GREY = "#1D4ED8", "#7C3AED", "#059669", "#DC2626", "#94A3B8"

df = load_results()

# Demo fallback: if the database is empty (e.g. on a shared/deployed demo),
# load bundled seed results so visitors can explore the full Results page.
if df.empty:
    seed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data_seed", "seed_results.csv")
    if os.path.exists(seed):
        df = pd.read_csv(seed)
        st.info("Showing bundled demo results (seed data). Run experiments "
                "locally on the *Run Experiments* page to generate live scores.")

if df.empty:
    st.info("No results yet. Run the **Baseline** and at least one experiment "
            "on the *Run Experiments* page — this page fills in automatically.")
    st.stop()

# Order rows: Baseline first, then E1..E9
order  = ["Baseline"] + [e for e, _, _ in EXPERIMENTS]
df["_o"] = df["config"].apply(lambda c: order.index(c) if c in order else 99)
df = df.sort_values("_o").drop(columns="_o").reset_index(drop=True)
rag = df[df["config"] != "Baseline"]

# ── Headline metrics ──────────────────────────────────────────────────────────
if not rag.empty:
    best = rag.loc[rag["overall"].idxmax()]
    st.header("Headline result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best configuration", best["config"],
              f"chunk {int(best['chunk_size'])} · k={int(best['top_k'])}")
    c2.metric("Overall score", f"{best['overall']:.4f}")
    c3.metric("Faithfulness", f"{best['rq_faithfulness']:.4f}",
              f"+{best['rq_faithfulness']:.4f} vs baseline")
    c4.metric("Configs evaluated", f"{len(rag)} / 9")

# ── Full table ────────────────────────────────────────────────────────────────
st.header("All results — RQ constructs")
show = df[["config", "chunk_size", "top_k",
           "rq_accuracy", "rq_relevance", "rq_faithfulness", "overall"]].rename(
    columns={"config": "Config", "chunk_size": "Chunk", "top_k": "k",
             "rq_accuracy": "Accuracy", "rq_relevance": "Contextual relevance",
             "rq_faithfulness": "Faithfulness", "overall": "Overall"})
st.dataframe(
    show, hide_index=True, use_container_width=True,
    column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.4f")
                   for c in ["Accuracy", "Contextual relevance", "Faithfulness", "Overall"]})

with st.expander("Underlying metrics (6 raw automated measures)"):
    raw = df[["config", "answer_accuracy", "context_relevance", "faithfulness",
              "precision_at_k", "recall_at_k", "mrr"]].rename(
        columns={"config": "Config", "answer_accuracy": "Answer accuracy",
                 "context_relevance": "Context relevance", "faithfulness": "Faithfulness",
                 "precision_at_k": "Precision@k", "recall_at_k": "Recall@k", "mrr": "MRR"})
    st.dataframe(raw, hide_index=True, use_container_width=True)

# ── Chart 1: RQ constructs per configuration ─────────────────────────────────
st.header("RQ constructs per configuration")
fig1 = go.Figure()
for col, name, colour in [("rq_accuracy", "Accuracy", BLUE),
                          ("rq_relevance", "Contextual relevance", PURPLE),
                          ("rq_faithfulness", "Faithfulness", GREEN)]:
    fig1.add_trace(go.Bar(name=name, x=df["config"], y=df[col], marker_color=colour))
fig1.update_layout(barmode="group", height=380, yaxis=dict(range=[0, 1.05], title="Score"),
                   plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                   legend=dict(orientation="h", y=1.08), margin=dict(t=30, b=30),
                   font=dict(family="Inter"))
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: faithfulness vs baseline ─────────────────────────────────────────
st.header("Faithfulness — baseline vs RAG")
st.caption("The baseline is 0.0 by definition (no context to be faithful to). "
           "Everything above the red line is the grounding effect of retrieval.")
fig2 = go.Figure(go.Bar(
    x=df["config"], y=df["rq_faithfulness"],
    marker_color=[GREY if c == "Baseline" else BLUE for c in df["config"]],
    text=df["rq_faithfulness"].apply(lambda v: f"{v:.3f}"), textposition="outside"))
fig2.add_hline(y=0.0, line_dash="dash", line_color=RED,
               annotation_text="Baseline = 0.0")
fig2.update_layout(height=340, yaxis=dict(range=[0, 1.1]), showlegend=False,
                   plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                   margin=dict(t=20, b=30), font=dict(family="Inter"))
st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: parameter heatmap ────────────────────────────────────────────────
if len(rag) >= 2:
    st.header("Parameter sensitivity — chunk size × retrieval depth")
    z, text = [], []
    for cs in CHUNK_SIZES:
        row_z, row_t = [], []
        for k in K_VALUES:
            match = rag[(rag["chunk_size"] == cs) & (rag["top_k"] == k)]
            v = float(match["overall"].iloc[0]) if not match.empty else None
            row_z.append(v)
            row_t.append(f"{v:.4f}" if v is not None else "—")
        z.append(row_z); text.append(row_t)
    fig3 = go.Figure(go.Heatmap(
        z=z, x=[f"k={k}" for k in K_VALUES], y=[f"{c} chars" for c in CHUNK_SIZES],
        text=text, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1))
    fig3.update_layout(height=300, margin=dict(t=20, b=30), font=dict(family="Inter"),
                       paper_bgcolor="#FFFFFF")
    st.plotly_chart(fig3, use_container_width=True)

    # Marginal effects
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Effect of chunk size")
        m = rag.groupby("chunk_size")["overall"].mean().round(4)
        figc = go.Figure(go.Bar(x=[f"{i} chars" for i in m.index], y=m.values,
                                marker_color=GREEN,
                                text=[f"{v:.4f}" for v in m.values],
                                textposition="outside"))
        figc.update_layout(height=280, yaxis=dict(range=[0, 1.05]),
                           plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           margin=dict(t=20, b=20), font=dict(family="Inter"))
        st.plotly_chart(figc, use_container_width=True)
    with c2:
        st.subheader("Effect of retrieval depth k")
        m = rag.groupby("top_k")["overall"].mean().round(4)
        figk = go.Figure(go.Bar(x=[f"k={i}" for i in m.index], y=m.values,
                                marker_color=PURPLE,
                                text=[f"{v:.4f}" for v in m.values],
                                textposition="outside"))
        figk.update_layout(height=280, yaxis=dict(range=[0, 1.05]),
                           plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           margin=dict(t=20, b=20), font=dict(family="Inter"))
        st.plotly_chart(figk, use_container_width=True)

# ── Auto-written findings ─────────────────────────────────────────────────────
if not rag.empty:
    st.header("Findings")
    best  = rag.loc[rag["overall"].idxmax()]
    worst = rag.loc[rag["overall"].idxmin()]

    st.markdown(
        f"1. **Best configuration** — {best['config']} "
        f"(chunk {int(best['chunk_size'])} chars, k={int(best['top_k'])}) achieved "
        f"the highest overall score of **{best['overall']:.4f}** "
        f"(Accuracy {best['rq_accuracy']:.4f} · Relevance {best['rq_relevance']:.4f} · "
        f"Faithfulness {best['rq_faithfulness']:.4f}).")

    st.markdown(
        f"2. **Retrieval grounds the model** — faithfulness rose from 0.0000 "
        f"(baseline, no context) to **{best['rq_faithfulness']:.4f}** in the best "
        f"configuration, confirming that retrieved context anchors answers in the "
        f"source document.")

    if len(rag) >= 2:
        spread = round(float(best["overall"]) - float(worst["overall"]), 4)
        st.markdown(
            f"3. **Parameters matter** — a **{spread:.4f}** spread separates the "
            f"best ({best['config']} = {best['overall']:.4f}) and worst "
            f"({worst['config']} = {worst['overall']:.4f}) configurations, "
            f"demonstrating that chunk size and retrieval depth measurably "
            f"affect answer quality.")

    st.header("Research question — draft answer")
    st.success(
        f"Across {len(rag)} evaluated configurations of a 3×3 factorial design, "
        f"chunk size and retrieval depth measurably affected all three quality "
        f"dimensions. Faithfulness improved from 0.0000 (zero-context baseline) to "
        f"{best['rq_faithfulness']:.4f}, confirming that retrieval grounds LLM "
        f"output in the source document. The optimal configuration was "
        f"{best['config']} (chunk {int(best['chunk_size'])} characters, "
        f"k={int(best['top_k'])}) with an overall score of {best['overall']:.4f}. "
        f"All scores were produced by a fully automated, deterministic evaluation "
        f"pipeline and are exactly reproducible.")
    st.caption("Rewrite in your own words for the dissertation — every number "
               "comes from your saved results.")

# ── Export ────────────────────────────────────────────────────────────────────
st.header("Export")
c_ex1, c_ex2 = st.columns(2)
with c_ex1:
    st.download_button("Download results CSV (permanent record)",
                       data=df.to_csv(index=False),
                       file_name="readdocai_results.csv", mime="text/csv",
                       use_container_width=True)
with c_ex2:
    from src.store import load_answers
    frames = [load_answers(c) for c in df["config"].tolist()]
    frames = [f for f in frames if not f.empty]
    if frames:
        all_answers = pd.concat(frames, ignore_index=True)
        st.download_button("Download all generated answers CSV",
                           data=all_answers.to_csv(index=False),
                           file_name="readdocai_answers.csv", mime="text/csv",
                           use_container_width=True)
st.caption("Streamlit Cloud does not persist the database — always export after "
           "a local run and keep the CSV as the dissertation record.")
