"""
2_EDA.py — Exploratory analysis of the uploaded document.
==========================================================
Validates preprocessing before any evaluation: page/chunk statistics,
chunk-length distributions per chunk size, and word frequency.
"""

import os, sys, re
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import apply_theme, PRIMARY, ACCENT
from src.ingest import extract_pages, chunk_pages
from src.config import CHUNK_SIZES

st.set_page_config(page_title="EDA — ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("Exploratory Data Analysis",
            "Document and chunk statistics — validates preprocessing before evaluation.")

paths = st.session_state.get("file_paths") or []
if not paths:
    st.info("Upload a document on the **app** page first — EDA runs on the uploaded file.")
    st.stop()

STOPWORDS = set("""the a an and or of to in is are was were be been being for on with as by
at from that this these those it its which can may will would should not no if then than
such other into each we you they he she i our their his her them us also more most very
have has had do does did but about between within where when what how there here over
""".split())


@st.cache_data(show_spinner="Analysing document...")
def analyse(file_paths: tuple):
    pages = []
    for fp in file_paths:
        pages.extend(extract_pages(fp))
    page_lengths = [len(p["text"]) for p in pages]

    per_size = {}
    for cs in CHUNK_SIZES:
        chunks = chunk_pages(pages, cs)
        per_size[cs] = [len(c["text"]) for c in chunks]

    words = re.findall(r"[a-zA-Z]{3,}", " ".join(p["text"] for p in pages).lower())
    words = [w for w in words if w not in STOPWORDS]
    top_words = Counter(words).most_common(20)

    return page_lengths, per_size, top_words, len(words)


page_lengths, per_size, top_words, total_words = analyse(tuple(paths))

# ── Headline stats ────────────────────────────────────────────────────────────
st.header("Document overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages with text", f"{len(page_lengths):,}")
c2.metric("Total characters", f"{sum(page_lengths):,}")
c3.metric("Total words (cleaned)", f"{total_words:,}")
c4.metric("Mean page length", f"{int(sum(page_lengths)/max(len(page_lengths),1)):,} chars")

# ── Chunk counts per size ─────────────────────────────────────────────────────
st.header("Chunk counts per chunk size")
st.caption("Smaller chunks → more, narrower segments. Larger chunks → fewer, broader segments. "
           "This inverse relationship is the mechanism behind the precision/context trade-off.")
counts = {cs: len(v) for cs, v in per_size.items()}
fig1 = go.Figure(go.Bar(
    x=[f"{cs} chars" for cs in counts], y=list(counts.values()),
    marker_color=[PRIMARY, ACCENT, "#B794F6"],
    text=[f"{v:,}" for v in counts.values()], textposition="outside"))
fig1.update_layout(height=320, yaxis_title="Number of chunks",
                   plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                   margin=dict(t=20, b=30), font=dict(family="Inter"))
st.plotly_chart(fig1, use_container_width=True)

# ── Chunk-length distributions ────────────────────────────────────────────────
st.header("Chunk-length distribution per chunk size")
st.caption("Confirms the sliding-window chunker produces consistent segment lengths "
           "at each configuration (final page fragments cause the small left tails).")
fig2 = go.Figure()
for cs, colour in zip(CHUNK_SIZES, [PRIMARY, ACCENT, "#B794F6"]):
    fig2.add_trace(go.Histogram(x=per_size[cs], name=f"{cs} chars",
                                marker_color=colour, opacity=0.65, nbinsx=40))
fig2.update_layout(barmode="overlay", height=360,
                   xaxis_title="Chunk length (characters)", yaxis_title="Frequency",
                   plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                   legend=dict(orientation="h", y=1.08),
                   margin=dict(t=20, b=30), font=dict(family="Inter"))
st.plotly_chart(fig2, use_container_width=True)

# ── Page length distribution ──────────────────────────────────────────────────
st.header("Page-length distribution")
fig3 = go.Figure(go.Histogram(x=page_lengths, nbinsx=40, marker_color=PRIMARY))
fig3.update_layout(height=300, xaxis_title="Page length (characters)",
                   yaxis_title="Pages", plot_bgcolor="#FFFFFF",
                   paper_bgcolor="#FFFFFF", margin=dict(t=20, b=30),
                   font=dict(family="Inter"))
st.plotly_chart(fig3, use_container_width=True)

# ── Word frequency ────────────────────────────────────────────────────────────
st.header("Top 20 content words")
st.caption("Stopwords removed. Confirms the document's subject matter aligns with the test questions.")
wf = pd.DataFrame(top_words, columns=["word", "count"]).iloc[::-1]
fig4 = go.Figure(go.Bar(x=wf["count"], y=wf["word"], orientation="h",
                        marker_color=ACCENT))
fig4.update_layout(height=520, xaxis_title="Frequency",
                   plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                   margin=dict(t=20, b=30), font=dict(family="Inter"))
st.plotly_chart(fig4, use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
st.header("Chunking summary table")
rows = []
for cs in CHUNK_SIZES:
    lens = per_size[cs]
    rows.append({"Chunk size": f"{cs} chars", "Chunks": f"{len(lens):,}",
                 "Mean length": f"{int(sum(lens)/max(len(lens),1)):,}",
                 "Min": min(lens) if lens else 0, "Max": max(lens) if lens else 0})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
