"""
4_Run_Experiments.py — One-click fully automated evaluation.
=============================================================
Per configuration:  generate 10 answers (Ollama) → score 6 metrics
(deterministic embeddings) → save. No manual input anywhere.
"""

import os, sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import apply_theme
from src import retriever
from src.llm import generate_answer, generate_baseline, ollama_available
from src.metrics import score_question, aggregate
from src.store import init_db, save_answer, save_result, load_results, load_answers
from src.config import EXPERIMENTS, QUESTIONS, OLLAMA_MODEL

st.set_page_config(page_title="Run Experiments — ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("Run Experiments",
            "Fully automated: generate answers → compute metrics → save. Zero manual scoring.")
init_db()

REFUSAL_MARK = "could not find"


def is_refusal(text: str) -> bool:
    return REFUSAL_MARK in (text or "").lower()


# ── Pre-flight checks ─────────────────────────────────────────────────────────
ok, msg = ollama_available()
if not ok:
    st.error(msg)
    st.stop()
st.success(msg)

file_paths = st.session_state.get("file_paths")
if not file_paths and not retriever.is_ready():
    st.warning("Upload a document and build indexes on the main **app** page first.")
    st.stop()

# ── Coverage ──────────────────────────────────────────────────────────────────
st.header("Coverage")
results   = load_results()
done      = set(results["config"].tolist()) if not results.empty else set()
cov_cols  = st.columns(5)
all_items = [("Baseline", None, None)] + list(EXPERIMENTS)
for i, (name, cs, k) in enumerate(all_items):
    label = "✅" if name in done else "⬜"
    cov_cols[i % 5].markdown(f"{label} **{name}**" +
                             (f"<br><small>{cs} / k={k}</small>" if cs else ""),
                             unsafe_allow_html=True)

# ── Selector ──────────────────────────────────────────────────────────────────
st.header("Run")
options = ["Baseline — no document context"] + \
          [f"{e} — chunk {c} chars, k={k}" for e, c, k in EXPERIMENTS]
choice  = st.selectbox("Configuration", options, index=5)  # default E5

is_baseline = choice.startswith("Baseline")
if is_baseline:
    config, chunk_size, top_k = "Baseline", None, None
else:
    idx = options.index(choice) - 1
    config, chunk_size, top_k = EXPERIMENTS[idx]

if config in done:
    st.info(f"{config} already evaluated — running again overwrites its scores.")

parallel = st.toggle(
    "⚡ Parallel answer generation (3 workers)",
    value=True,
    help="Sends 3 questions to Ollama concurrently. The LLM is used ONLY for "
         "generating answers — all metric scoring stays deterministic and "
         "sequential. Set OLLAMA_NUM_PARALLEL=3 before starting Ollama for the "
         "full speed-up.")

est_min = (2 if parallel else 4) if not is_baseline else (2 if parallel else 3)
st.caption(f"Estimated time: ~{est_min}–{est_min + 3} minutes "
           f"({OLLAMA_MODEL} on local CPU, 10 answers + instant metrics).")

if st.button(f"▶ Run {config}", type="primary", use_container_width=True):

    # Step 1 — index (skip for baseline)
    if not is_baseline:
        with st.spinner(f"Loading index for chunk size {chunk_size}..."):
            retriever.build_index(file_paths, chunk_size)

    # Step 2 — retrieval first (sequential), then generation
    prog   = st.progress(0)
    status = st.empty()

    status.info("Retrieving context for all questions...")
    retrieved = {}
    for q in QUESTIONS:
        retrieved[q["id"]] = [] if is_baseline else retriever.search(q["text"], top_k=top_k)
    prog.progress(0.1)

    # Generation — parallel or sequential (LLM used ONLY here)
    answers = {}
    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        status.info("Generating 10 answers in parallel (3 workers)...")

        def _gen(q):
            if is_baseline:
                return q["id"], generate_baseline(q["text"])
            return q["id"], generate_answer(q["text"], retrieved[q["id"]])

        done_n = 0
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_gen, q) for q in QUESTIONS]
            for fut in as_completed(futures):
                qid, ans = fut.result()
                answers[qid] = ans
                done_n += 1
                prog.progress(0.1 + 0.7 * done_n / len(QUESTIONS))
                status.info(f"Generated {done_n}/10 answers...")
    else:
        for i, q in enumerate(QUESTIONS):
            status.info(f"Q{q['id']}/10 — generating answer...")
            answers[q["id"]] = (generate_baseline(q["text"]) if is_baseline
                                else generate_answer(q["text"], retrieved[q["id"]]))
            prog.progress(0.1 + 0.7 * (i + 1) / len(QUESTIONS))

    # Step 3 — deterministic scoring (no LLM involved)
    status.info("Scoring all answers (deterministic metrics)...")
    per_question = []
    for i, q in enumerate(QUESTIONS):
        chunks = retrieved[q["id"]]
        answer = answers[q["id"]]
        m = score_question(q["id"], q["text"], answer, chunks,
                           is_baseline=is_baseline)
        per_question.append(m)

        # Hide source pages when the answer is a refusal (out-of-scope questions)
        if is_refusal(answer) or not chunks:
            sources = ""
        else:
            sources = ", ".join(sorted({f"p.{c['page_number']}" for c in chunks}))

        save_answer(config, q["id"], q["text"], answer, sources, m)
        prog.progress(0.8 + 0.2 * (i + 1) / len(QUESTIONS))

    prog.empty(); status.empty()

    # Step 4 — aggregate + persist
    agg = aggregate(per_question)
    save_result(config, chunk_size or 0, top_k or 0, agg, len(per_question))

    st.success(f"✓ {config} complete — all metrics computed automatically.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RQ · Accuracy",     f"{agg['rq_accuracy']:.4f}")
    c2.metric("RQ · Relevance",    f"{agg['rq_relevance']:.4f}")
    c3.metric("RQ · Faithfulness", f"{agg['rq_faithfulness']:.4f}")
    c4.metric("Overall",           f"{agg['overall']:.4f}")

# ── Inspect answers for the selected configuration ────────────────────────────
st.header("Generated answers (audit trail)")
df_ans = load_answers(config)
if df_ans.empty:
    st.caption("No answers stored yet for this configuration.")
else:
    for _, row in df_ans.iterrows():
        with st.expander(f"Q{row['question_id']} — {row['question'][:80]}"):
            st.markdown(row["answer"])
            # Only show sources if there are any AND the answer isn't a refusal
            if row["sources"] and not is_refusal(row["answer"]):
                st.caption(f"Sources: {row['sources']}")