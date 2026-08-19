"""
2_Run_Experiments.py — Automated evaluation runner (lean, deterministic).
==========================================================================
ReadDoc AI | MSc Data Science and Analytics

Three constructs (all deterministic, no LLM judge):
  Accuracy             → Answer Accuracy
  Contextual relevance → Context F1@k
  Faithfulness         → sentence-level grounding
"""

import os, sys, time, pickle
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import apply_theme
from src import retriever
from src.llm import generate_answer, generate_baseline, ollama_available
from src import metrics as M
from src.store import init_db, save_answer, save_result, load_results, load_answers
from src.config import EXPERIMENTS, QUESTIONS, OLLAMA_MODEL, CHUNK_SIZES, GROUND_TRUTH

st.set_page_config(page_title="Run Experiments — ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("Run Experiments",
            "Run all 9 + baseline automatically. Deterministic metrics; results persist.")
init_db()

# Path where silver evidence is cached so it survives page navigation
GOLD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "silver_evidence.pkl")

ORDER = ["Baseline"] + [e for e, _, _ in EXPERIMENTS]


def _tok(t):
    return 0 if not t else max(1, round(len(t) / 4))


def load_gold():
    """Return silver evidence from session, or load it from disk if present."""
    if "gold_evidence" in st.session_state:
        return st.session_state["gold_evidence"]
    if os.path.exists(GOLD_PATH):
        try:
            with open(GOLD_PATH, "rb") as f:
                g = pickle.load(f)
            st.session_state["gold_evidence"] = g
            return g
        except Exception:
            return None
    return None


def run_config(config, cs, k, is_base, gold, file_paths, run_id, prog=None):
    """Execute a single configuration end-to-end and persist results."""
    model = retriever.get_embedding_model()
    if not is_base:
        retriever.build_index(file_paths, cs)

    per_q = []
    for i, q in enumerate(QUESTIONS):
        qid = q["id"]

        # retrieval (skip for baseline) + generation, with timing
        if is_base:
            chunks, rms = [], None
            t0 = time.perf_counter()
            ans = generate_baseline(q["text"])
            gs = round(time.perf_counter() - t0, 2)
        else:
            t0 = time.perf_counter()
            full = retriever.search(q["text"], top_k=10)      # top-10 once
            rms = round((time.perf_counter() - t0) * 1000, 2)
            chunks = full[:k]                                 # slice to k (nested)
            t0 = time.perf_counter()
            ans = generate_answer(q["text"], chunks)
            gs = round(time.perf_counter() - t0, 2)

        # encode chunks once, reuse in scoring
        cvecs = (model.encode([c["text"] for c in chunks], batch_size=64,
                              convert_to_numpy=True) if chunks else None)

        # deterministic three-construct scoring
        row = M.score_question(qid, q["text"], ans, chunks, gold,
                               is_baseline=is_base, chunk_vecs=cvecs)
        row.update({"category": GROUND_TRUTH[qid]["category"],
                    "_chunk_size": cs or 0, "_top_k": k or 0,
                    "retrieval_ms": rms, "generation_s": gs,
                    "total_s": round((rms or 0) / 1000 + gs, 2),
                    "n_chunks": len(chunks),
                    "ctx_tokens": _tok(" ".join(c["text"] for c in chunks)),
                    "ans_tokens": _tok(ans)})

        # audit sources: only chunks that covered evidence
        pages = sorted({f"p.{c['page_number']}" for c, f in
                        zip(chunks, row.get("chunk_relevance_flags", [])) if f}) \
            if chunks else []
        save_answer(config, qid, q["text"], ans, ", ".join(pages), row, run_id=run_id)
        per_q.append(row)
        if prog:
            prog.progress((i + 1) / len(QUESTIONS))

    agg = M.aggregate(per_q)
    save_result(config, cs or 0, k or 0, agg, len(per_q), run_id=run_id)
    return agg


# ── Pre-flight ────────────────────────────────────────────────────────────────
ok, msg = ollama_available()
if not ok:
    st.error(msg)
    st.stop()
st.success(msg)

file_paths = st.session_state.get("file_paths")
if not file_paths and not retriever.is_ready():
    st.warning("Upload a document and build indexes on the main **app** page first.")
    st.stop()

# ── Silver evidence (persists across page visits) ─────────────────────────────
st.header("Silver evidence")
existing_gold = load_gold()
if existing_gold:
    st.success(f"Silver evidence ready ({sum(len(v) for v in existing_gold.values())} "
               f"units) — no need to rebuild.")
else:
    st.info("Silver evidence not built yet. Build it once before running experiments.")

if st.button("Build / rebuild silver evidence", use_container_width=True):
    with st.spinner(f"Building at {CHUNK_SIZES[0]}-char chunks..."):
        retriever.build_index(file_paths, CHUNK_SIZES[0])
        gold = M.build_gold_evidence(st.session_state.get("active_chunks", []))
        st.session_state["gold_evidence"] = gold
        os.makedirs(os.path.dirname(GOLD_PATH), exist_ok=True)
        with open(GOLD_PATH, "wb") as f:
            pickle.dump(gold, f)                   # persist to disk
    st.success(f"Silver evidence built: {sum(len(v) for v in gold.values())} units.")

gold_evidence = load_gold()
if not gold_evidence:
    st.warning("Build silver evidence before running experiments.")
    st.stop()

# ── Settings ──────────────────────────────────────────────────────────────────
run_id = st.text_input("Run ID", value="run1",
    help="Label runs (run1, run2...) if you want to keep repeated runs separate.")

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL — Baseline + E1..E9 automatically
# ══════════════════════════════════════════════════════════════════════════════
st.header("Run all configurations")
st.caption("Runs Baseline, then E1 through E9, one after another. "
           "Fully deterministic, ~10–15 minutes on local CPU.")

if st.button("▶▶ RUN ALL (Baseline + E1–E9)", type="primary", use_container_width=True):
    items = [("Baseline", None, None)] + list(EXPERIMENTS)
    op = st.progress(0.0, text="Starting...")
    log = st.empty()
    summary_rows = []
    for idx, (name, cs, k) in enumerate(items):
        log.info(f"Running {name} ({idx+1}/{len(items)})...")
        cp = st.progress(0.0)
        try:
            agg = run_config(name, cs, k, name == "Baseline",
                             gold_evidence, file_paths, run_id, prog=cp)
            summary_rows.append({"Config": name,
                                 "Accuracy": agg.get("answer_accuracy"),
                                 "Context F1@k": agg.get("context_f1_at_k"),
                                 "Faithfulness": agg.get("faithfulness")})
        except Exception as e:
            log.error(f"{name} failed: {e}")
        cp.empty()
        op.progress((idx + 1) / len(items), text=f"Completed {idx+1}/{len(items)}")
    op.progress(1.0, text="All configurations complete ✓")
    log.success("All experiments finished. See the results below and on the Results page.")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    st.balloons()

# ══════════════════════════════════════════════════════════════════════════════
# RUN ONE — single configuration
# ══════════════════════════════════════════════════════════════════════════════
st.header("Run a single configuration")
done_results = load_results(run_id=run_id)
done = set(done_results["config"].tolist()) if not done_results.empty else set()

options = ["Baseline — no document context"] + \
          [f"{e} — chunk {c} chars, k={k}" for e, c, k in EXPERIMENTS]
choice = st.selectbox("Configuration", options, index=1)
is_baseline = choice.startswith("Baseline")
if is_baseline:
    config, chunk_size, top_k = "Baseline", None, None
else:
    i = options.index(choice) - 1
    config, chunk_size, top_k = EXPERIMENTS[i]

if config in done:
    st.info(f"{config} already run for {run_id} — running again overwrites it.")

if st.button(f"▶ Run {config}", use_container_width=True):
    prog = st.progress(0.0, text=config)
    agg = run_config(config, chunk_size, top_k, is_baseline,
                     gold_evidence, file_paths, run_id, prog=prog)
    prog.empty()
    st.success(f"✓ {config} complete.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{agg['answer_accuracy']:.4f}" if agg.get('answer_accuracy') is not None else "N/A")
    c2.metric("Context F1@k", f"{agg['context_f1_at_k']:.4f}" if agg.get('context_f1_at_k') is not None else "N/A")
    c3.metric("Faithfulness", f"{agg['faithfulness']:.4f}" if agg.get('faithfulness') is not None else "N/A")

# ══════════════════════════════════════════════════════════════════════════════
# COMPLETED EXPERIMENTS — results + Q&A under each config (from database)
# ══════════════════════════════════════════════════════════════════════════════
st.header("Completed experiments")
st.caption("Loaded from the database — your finished runs persist across page visits.")

all_results = load_results()
if all_results.empty:
    st.info("No experiments completed yet. Click RUN ALL above to start.")
else:
    all_results = all_results.copy()
    all_results["_o"] = all_results["config"].apply(
        lambda c: ORDER.index(c) if c in ORDER else 99)
    all_results = all_results.sort_values("_o").drop(columns="_o")

    st.success(f"{len(all_results)} configuration(s) completed and saved.")

    # summary table
    summary = all_results[["config", "answer_accuracy", "context_f1_at_k",
                           "faithfulness"]].copy()
    summary.columns = ["Config", "Accuracy", "Context F1@k", "Faithfulness"]
    for c in ["Accuracy", "Context F1@k", "Faithfulness"]:
        summary[c] = summary[c].apply(
            lambda v: "N/A" if v is None or pd.isna(v) else f"{v:.4f}")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    # per-config expandable Q&A
    st.subheader("Questions & answers under each experiment")
    for cfg in all_results["config"].tolist():
        row = all_results[all_results["config"] == cfg].iloc[0]
        acc, cf1, fth = row["answer_accuracy"], row["context_f1_at_k"], row["faithfulness"]
        header = f"{cfg}"
        if acc is not None and not pd.isna(acc):
            header += f"  —  Accuracy {acc:.3f}"
        if cf1 is not None and not pd.isna(cf1):
            header += f"  ·  Context F1 {cf1:.3f}"
        if fth is not None and not pd.isna(fth):
            header += f"  ·  Faithfulness {fth:.3f}"

        with st.expander(header):
            answers = load_answers(config=cfg)
            if answers.empty:
                st.caption("No stored answers for this configuration.")
            else:
                for _, a in answers.iterrows():
                    st.markdown(f"**Q{a['question_id']} ({a['category']}):** {a['question']}")
                    st.markdown(a["answer"])
                    bits = []
                    if a.get("answer_accuracy") is not None and not pd.isna(a["answer_accuracy"]):
                        bits.append(f"Accuracy {a['answer_accuracy']:.3f}")
                    if a.get("context_f1_at_k") is not None and not pd.isna(a["context_f1_at_k"]):
                        bits.append(f"Context F1 {a['context_f1_at_k']:.3f}")
                    if a.get("faithfulness") is not None and not pd.isna(a["faithfulness"]):
                        bits.append(f"Faithfulness {a['faithfulness']:.3f}")
                    if a.get("sources") and "could not find" not in str(a["answer"]).lower():
                        bits.append(f"Sources: {a['sources']}")
                    if bits:
                        st.caption(" · ".join(bits))
                    st.divider()