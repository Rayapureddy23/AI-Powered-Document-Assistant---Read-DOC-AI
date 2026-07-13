"""2_Run_Experiments.py — Fast deterministic runner. No LLM judge."""
import os, sys, time
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
apply_theme("Run Experiments", "Fast, fully deterministic. Run all 9 + baseline automatically.")
init_db()

def _tok(t): return 0 if not t else max(1, round(len(t) / 4))

def run_config(config, cs, k, is_base, gold, paths, run_id, prog=None):
    model = retriever.get_embedding_model()
    if not is_base:
        retriever.build_index(paths, cs)
    per_q = []
    for i, q in enumerate(QUESTIONS):
        qid = q["id"]
        if is_base:
            chunks, rms = [], None
            t0 = time.perf_counter(); ans = generate_baseline(q["text"])
            gs = round(time.perf_counter() - t0, 2)
        else:
            t0 = time.perf_counter(); full = retriever.search(q["text"], top_k=10)
            rms = round((time.perf_counter() - t0) * 1000, 2)
            chunks = full[:k]
            t0 = time.perf_counter(); ans = generate_answer(q["text"], chunks)
            gs = round(time.perf_counter() - t0, 2)
        cvecs = (model.encode([c["text"] for c in chunks], batch_size=64,
                              convert_to_numpy=True) if chunks else None)
        row = M.score_question(qid, q["text"], ans, chunks, gold,
                               is_baseline=is_base, chunk_vecs=cvecs)
        row.update({"category": GROUND_TRUTH[qid]["category"],
                    "_chunk_size": cs or 0, "_top_k": k or 0,
                    "retrieval_ms": rms, "generation_s": gs,
                    "total_s": round((rms or 0)/1000 + gs, 2),
                    "n_chunks": len(chunks),
                    "ctx_tokens": _tok(" ".join(c["text"] for c in chunks)),
                    "ans_tokens": _tok(ans)})
        pages = sorted({f"p.{c['page_number']}" for c, f in
                        zip(chunks, row.get("chunk_relevance_flags", []))
                        if f}) if chunks else []
        save_answer(config, qid, q["text"], ans, ", ".join(pages), row, run_id=run_id)
        per_q.append(row)
        if prog: prog.progress((i + 1) / len(QUESTIONS))
    agg = M.aggregate(per_q)
    save_result(config, cs or 0, k or 0, agg, len(per_q), run_id=run_id)
    return agg

ok, msg = ollama_available()
if not ok: st.error(msg); st.stop()
st.success(msg)
paths = st.session_state.get("file_paths")
if not paths and not retriever.is_ready():
    st.warning("Upload a document and build indexes on the app page first."); st.stop()

st.header("Silver evidence")
if st.button("Build silver evidence (once)", use_container_width=True):
    with st.spinner("Building..."):
        retriever.build_index(paths, CHUNK_SIZES[0])
        st.session_state["gold_evidence"] = M.build_gold_evidence(
            st.session_state.get("active_chunks", []))
    st.success(f"Built {sum(len(v) for v in st.session_state['gold_evidence'].values())} units.")
gold = st.session_state.get("gold_evidence")
if not gold: st.warning("Build silver evidence first."); st.stop()

run_id = st.text_input("Run ID", value="run1")

st.header("Run all")
st.caption("Runs Baseline + E1–E9 automatically. Fully deterministic, ~10–15 min.")
if st.button("▶▶ RUN ALL (Baseline + E1–E9)", type="primary", use_container_width=True):
    items = [("Baseline", None, None)] + list(EXPERIMENTS)
    op = st.progress(0.0, text="Starting..."); log = st.empty(); summ = []
    for idx, (name, cs, k) in enumerate(items):
        log.info(f"Running {name} ({idx+1}/{len(items)})...")
        cp = st.progress(0.0)
        try:
            agg = run_config(name, cs, k, name == "Baseline", gold, paths, run_id, prog=cp)
            summ.append({"Config": name, "Accuracy": agg["answer_accuracy"],
                         "Context F1@k": agg["context_f1_at_k"],
                         "Faithfulness": agg["faithfulness"]})
        except Exception as e:
            log.error(f"{name} failed: {e}")
        cp.empty()
        op.progress((idx+1)/len(items), text=f"{idx+1}/{len(items)} done")
    op.progress(1.0, text="Complete ✓"); log.success("All experiments finished.")
    st.dataframe(pd.DataFrame(summ), hide_index=True, use_container_width=True)
    st.balloons()