"""
app.py — ReadDoc AI chat interface.
Upload a document → build indexes → select chunk size / k → ask questions.
"""

import os, sys, shutil
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import retriever
from src.llm import stream_answer, ollama_available
from src.config import CHUNK_SIZES, K_VALUES, OLLAMA_MODEL
from src.ui import apply_theme
from src.store import init_db

st.set_page_config(page_title="ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("ReadDoc AI",
            "RAG document Q&A — upload, build indexes, ask questions with page citations.",
            show_status=False)   # this page has its own richer sidebar
init_db()
st.session_state.setdefault("built_sizes", set())
st.session_state.setdefault("chat_chunk", 600)
st.session_state.setdefault("chat_k", 5)

UPLOAD_DIR = os.path.join("data", "uploads")

# ══════════════════════════ SIDEBAR ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 ReadDoc AI")
    st.caption("Retrieval-Augmented Generation")
    st.divider()

    # ── 1 · Upload ────────────────────────────────────────────────────────────
    st.markdown("### 1 · Upload document")
    uploads = st.file_uploader("PDF / HTML / TXT",
                               type=["pdf", "html", "htm", "txt"],
                               accept_multiple_files=True,
                               label_visibility="collapsed")
    if uploads:
        new_names = {uf.name for uf in uploads}
        old_names = {os.path.basename(p)
                     for p in st.session_state.get("file_paths", [])}
        if new_names != old_names:
            # Fresh upload → clear previous uploads and stale indexes
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            paths = []
            for uf in uploads:
                p = os.path.join(UPLOAD_DIR, uf.name)
                with open(p, "wb") as f:
                    f.write(uf.getbuffer())
                paths.append(os.path.abspath(p))
            st.session_state["file_paths"]  = paths
            st.session_state["built_sizes"] = set()
            for k in ("active_index", "active_chunks", "active_key"):
                st.session_state.pop(k, None)

    paths = st.session_state.get("file_paths") or []
    if paths:
        for p in paths:
            st.success(os.path.basename(p))
    else:
        st.info("No document uploaded yet.")

    # ── 2 · Build indexes ─────────────────────────────────────────────────────
    st.markdown("### 2 · Build indexes")
    if st.button("⚙ Build all indexes (300 / 600 / 1000)",
                 type="primary", use_container_width=True,
                 disabled=not paths):
        for cs in CHUNK_SIZES:
            with st.spinner(f"Building chunk size {cs}..."):
                info = retriever.build_index(paths, cs)
            st.session_state["built_sizes"].add(cs)
            st.success(f"{cs} chars — {info['chunks']:,} chunks")

    built = st.session_state["built_sizes"]
    if built:
        st.caption("Built: " + ", ".join(str(b) for b in sorted(built)) + " chars")

    # ── 3 · Chat retrieval settings (buttons) ─────────────────────────────────
    st.markdown("### 3 · Retrieval settings")
    st.caption("Chunk size")
    chunk_choice = st.segmented_control(
        "Chunk size", options=CHUNK_SIZES,
        default=st.session_state["chat_chunk"],
        label_visibility="collapsed")
    if chunk_choice:
        st.session_state["chat_chunk"] = chunk_choice

    st.caption("Retrieval depth k")
    k_choice = st.segmented_control(
        "Retrieval depth", options=K_VALUES,
        default=st.session_state["chat_k"],
        label_visibility="collapsed")
    if k_choice:
        st.session_state["chat_k"] = k_choice

    # keep the active index in sync with the selected chunk size
    if paths and st.session_state["chat_chunk"] in built:
        retriever.build_index(paths, st.session_state["chat_chunk"])  # cached, instant

    st.divider()

    # ── Status footer ─────────────────────────────────────────────────────────
    ok, msg = ollama_available()
    (st.success if ok else st.warning)(msg)
    st.caption(f"Embeddings: all-MiniLM-L6-v2 · Index: FAISS")

    if st.button("🗑 Reset session", use_container_width=True):
        shutil.rmtree("data", ignore_errors=True)
        for k in ("file_paths", "built_sizes", "chat",
                  "active_index", "active_chunks", "active_key"):
            st.session_state.pop(k, None)
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════ MAIN — CHAT ══════════════════════════════════════
if not paths:
    st.markdown("#### Welcome")
    st.markdown(
        "1. **Upload** a document in the sidebar\n"
        "2. **Build** the indexes (one click, ~4 minutes first time)\n"
        "3. **Select** chunk size and retrieval depth\n"
        "4. **Ask** questions — answers cite the source pages")
    st.stop()

st.caption(f"Chat retrieval → chunk **{st.session_state['chat_chunk']}** chars · "
           f"k = **{st.session_state['chat_k']}**")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your document..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ok, _ = ollama_available()
        if not retriever.is_ready():
            reply = "Please build the indexes first (sidebar, step 2)."
            st.markdown(reply)
        elif not ok:
            reply = ("Demo note: the local LLM (Ollama) is not available on this "
                     "server, so live answers are disabled. Methodology, EDA and "
                     "Results pages are fully interactive.")
            st.markdown(reply)
        else:
            is_summary = any(w in prompt.lower() for w in
                             ("summar", "overview", "what is this document",
                              "about this document", "main topics"))
            if is_summary:
                # Sample evenly across the whole document, not just top-k
                all_chunks = st.session_state.get("active_chunks", [])
                step   = max(len(all_chunks) // 12, 1)
                chunks = all_chunks[::step][:12]
                from src.llm import generate_summary
                with st.spinner("Summarising the document..."):
                    reply = generate_summary(prompt, chunks)
                st.markdown(reply)
            else:
                chunks  = retriever.search(prompt, top_k=st.session_state["chat_k"])
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.chat[:-1]][-6:]
                reply = st.write_stream(stream_answer(prompt, chunks, history))

            # Show sources ONLY when the model actually answered
            refused = "could not find this in your uploaded documents" in reply.lower()
            if chunks and not refused:
                pages = sorted({f"p.{c['page_number']}" for c in chunks})
                st.caption("Sources: " + ", ".join(pages))
    st.session_state.chat.append({"role": "assistant", "content": reply})