"""
app.py — ReadDoc AI chat interface.
"""

import os, sys, shutil
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import retriever
from src.llm import stream_answer, generate_summary, ollama_available
from src.config import CHUNK_SIZES, K_VALUES, OLLAMA_MODEL
from src.ui import apply_theme
from src.store import init_db

st.set_page_config(page_title="ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("ReadDoc AI",
            "RAG document Q&A — upload, build indexes, ask questions with page citations.",
            show_status=False)
init_db()
st.session_state.setdefault("built_sizes", set())
st.session_state.setdefault("chat_chunk", 600)
st.session_state.setdefault("chat_k", 5)

UPLOAD_DIR = os.path.join("data", "uploads")

# ══════════════════════════ SIDEBAR ══════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="padding:14px 16px;background:linear-gradient(135deg,#5C2D91,#7C3AED);'
        'border-radius:12px;margin-bottom:14px">'
        '<span style="color:#fff;font-size:1.05rem;font-weight:700">📄 ReadDoc AI</span><br>'
        '<span style="color:#DDD6FE;font-size:0.78rem">Retrieval-Augmented Generation</span>'
        '</div>', unsafe_allow_html=True)

    # ── Document ──────────────────────────────────────────────────────────────
    st.markdown("**Document**")
    uploads = st.file_uploader("Upload", type=["pdf", "html", "htm", "txt"],
                               accept_multiple_files=True,
                               label_visibility="collapsed")
    if uploads:
        new_names = {uf.name for uf in uploads}
        old_names = {os.path.basename(p) for p in st.session_state.get("file_paths", [])}
        if new_names != old_names:
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
    for p in paths:
        st.markdown(f"<div style='background:#EDE9F7;border-radius:8px;padding:6px 10px;"
                    f"font-size:0.82rem;color:#3B2A5E;margin-bottom:4px'>"
                    f"✓ {os.path.basename(p)}</div>", unsafe_allow_html=True)

    if paths:
        built = st.session_state["built_sizes"]
        all_built = set(CHUNK_SIZES) <= built
        if not all_built:
            if st.button("⚙ Build indexes", type="primary", use_container_width=True):
                prog = st.progress(0.0)
                for i, cs in enumerate(CHUNK_SIZES):
                    prog.progress(i / len(CHUNK_SIZES), text=f"Chunk {cs}...")
                    retriever.build_index(paths, cs)
                    st.session_state["built_sizes"].add(cs)
                prog.progress(1.0, text="Done")
                st.rerun()
        else:
            st.markdown("<div style='color:#059669;font-size:0.82rem;font-weight:600'>"
                        "● Indexes ready (300 / 600 / 1000)</div>",
                        unsafe_allow_html=True)

    st.divider()

    # ── Retrieval settings ────────────────────────────────────────────────────
    st.markdown("**Retrieval settings**")
    c = st.segmented_control("Chunk size (chars)", options=CHUNK_SIZES,
                             default=st.session_state["chat_chunk"])
    if c:
        st.session_state["chat_chunk"] = c
    k = st.segmented_control("Retrieval depth (k)", options=K_VALUES,
                             default=st.session_state["chat_k"])
    if k:
        st.session_state["chat_k"] = k

    # Switch active index ONLY when the selection changed (zero-cost otherwise)
    if paths and st.session_state["chat_chunk"] in st.session_state["built_sizes"]:
        retriever.activate_if_needed(paths, st.session_state["chat_chunk"])

    st.divider()

    # ── Status ────────────────────────────────────────────────────────────────
    ok, msg = ollama_available()
    dot = "#059669" if ok else "#D97706"
    st.markdown(f"<div style='font-size:0.78rem;color:#6B7280'>"
                f"<span style='color:{dot}'>●</span> {msg}<br>"
                f"Embeddings: MiniLM-L6-v2 · Index: FAISS</div>",
                unsafe_allow_html=True)

    if st.button("Reset session", use_container_width=True):
        shutil.rmtree("data", ignore_errors=True)
        for key in ("file_paths", "built_sizes", "chat",
                    "active_index", "active_chunks", "active_key"):
            st.session_state.pop(key, None)
        st.cache_resource.clear()
        st.rerun()

# ══════════════════════════ MAIN — CHAT ══════════════════════════════════════
if not paths:
    st.markdown("#### 👋 Welcome")
    st.markdown(
        "1. **Upload** a document in the sidebar\n"
        "2. **Build** the indexes (one click)\n"
        "3. **Select** chunk size and retrieval depth\n"
        "4. **Ask** questions — answers cite the source pages")
    st.stop()

st.caption(f"Retrieval → chunk **{st.session_state['chat_chunk']}** chars · "
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
        chunks = []
        if not retriever.is_ready():
            reply = "Please build the indexes first (sidebar)."
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
                all_chunks = st.session_state.get("active_chunks", [])
                step   = max(len(all_chunks) // 12, 1)
                chunks = all_chunks[::step][:12]
                with st.spinner("Summarising the document..."):
                    reply = generate_summary(prompt, chunks)
                st.markdown(reply)
            else:
                chunks  = retriever.search(prompt, top_k=st.session_state["chat_k"])
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.chat[:-1]][-6:]
                reply = st.write_stream(stream_answer(prompt, chunks, history))

            refused = "could not find this in your uploaded documents" in reply.lower()
            if chunks and not refused:
                pages = sorted({f"p.{c['page_number']}" for c in chunks})
                st.caption("Sources: " + ", ".join(pages))
    st.session_state.chat.append({"role": "assistant", "content": reply})