"""
app.py — ReadDoc AI chat interface.
Upload a document → build indexes → chat with page-cited answers.
"""

import os, sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import retriever
from src.llm import stream_answer, ollama_available
from src.config import CHUNK_SIZES, OLLAMA_MODEL
from src.ui import apply_theme
from src.store import init_db

st.set_page_config(page_title="ReadDoc AI", page_icon="📄", layout="wide")
apply_theme("ReadDoc AI",
            "RAG document Q&A — upload a document, build the indexes, ask questions.")
init_db()
st.session_state.setdefault("built_sizes", set())

# ── Sidebar: chunk-size control, upload, index ────────────────────────────────
with st.sidebar:
    st.markdown("#### ⚙️ Chunk size (chat)")
    custom_size = st.number_input(
        "Active / custom chunk size (chars)",
        min_value=100, max_value=3000, step=50,
        value=st.session_state.get("active_chunk_size", 600),
        help="Chat retrieval uses this chunk size. Standard study sizes "
             "(300 / 600 / 1000) stay untouched — a custom value builds an "
             "extra index without removing anything.")
    st.session_state["active_chunk_size"] = custom_size

    st.markdown("#### 📤 Upload document")
    uploads = st.file_uploader("PDF / HTML / TXT",
                               type=["pdf", "html", "htm", "txt"],
                               accept_multiple_files=True)
    if uploads:
        os.makedirs(os.path.join("data", "uploads"), exist_ok=True)
        paths = []
        for uf in uploads:
            p = os.path.join("data", "uploads", uf.name)
            with open(p, "wb") as f:
                f.write(uf.getbuffer())
            paths.append(os.path.abspath(p))
        st.session_state["file_paths"] = paths

    paths = st.session_state.get("file_paths") or []

    if paths:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Build standard\n(300/600/1000)", type="primary",
                         use_container_width=True):
                for cs in CHUNK_SIZES:
                    with st.spinner(f"Chunk {cs}..."):
                        info = retriever.build_index(paths, cs)
                    st.session_state["built_sizes"].add(cs)
                    st.success(f"{cs}: {info['chunks']} chunks")
        with col_b:
            if st.button(f"Build custom\n({custom_size})",
                         use_container_width=True):
                with st.spinner(f"Chunk {custom_size}..."):
                    info = retriever.build_index(paths, custom_size)
                st.session_state["built_sizes"].add(custom_size)
                st.success(f"{custom_size}: {info['chunks']} chunks")

        # keep chat on the selected size if that index exists
        if custom_size in st.session_state["built_sizes"]:
            retriever.build_index(paths, custom_size)   # instant cache load

    st.divider()
    ok, msg = ollama_available()
    (st.success if ok else st.warning)(msg)
    st.caption(f"LLM: {OLLAMA_MODEL} · Embeddings: all-MiniLM-L6-v2 · FAISS")

    if st.button("🗑 Reset all data", use_container_width=True):
        import shutil
        shutil.rmtree("data", ignore_errors=True)
        for k in ("file_paths", "built_sizes", "chat", "active_index",
                  "active_chunks", "active_key"):
            st.session_state.pop(k, None)
        st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
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
            reply = "Please upload a document and build the indexes first (sidebar)."
            st.markdown(reply)
        elif not ok:
            reply = ("Demo note: the local LLM (Ollama) is not available on this "
                     "server, so live answers are disabled here. The Methodology "
                     "and Results pages are fully interactive.")
            st.markdown(reply)
        else:
            chunks  = retriever.search(prompt, top_k=5)
            history = [{"role": m["role"], "content": m["content"]}
                       for m in st.session_state.chat[:-1]][-6:]
            reply = st.write_stream(stream_answer(prompt, chunks, history))
            if chunks:
                pages = sorted({f"p.{c['page_number']}" for c in chunks})
                st.caption("Sources: " + ", ".join(pages))
    st.session_state.chat.append({"role": "assistant", "content": reply})
