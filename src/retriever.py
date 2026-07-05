"""
retriever.py — Embeddings + FAISS vector search.
=================================================
One cached index per (document set, chunk size). Exact L2 search
(IndexFlatL2) guarantees deterministic, reproducible retrieval —
the same query always returns the same chunks.
"""

import os, pickle, hashlib
import numpy as np
import streamlit as st

from src.config import EMBEDDING_MODEL
from src.ingest import extract_pages, chunk_pages

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _cache_key(file_paths: list, chunk_size: int) -> str:
    h = hashlib.md5("|".join(sorted(file_paths)).encode()).hexdigest()[:10]
    return f"{chunk_size}_{h}"


def build_index(file_paths: list, chunk_size: int) -> dict:
    """Build (or load from cache) the FAISS index for one chunk size."""
    import faiss

    key        = _cache_key(file_paths, chunk_size)
    index_path = os.path.join(DATA_DIR, f"faiss_{key}.index")
    chunks_path = os.path.join(DATA_DIR, f"chunks_{key}.pkl")

    # Cached → instant load
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        st.session_state["active_index"]  = index
        st.session_state["active_chunks"] = chunks
        st.session_state["active_key"]    = key
        return {"chunks": len(chunks), "cached": True}

    # Fresh build
    pages  = []
    for fp in file_paths:
        pages.extend(extract_pages(fp))
    chunks = chunk_pages(pages, chunk_size)
    if not chunks:
        raise ValueError("No text extracted from the uploaded document(s).")

    model      = get_embedding_model()
    embeddings = model.encode([c["text"] for c in chunks],
                              show_progress_bar=False, batch_size=64)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    st.session_state["active_index"]  = index
    st.session_state["active_chunks"] = chunks
    st.session_state["active_key"]    = key
    return {"chunks": len(chunks), "cached": False}


def search(question: str, top_k: int) -> list:
    """Return top-k chunks for a question from the active index."""
    index  = st.session_state.get("active_index")
    chunks = st.session_state.get("active_chunks")
    if index is None or not chunks:
        return []
    q_vec = get_embedding_model().encode([question]).astype("float32")
    _, ids = index.search(q_vec, min(top_k, len(chunks)))
    return [chunks[i] for i in ids[0] if 0 <= i < len(chunks)]


def is_ready() -> bool:
    return st.session_state.get("active_index") is not None
