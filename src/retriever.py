"""
retriever.py — Embeddings + FAISS vector search.
=================================================
Three cache layers:
  1. In-memory (st.cache_resource) — index+chunks load ONCE per session
  2. On-disk (data/faiss_*.index)  — survives restarts, skips re-embedding
  3. Embedding model               — loaded once
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


@st.cache_resource(show_spinner=False)
def _load_or_build(key: str, file_paths: tuple, chunk_size: int):
    """Heavy work happens here exactly once per key per session.
    Returns (faiss_index, chunks). Cached in memory by Streamlit."""
    import faiss

    index_path  = os.path.join(DATA_DIR, f"faiss_{key}.index")
    chunks_path = os.path.join(DATA_DIR, f"chunks_{key}.pkl")

    # Disk cache hit → single load into memory for the whole session
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks, True

    # Fresh build (embedding is the slow step — batch 128 for speed)
    pages = []
    for fp in file_paths:
        pages.extend(extract_pages(fp))
    chunks = chunk_pages(pages, chunk_size)
    if not chunks:
        raise ValueError("No text extracted from the uploaded document(s).")

    model = get_embedding_model()
    embeddings = model.encode([c["text"] for c in chunks],
                              show_progress_bar=False,
                              batch_size=128,
                              convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    return index, chunks, False


def build_index(file_paths: list, chunk_size: int) -> dict:
    """Activate the index for one chunk size. Instant if already in memory."""
    key = _cache_key(file_paths, chunk_size)
    index, chunks, cached = _load_or_build(key, tuple(file_paths), chunk_size)
    st.session_state["active_index"]  = index
    st.session_state["active_chunks"] = chunks
    st.session_state["active_key"]    = key
    return {"chunks": len(chunks), "cached": cached}


def activate_if_needed(file_paths: list, chunk_size: int):
    """Switch the active index ONLY when the selection changed — zero cost
    on normal reruns. This replaces the per-rerun build_index call."""
    key = _cache_key(file_paths, chunk_size)
    if st.session_state.get("active_key") != key:
        build_index(file_paths, chunk_size)


def search(question: str, top_k: int) -> list:
    index  = st.session_state.get("active_index")
    chunks = st.session_state.get("active_chunks")
    if index is None or not chunks:
        return []
    q_vec = get_embedding_model().encode([question]).astype("float32")
    _, ids = index.search(q_vec, min(top_k, len(chunks)))
    return [chunks[i] for i in ids[0] if 0 <= i < len(chunks)]


def is_ready() -> bool:
    return st.session_state.get("active_index") is not None
def search_max(question: str, max_k: int = 10) -> list:
    """Retrieve the top max_k chunks once; slice for smaller k downstream."""
    return search(question, top_k=max_k)