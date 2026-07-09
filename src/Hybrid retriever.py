"""
hybrid_retriever.py
-------------------
Drop-in upgrade for a FAISS + MiniLM RAG pipeline.

Adds three stages on top of plain vector search:
  1. BM25 keyword retrieval (catches exact terms vector search misses)
  2. Reciprocal Rank Fusion (merges BM25 + vector rankings)
  3. Cross-encoder reranking (re-scores finalists by true query-doc relevance)

How to wire it in
------------------
You already have (names may differ in your retriever.py):
  - `chunks`: list[str]              the text chunks you indexed
  - `embed_model`: SentenceTransformer("all-MiniLM-L6-v2")
  - `faiss_index`: a faiss index built from those chunk embeddings

Then:
    from hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(chunks, embed_model, faiss_index)
    top_chunks = retriever.retrieve("your question", final_k=5)
    # feed top_chunks to your LLM prompt exactly as before

Install:
    pip install rank-bm25 sentence-transformers faiss-cpu numpy
"""

from __future__ import annotations
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def _tokenize(text: str) -> list[str]:
    """Cheap tokenizer for BM25. Lowercase + split on non-alphanumerics."""
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """
    Merge several ranked lists of chunk-ids into one.

    Each list is ordered best-first. A document's fused score is the sum of
    1 / (k + rank) across every list it appears in. k=60 is the standard
    constant from the original RRF paper; it damps the influence of very
    high ranks so no single retriever dominates.

    Returns [(chunk_id, fused_score), ...] sorted best-first.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[str],
        embed_model,
        faiss_index,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        chunks       : list of chunk texts (index i must match faiss row i)
        embed_model  : your existing SentenceTransformer (MiniLM)
        faiss_index  : your existing FAISS index over those chunks
        """
        self.chunks = chunks
        self.embed_model = embed_model
        self.faiss_index = faiss_index

        # Build BM25 once over the same chunks.
        self._bm25 = BM25Okapi([_tokenize(c) for c in chunks])

        # Cross-encoder is loaded once and reused. Small + CPU-friendly.
        self._reranker = CrossEncoder(cross_encoder_name)

    # ---- stage 1a: vector search (your existing behaviour) --------------
    def _vector_search(self, query: str, k: int) -> list[int]:
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        _, ids = self.faiss_index.search(q_emb, k)
        return [int(i) for i in ids[0] if i != -1]

    # ---- stage 1b: keyword search ---------------------------------------
    def _bm25_search(self, query: str, k: int) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        return list(np.argsort(scores)[::-1][:k])

    # ---- full pipeline ---------------------------------------------------
    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,   # wide net from each retriever
        fuse_k: int = 12,        # how many fused candidates to rerank
        final_k: int = 5,        # what the LLM actually sees
        min_rerank_score: float | None = None,  # relevance gate; None = off
    ) -> list[dict]:
        """
        Returns a list of dicts: {"chunk_id", "text", "rerank_score"}
        ordered best-first, length <= final_k.

        If min_rerank_score is set and nothing clears it, returns [] so your
        caller can emit a "no relevant content found" message instead of
        forcing the LLM to answer from weak context.
        """
        # 1. two retrievers, wide
        vec_ids = self._vector_search(query, candidate_k)
        bm25_ids = self._bm25_search(query, candidate_k)

        # 2. fuse
        fused = reciprocal_rank_fusion([vec_ids, bm25_ids])
        candidate_ids = [doc_id for doc_id, _ in fused[:fuse_k]]
        if not candidate_ids:
            return []

        # 3. rerank the finalists with the cross-encoder
        pairs = [(query, self.chunks[i]) for i in candidate_ids]
        rerank_scores = self._reranker.predict(pairs)

        ranked = sorted(
            zip(candidate_ids, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 4. optional relevance gate (no-answer fallback)
        if min_rerank_score is not None:
            ranked = [r for r in ranked if r[1] >= min_rerank_score]
            if not ranked:
                return []

        return [
            {"chunk_id": int(i), "text": self.chunks[i], "rerank_score": float(s)}
            for i, s in ranked[:final_k]
        ]


# ------------------------------------------------------------------------
# Quick self-test with toy data (run: python hybrid_retriever.py)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    import faiss
    from sentence_transformers import SentenceTransformer

    demo_chunks = [
        "The mitochondrion is the powerhouse of the cell.",
        "Section 12(a) requires a public consultation before the policy change.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "A public consultation must run for at least eight weeks under the guidance.",
        "Cross-encoders score a query and document jointly for better ranking.",
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(demo_chunks, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    r = HybridRetriever(demo_chunks, model, index)
    results = r.retrieve("How long must a public consultation last?", final_k=3)
    for rank, res in enumerate(results, 1):
        print(f"{rank}. score={res['rerank_score']:.3f}  {res['text']}")