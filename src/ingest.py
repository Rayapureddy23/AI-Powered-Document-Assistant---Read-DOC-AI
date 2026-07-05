"""
ingest.py — Document extraction and chunking.
==============================================
PDF (PyPDF) and HTML (BeautifulSoup) → page records → overlapping chunks.
Page numbers are preserved end-to-end for citations and retrieval metrics.
"""

import os
from src.config import CHUNK_OVERLAP


def extract_pages(file_path: str) -> list:
    """Return [{'text', 'file_name', 'page_number'}, ...] for one document."""
    name = os.path.basename(file_path)
    ext  = os.path.splitext(file_path)[1].lower()
    pages = []

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "file_name": name, "page_number": i})

    elif ext in (".html", ".htm"):
        from bs4 import BeautifulSoup
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            pages.append({"text": text, "file_name": name, "page_number": 1})

    else:  # plain text fallback
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            pages.append({"text": text, "file_name": name, "page_number": 1})

    return pages


def chunk_pages(pages: list, chunk_size: int) -> list:
    """Sliding-window chunking with overlap; each chunk keeps its page number."""
    chunks = []
    step = max(chunk_size - CHUNK_OVERLAP, 1)
    for page in pages:
        text = page["text"]
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if len(piece) >= 50:  # drop tiny fragments
                chunks.append({
                    "text":        piece,
                    "file_name":   page["file_name"],
                    "page_number": page["page_number"],
                })
            if start + chunk_size >= len(text):
                break
    return chunks
