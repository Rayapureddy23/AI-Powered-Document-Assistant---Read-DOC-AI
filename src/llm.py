"""
llm.py — Answer generation via Ollama (local).
===============================================


Prerequisite (one time):
    ollama pull llama3.2:3b
"""

import streamlit as st
from src.config import (OLLAMA_URL, OLLAMA_MODEL,
                        RAG_SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT,
                        SUMMARY_SYSTEM_PROMPT)


@st.cache_resource(show_spinner=False)
def _client():
    from openai import OpenAI
    return OpenAI(base_url=OLLAMA_URL, api_key="ollama")


def ollama_available() -> tuple:
    """(ok, message) — pre-flight check before any run."""
    try:
        models = [m.id for m in _client().models.list().data]
        if OLLAMA_MODEL in models:
            return True, f"Ollama ready — {OLLAMA_MODEL}"
        return False, (f"Model '{OLLAMA_MODEL}' not found. "
                       f"Run:  ollama pull {OLLAMA_MODEL}")
    except Exception:
        return False, ("Ollama is not running. Install from ollama.com, then "
                       f"run:  ollama pull {OLLAMA_MODEL}")


def _format_context(chunks: list) -> str:
    out = ""
    for i, c in enumerate(chunks, 1):
        out += (f"\n--- Chunk {i} [{c['file_name']} | "
                f"Page {c['page_number']}] ---\n{c['text']}\n")
    return out


def generate_answer(question: str, chunks: list) -> str:
    """RAG answer — constrained to the retrieved context."""
    user = (f"Context from uploaded documents:\n{_format_context(chunks)}\n\n---\n\n"
            f"Question: {question}") if chunks else question
    resp = _client().chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT},
                  {"role": "user",   "content": user}],
        max_tokens=512,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def generate_baseline(question: str) -> str:
    """Control condition — no document context at all."""
    resp = _client().chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                  {"role": "user",   "content": question}],
        max_tokens=512,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def generate_summary(question: str, chunks: list) -> str:
    """Summary-mode answer: samples chunks from across the document."""
    user = (f"Excerpts from the document:\n{_format_context(chunks)}\n\n---\n\n"
            f"Request: {question}")
    resp = _client().chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                  {"role": "user",   "content": user}],
        max_tokens=700,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def stream_answer(question: str, chunks: list, history: list):
    """Streaming variant for the chat page."""
    user = (f"Context from uploaded documents:\n{_format_context(chunks)}\n\n---\n\n"
            f"Question: {question}") if chunks else question
    messages = ([{"role": "system", "content": RAG_SYSTEM_PROMPT}]
                + history
                + [{"role": "user", "content": user}])
    stream = _client().chat.completions.create(
        model=OLLAMA_MODEL, messages=messages,
        max_tokens=512, temperature=0.1, stream=True)
    for part in stream:
        delta = part.choices[0].delta.content
        if delta:
            yield delta