"""
judge.py — LLM-judge Faithfulness (NON-DETERMINISTIC).

⚠ NON-DETERMINISTIC — uses a local Ollama LLM as judge. Report as mean ± std
over repeated runs; keep separate from deterministic metrics.

Only Faithfulness is judged (the claim-level answer metrics were removed for
speed). This is ~2 judge calls per question instead of ~6.

  Faithfulness = generated claims supported by retrieved context
                 ÷ total generated claims
"""

import json
import numpy as np
import streamlit as st
from src.config import OLLAMA_URL, OLLAMA_MODEL, GROUND_TRUTH


@st.cache_resource(show_spinner=False)
def _judge_client():
    from openai import OpenAI
    return OpenAI(base_url=OLLAMA_URL, api_key="ollama")


def _ask(prompt: str, max_tokens: int = 400) -> str:
    r = _judge_client().chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "system",
                   "content": "You are a strict evaluation judge. "
                              "Reply ONLY with the requested JSON."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0,
        extra_body={"keep_alive": "60m"})
    return r.choices[0].message.content.strip()


def _parse(raw, fallback):
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        s = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0])
        e = max(raw.rfind("}"), raw.rfind("]")) + 1
        return json.loads(raw[s:e])
    except Exception:
        return fallback


def _extract_claims(text: str) -> list:
    if not text or len(text.strip()) < 5:
        return []
    p = ("Break this text into atomic factual claims. "
         'Reply ONLY as JSON: {"claims": ["...", "..."]}.\n\n' + text)
    d = _parse(_ask(p), {"claims": []})
    cl = d.get("claims", []) if isinstance(d, dict) else []
    return [c.strip() for c in cl if isinstance(c, str) and c.strip()]


def score_faithfulness(qid: int, answer: str, chunks: list) -> dict:
    """Faithfulness = generated claims supported by retrieved context / total.
    NON-DETERMINISTIC. None for out-of-scope or no context."""
    gt = GROUND_TRUTH[qid]
    if gt["out_of_scope"] or not chunks:
        return {"faithfulness": None, "faithfulness_detail": None}

    context = "\n\n".join(f"[Page {c['page_number']}] {c['text']}" for c in chunks)
    claims = _extract_claims(answer)
    if not claims:
        return {"faithfulness": 0.0, "faithfulness_detail": json.dumps([])}

    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    p = ("For each numbered claim, decide if the source SUPPORTS it "
         "(explicitly states or directly implies). "
         'Reply ONLY as JSON: {"results":[{"n":1,"supported":true}, ...]}.\n\n'
         f"Source:\n{context}\n\nClaims:\n{numbered}")
    d = _parse(_ask(p, max_tokens=500), {"results": []})
    res = d.get("results", []) if isinstance(d, dict) else []
    smap = {r.get("n"): (1 if r.get("supported") else 0)
            for r in res if isinstance(r, dict)}
    detail = [{"claim": c, "supported": smap.get(i + 1, 0)}
              for i, c in enumerate(claims)]
    supported = sum(x["supported"] for x in detail)
    faith = round(supported / len(claims), 4) if claims else 0.0
    return {"faithfulness": faith, "faithfulness_detail": json.dumps(detail)}


def aggregate_judge(per_question: list) -> dict:
    vals = [q["faithfulness"] for q in per_question
            if q.get("faithfulness") is not None]
    return {"faithfulness": round(float(np.mean(vals)), 4) if vals else None}