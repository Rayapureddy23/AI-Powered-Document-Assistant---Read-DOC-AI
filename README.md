# ReadDoc AI

### Optimising Retrieval-Augmented Generation
**An Empirical Study of Chunk Size and Retrieval Depth on Answer Quality**

MSc Data Science and Analytics · Research Project

[Research Question](#research-question) · [Metrics](#evaluation-metrics) · [Architecture](#system-architecture) · [Setup](#setup) · [Results](#results) · [Tests](#tests)

</div>

---

## Overview

ReadDoc AI is a Retrieval-Augmented Generation (RAG) pipeline and evaluation framework that answers questions from unstructured documents and measures how two core design parameters — **chunk size** and **retrieval depth** — affect answer quality.

The system runs entirely locally on Ollama (no API keys, no quotas) and evaluates every configuration with a **fully deterministic** metric suite: identical inputs always produce identical scores, making every result exactly reproducible.

## Research Question

> **How does varying chunk size and retrieval depth in a Retrieval-Augmented Generation pipeline affect the accuracy, contextual relevance, and faithfulness of answers generated from unstructured documents?**

The question names three constructs. Each maps to a dedicated, deterministic metric.

## Experimental Design

A **3 × 3 factorial** — chunk size ∈ {300, 600, 1000} characters × retrieval depth *k* ∈ {3, 5, 10} — giving nine configurations (E1–E9), plus a zero-context **baseline** where the model answers without any retrieved document context.

|              | *k* = 3 | *k* = 5 | *k* = 10 |
| :----------- | :-----: | :-----: | :------: |
| **300 chars**  | E1 | E2 | E3 |
| **600 chars**  | E4 | E5 | E6 |
| **1000 chars** | E7 | E8 | E9 |

All configurations are evaluated on the same fixed 10-question test set (5 factual, 3 inferential, 2 out-of-scope). Only chunk size and *k* vary — the model, embeddings, prompts, and index type are held constant.

## Evaluation Metrics

Three constructs, each measured deterministically from sentence-transformer embeddings. **No LLM judge** — this eliminates the non-determinism, parse failures, and cost of judge-based evaluation.

| Construct | Metric | Definition |
| :-------- | :----- | :--------- |
| **Accuracy** | Answer Accuracy | cosine similarity between the generated answer and the expert reference answer |
| **Contextual relevance** | Context F1@k | macro-averaged harmonic mean of Context Precision@k and Evidence Recall@k |
| **Faithfulness** | Faithfulness | mean over answer sentences of the maximum similarity to any retrieved chunk (sentence-level grounding) |

**Supporting sub-metrics** decompose contextual relevance: *Context Precision@k* (relevant retrieved chunks ÷ k) and *Evidence Recall@k* (unique silver-evidence units covered ÷ total). Efficiency (latency, token counts) is recorded as supporting context.

### Silver evidence

Evidence units are derived automatically: a corpus chunk becomes a silver-evidence unit if its similarity to the reference answer exceeds a threshold, computed once at the smallest chunk size and frozen across all configurations.

> **Limitation (documented):** silver evidence is model-derived, not human-annotated. Context Precision and Evidence Recall therefore measure retrieval consistency with the reference-similarity signal rather than agreement with human judgment.

## System Architecture

```
   Document ─▶ Ingest ─▶ Chunk ─▶ Embed ─┐
   (PDF/HTML)  (PyPDF)  (sliding  (MiniLM  │
                        window)   -L6-v2)  ▼
                                        FAISS  (IndexFlatL2, exact)
                                           │
   Question ─▶ Retrieve ◀───(context)──────┘
   (10 fixed)  (top-k)  ─▶ Generate ─▶ Answer
                          (Llama 3.2)     │
                                          ▼
                       Deterministic Evaluation (no LLM judge)
                Accuracy · Context Precision/Recall/F1@k · Faithfulness
```

Retrieval uses a **top-10-once, slice-to-k** design so that k=3 ⊂ k=5 ⊂ k=10 — a fair, nested comparison. The LLM is used only to generate answers; all scoring is deterministic.

## Technology Stack

| Component | Choice | Rationale |
| :-------- | :----- | :-------- |
| Language model | `llama3.2:3b` via Ollama | Local, zero API keys, reproducible (temp 0.1) |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim; shared by retrieval and metrics |
| Vector index | FAISS `IndexFlatL2` | Exact nearest-neighbour → deterministic retrieval |
| Ingestion | PyPDF + BeautifulSoup | PDF / HTML → page-tagged chunks |
| Interface | Streamlit | Chat, experiment runner, results dashboard |
| Storage | SQLite + CSV export | Answer audit trail + aggregated scores |

## Project Structure

```
readdoc-ai/
├── app.py                      # Chat interface: upload → build indexes → ask
├── src/
│   ├── config.py               # Design, questions, ground truth, thresholds
│   ├── ingest.py               # PDF/HTML extraction + sliding-window chunking
│   ├── retriever.py            # Embeddings + cached FAISS index per chunk size
│   ├── llm.py                  # Ollama client — answer generation only
│   ├── metrics.py              # Deterministic metrics + silver-evidence builder
│   ├── store.py                # SQLite persistence (answers · results · runs)
│   └── ui.py                   # Shared theme
├── pages/
│   ├── 1_Methodology.py        # RQ, design, metric formulas + worked examples
│   ├── 2_Run_Experiments.py    # RUN ALL — Baseline + E1–E9, fully deterministic
│   └── 3_Results.py            # Three-construct dashboard + heatmaps
├── tests/
│   └── test_metrics.py         # Unit tests validating every metric formula
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1 — Ollama (one time)
#     install from https://ollama.com, then:
ollama pull llama3.2:3b

# 2 — Python environment
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3 — Launch
streamlit run app.py
```

## Usage

1. **app** — upload the source document, click *Build indexes*.
2. **Run Experiments** — click *Build silver evidence* (once), then **▶▶ RUN ALL** to evaluate Baseline + E1–E9 automatically. Fully deterministic, ~10–15 minutes on local CPU.
3. **Results** — three construct tables, parameter heatmaps, best-observed configuration. Export the CSVs as the permanent record.

> Streamlit Cloud cannot run Ollama and wipes the database on redeploy. All data collection is local; the exported CSVs are the record of results.

## Results

Representative findings from a full run (E5 best on contextual relevance):

| Config | Accuracy | Context F1@k | Faithfulness |
| :----- | :------: | :----------: | :----------: |
| Baseline | 0.640 | N/A | N/A |
| E1 (300, 3) | 0.798 | 0.526 | 0.592 |
| E5 (600, 5) | 0.770 | **0.541** | 0.555 |
| E7 (1000, 3) | 0.807 | 0.387 | 0.520 |

**Key findings:**

- **Retrieval improves accuracy** — answer accuracy rises from 0.64 (zero-context baseline) to ~0.80 under retrieval, a clear demonstration of RAG's value.
- **Retrieval grounds answers** — faithfulness is undefined at baseline (no context) and measurable (0.52–0.64) under retrieval.
- **Contextual relevance depends on parameters** — Context F1@k peaks at E5 (600 chars, k=5); the precision–recall trade-off across k is the central retrieval finding.

## Tests

```bash
pip install pytest
python -m pytest tests/test_metrics.py -v
```

Unit tests validate every metric formula against small worked examples (e.g. Context Precision@5 = 2/5 = 0.4000, Evidence Recall@5 = 2/3 = 0.6667, Context F1@5 = 0.5000), including duplicate-evidence handling, nested-retrieval invariants, baseline N/A behaviour, and database round-trips.

## Methodological Notes

- **Determinism** — all metrics are computed from embeddings; identical inputs yield identical scores.
- **Silver evidence** — model-derived (not human-annotated); the associated limitation is documented above.
- **Accuracy** — an embedding-similarity measure; claim-level factual verification is identified as future work.
- **Dataset scale** — 10 questions (2 out-of-scope) make findings indicative rather than conclusive.
- **Best-configuration framing** — reported as the *best observed* configuration, with a near-tie warning when configurations differ by less than 0.02. No single composite score drives selection.

## References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 33.
2. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
3. Manning, C., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
5. Johnson, J., Douze, M., & Jégou, H. (2021). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data 7(3).

---

Author : Dharmendra Kumar Reddy Rayapureddy
Msc Data Science 
University of Hertfordshire