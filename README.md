# ReadDoc AI

**An Empirical Study of Chunking Strategy and Retrieval Depth in Retrieval-Augmented Generation**

MSc Data Science and Analytics — Research Project

---

## Research Question

> *How does varying chunk size and retrieval depth in a Retrieval-Augmented Generation pipeline affect the accuracy, contextual relevance, and faithfulness of answers generated from unstructured documents?*

## Design

3 × 3 factorial (chunk size ∈ {300, 600, 1000} chars × k ∈ {3, 5, 10}) = 9 configurations
plus a zero-context **baseline**, all evaluated on a fixed 10-question test set with a
**fully automated, deterministic metric pipeline** — no manual scoring, no LLM judge.

| RQ construct | Automated metrics |
|---|---|
| Accuracy | cos(E(answer), E(reference)) · Recall@k |
| Contextual relevance | mean cos(E(question), E(chunkᵢ)) · Precision@k · MRR |
| Faithfulness | cos(E(answer), mean E(chunks)) — baseline ≡ 0.0 |

All metrics are computed from `all-MiniLM-L6-v2` embeddings and ground-truth page
labels: identical inputs always produce identical scores.

## Stack

Ollama (`llama3.2:3b`, local, zero API keys) · sentence-transformers · FAISS IndexFlatL2 ·
SQLite · Streamlit · Plotly

## Project layout

```
├── app.py                    # Chat interface (upload → index → ask)
├── src/
│   ├── config.py             # Single source of truth: design, questions, ground truth
│   ├── ingest.py             # PDF/HTML extraction + overlapping chunking
│   ├── retriever.py          # Embeddings + cached FAISS index per chunk size
│   ├── llm.py                # Ollama client (generation only — no judge)
│   ├── metrics.py            # 6 deterministic automated metrics
│   ├── store.py              # SQLite persistence
│   └── ui.py                 # Shared page theme
└── pages/
    ├── 1_Methodology.py      # RQ, design, metric formulas (LaTeX) + worked examples
    ├── 2_Run_Experiments.py  # One-click: generate answers → score → save
    └── 3_Results.py          # Tables, charts, heatmap, findings, CSV export
```

## Setup

```bash
# 1. Ollama (one time)
#    install from https://ollama.com then:
ollama pull llama3.2:3b

# 2. Python environment
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## Workflow

1. **app** — upload the source PDF, click *Build indexes (all chunk sizes)*.
2. **Run Experiments** — run *Baseline*, then E1…E9 (one click each, ~5 min per run).
3. **Results** — tables, charts and findings populate automatically; export the CSV.

> Streamlit Cloud does not persist the SQLite database. Collect all data locally and
> keep the exported CSV as the permanent dissertation record.

## Key references

1. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217
2. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 33.
3. Manning, C., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.* EMNLP 2019.
5. Johnson, J., Douze, M., & Jégou, H. (2021). *Billion-scale similarity search with GPUs.* IEEE Big Data 7(3).
