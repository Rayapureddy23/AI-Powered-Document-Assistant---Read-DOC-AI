"""
config.py — Single source of truth 


Research question:
    "How does varying chunk size and retrieval depth in a RAG pipeline
     affect the accuracy, contextual relevance, and faithfulness of
     answers generated from unstructured documents?"

Everything configurable lives here. No magic numbers elsewhere.
"""

# ── LLM (Ollama — local, zero API, zero quota) ────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"          # change to match `ollama list`

# ── Embedding model (local, shared by retrieval AND metrics) ─────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim sentence transformer

# ── Experimental design: 3 × 3 factorial ─────────────────────────────────────
CHUNK_SIZES   = [300, 600, 1000]      # characters
K_VALUES      = [3, 5, 10]            # retrieval depth
CHUNK_OVERLAP = 100

EXPERIMENTS = [
    ("E1", 300, 3),  ("E2", 300, 5),  ("E3", 300, 10),
    ("E4", 600, 3),  ("E5", 600, 5),  ("E6", 600, 10),
    ("E7", 1000, 3), ("E8", 1000, 5), ("E9", 1000, 10),
]
EXP_CONFIG = {e: (c, k) for e, c, k in EXPERIMENTS}

# ── Relevance / evidence thresholds ───────────────────────────────────────────
RELEVANCE_THRESHOLD     = 0.5    # chunk counts as relevant if sim to reference ≥ this
GOLD_EVIDENCE_THRESHOLD = 0.55   # Option A: a corpus chunk becomes a gold-evidence
                                 # unit if its similarity to the reference answer
                                 # ≥ this, computed once at the smallest chunk size
                                 # and frozen for all configurations.

# ── Refusal handling ──────────────────────────────────────────────────────────
REFUSAL = "I could not find this in your uploaded documents."

# Multiple accepted refusal phrasings — automated refusal detection matches any.
REFUSAL_PATTERNS = [
    "could not find",
    "not in the",
    "no information",
    "does not contain",
    "cannot find",
    "unable to find",
    "no relevant",
    "isn't in",
    "is not covered",
]

# ── Fixed test set (identical across baseline + all 9 configurations) ────────
# category: factual | inferential | out_of_scope
QUESTIONS = [
    {"id": 1,  "cat": "factual",      "text": "What is the difference between supervised and unsupervised learning?"},
    {"id": 2,  "cat": "factual",      "text": "What is the bias-variance tradeoff in statistical learning?"},
    {"id": 3,  "cat": "factual",      "text": "How does k-fold cross-validation work?"},
    {"id": 4,  "cat": "factual",      "text": "How does the K-Means algorithm work?"},
    {"id": 5,  "cat": "factual",      "text": "What is Principal Component Analysis (PCA) and what is it used for?"},
    {"id": 6,  "cat": "inferential",  "text": "Why would you use a Random Forest instead of a single decision tree?"},
    {"id": 7,  "cat": "inferential",  "text": "What happens to a model when it is too complex or too simple?"},
    {"id": 8,  "cat": "inferential",  "text": "What is the key idea behind how SVMs find a decision boundary?"},
    {"id": 9,  "cat": "out_of_scope", "text": "What is the current stock price of Nvidia today?"},
    {"id": 10, "cat": "out_of_scope", "text": "What is the weather forecast for London this weekend?"},
]

# ── Ground truth (Option A — automated evidence) ─────────────────────────────
# Gold evidence is derived at runtime from the reference answer (see
# metrics.build_gold_evidence), so no manual evidence annotation is required.
# Each entry carries only what a human must define:
#   reference    : expert-written correct answer (used by BERTScore + judge)
#   category     : factual | inferential | out_of_scope
#   out_of_scope : True when the correct behaviour is refusal
GROUND_TRUTH = {
    1:  {"reference": "Supervised learning uses labelled data where the correct "
                      "output is known during training. Unsupervised learning finds "
                      "patterns in unlabelled data without predefined outputs.",
         "category": "factual", "out_of_scope": False},
    2:  {"reference": "The bias-variance tradeoff is the tension between "
                      "underfitting (high bias, overly simple models) and "
                      "overfitting (high variance, overly complex models).",
         "category": "factual", "out_of_scope": False},
    3:  {"reference": "K-fold cross-validation splits data into k equal parts. "
                      "Each part serves once as the test set while the remaining "
                      "k-1 parts train the model; performance is averaged over k runs.",
         "category": "factual", "out_of_scope": False},
    4:  {"reference": "K-Means assigns each point to the nearest centroid, then "
                      "recomputes each centroid as the mean of its assigned points, "
                      "repeating until the centroids converge.",
         "category": "factual", "out_of_scope": False},
    5:  {"reference": "PCA is a dimensionality-reduction technique that projects "
                      "data onto orthogonal axes of maximum variance, called "
                      "principal components, ordered by explained variance.",
         "category": "factual", "out_of_scope": False},
    6:  {"reference": "Random Forests average many decision trees trained on random "
                      "subsets of data and features, reducing variance and "
                      "overfitting relative to a single tree.",
         "category": "inferential", "out_of_scope": False},
    7:  {"reference": "An overly complex model overfits the training data and fails "
                      "to generalise; an overly simple model underfits and cannot "
                      "capture the underlying pattern.",
         "category": "inferential", "out_of_scope": False},
    8:  {"reference": "SVMs find the separating hyperplane that maximises the margin "
                      "between classes, defined by the support vectors — the points "
                      "closest to the decision boundary.",
         "category": "inferential", "out_of_scope": False},
    9:  {"reference": REFUSAL, "category": "out_of_scope", "out_of_scope": True},
    10: {"reference": REFUSAL, "category": "out_of_scope", "out_of_scope": True},
}

# ── System prompts ────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = f"""You are ReadDoc AI, a document question-answering assistant.

Use the provided context chunks to answer the question. The context comes from
the user's uploaded document.

Rules:
- Base your answer on the context. You may rephrase, combine and explain the
  ideas found in the chunks in your own words.
- Always cite the page number(s) you used, e.g. (Page 40).
- Only if the context contains NOTHING related to the question, reply exactly:
  "{REFUSAL}"
- Questions about current events, prices, weather, or anything clearly outside
  the document must be refused with that exact sentence."""

SUMMARY_SYSTEM_PROMPT = """You are ReadDoc AI. The user asked for a summary or
overview of their uploaded document. Below are excerpts sampled from across the
document. Write a clear, structured summary of what this document covers based
on these excerpts. Mention the main topics and themes. Cite page numbers where
useful. Do not refuse — summarise what is available."""

BASELINE_SYSTEM_PROMPT = ("You are a helpful assistant. Answer from your general "
                          "knowledge. Be direct and concise.")