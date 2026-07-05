"""
config.py — Single source of truth for the study design.
=========================================================
ReadDoc AI | MSc Data Science and Analytics

Research question:
    "How does varying chunk size and retrieval depth in a RAG pipeline
     affect the accuracy, contextual relevance, and faithfulness of
     answers generated from unstructured documents?"

Everything configurable lives here. No magic numbers anywhere else.
"""

# ── LLM (Ollama — local, zero API, zero quota) ────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"          # change to match `ollama list`

# ── Embedding model (local, shared by retrieval AND metrics) ─────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim sentence transformer

# ── Experimental design: 3 × 3 factorial ─────────────────────────────────────
CHUNK_SIZES  = [300, 600, 1000]       # characters
K_VALUES     = [3, 5, 10]             # retrieval depth
CHUNK_OVERLAP = 100

EXPERIMENTS = [
    ("E1", 300, 3),  ("E2", 300, 5),  ("E3", 300, 10),
    ("E4", 600, 3),  ("E5", 600, 5),  ("E6", 600, 10),
    ("E7", 1000, 3), ("E8", 1000, 5), ("E9", 1000, 10),
]
EXP_CONFIG = {e: (c, k) for e, c, k in EXPERIMENTS}

# ── Fixed test set (identical across baseline + all 9 configurations) ────────
# cat: Factual = answer stated directly in document
#      Inferential = requires combining document content
#      Out-of-scope = NOT in document; correct behaviour is refusal
QUESTIONS = [
    {"id": 1,  "cat": "Factual",      "text": "What is the difference between supervised and unsupervised learning?"},
    {"id": 2,  "cat": "Factual",      "text": "What is the bias-variance tradeoff in statistical learning?"},
    {"id": 3,  "cat": "Factual",      "text": "How does k-fold cross-validation work?"},
    {"id": 4,  "cat": "Factual",      "text": "How does the K-Means algorithm work?"},
    {"id": 5,  "cat": "Factual",      "text": "What is Principal Component Analysis (PCA) and what is it used for?"},
    {"id": 6,  "cat": "Inferential",  "text": "Why would you use a Random Forest instead of a single decision tree?"},
    {"id": 7,  "cat": "Inferential",  "text": "What happens to a model when it is too complex or too simple?"},
    {"id": 8,  "cat": "Inferential",  "text": "What is the key idea behind how SVMs find a decision boundary?"},
    {"id": 9,  "cat": "Out-of-scope", "text": "What is the current stock price of Nvidia today?"},
    {"id": 10, "cat": "Out-of-scope", "text": "What is the weather forecast for London this weekend?"},
]

# ── Ground truth ──────────────────────────────────────────────────────────────
# reference : expert-written correct answer  → used by Answer Accuracy metric
# pages     : page numbers containing the answer → used by retrieval metrics
#             (verify against your own copy of the source document)
REFUSAL = "I could not find this in your uploaded documents."

GROUND_TRUTH = {
    1:  {"pages": {40, 139},
         "reference": "Supervised learning uses labelled data where the correct "
                      "output is known during training. Unsupervised learning finds "
                      "patterns in unlabelled data without predefined outputs."},
    2:  {"pages": {45, 46},
         "reference": "The bias-variance tradeoff is the tension between "
                      "underfitting (high bias, overly simple models) and "
                      "overfitting (high variance, overly complex models)."},
    3:  {"pages": {52},
         "reference": "K-fold cross-validation splits data into k equal parts. "
                      "Each part serves once as the test set while the remaining "
                      "k-1 parts train the model; performance is averaged over k runs."},
    4:  {"pages": {98, 99},
         "reference": "K-Means assigns each point to the nearest centroid, then "
                      "recomputes each centroid as the mean of its assigned points, "
                      "repeating until the centroids converge."},
    5:  {"pages": {110, 111},
         "reference": "PCA is a dimensionality-reduction technique that projects "
                      "data onto orthogonal axes of maximum variance, called "
                      "principal components, ordered by explained variance."},
    6:  {"pages": {78, 79},
         "reference": "Random Forests average many decision trees trained on random "
                      "subsets of data and features, reducing variance and "
                      "overfitting relative to a single tree."},
    7:  {"pages": {45},
         "reference": "An overly complex model overfits the training data and fails "
                      "to generalise; an overly simple model underfits and cannot "
                      "capture the underlying pattern."},
    8:  {"pages": {88, 89},
         "reference": "SVMs find the separating hyperplane that maximises the margin "
                      "between classes, defined by the support vectors — the points "
                      "closest to the decision boundary."},
    9:  {"pages": set(), "reference": REFUSAL},
    10: {"pages": set(), "reference": REFUSAL},
}

# ── System prompts ────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = f"""You are ReadDoc AI, a document question-answering assistant.

Answer using ONLY the provided context chunks. Cite the page number.
If the answer is not in the context, reply exactly: "{REFUSAL}"
Never invent information that is not in the context."""

BASELINE_SYSTEM_PROMPT = ("You are a helpful assistant. Answer from your general "
                          "knowledge. Be direct and concise.")
