"""
store.py — SQLite persistence (deterministic 3-construct schema).
=================================================================

Tables:
  answers  — per-question audit trail
  results  — per-configuration aggregated scores (one row per config + run_id)
  runs     — run registry (run_id, timestamp)

Metrics stored map to the three RQ constructs:
  Accuracy             → answer_accuracy
  Contextual relevance → context_relevance, context_precision_at_k,
                         evidence_recall_at_k, context_f1_at_k
  Faithfulness         → faithfulness

⚠ SCHEMA CHANGE: incompatible with older databases. Delete the data folder
before first use (see next steps).
"""

import os, sqlite3
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "experiments.db")


def _conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, config TEXT, question_id INTEGER, category TEXT,
            chunk_size INTEGER, top_k INTEGER, question TEXT, answer TEXT,
            sources TEXT,
            answer_accuracy REAL, context_relevance REAL,
            context_precision_at_k REAL, evidence_recall_at_k REAL,
            context_f1_at_k REAL, faithfulness REAL,
            is_out_of_scope INTEGER, correct_refusal REAL,
            retrieval_ms REAL, generation_s REAL, total_s REAL,
            n_chunks INTEGER, ctx_tokens INTEGER, ans_tokens INTEGER,
            UNIQUE(run_id, config, question_id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, config TEXT, chunk_size INTEGER, top_k INTEGER,
            answer_accuracy REAL, context_relevance REAL,
            context_precision_at_k REAL, evidence_recall_at_k REAL,
            context_f1_at_k REAL, faithfulness REAL, n_questions INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(run_id, config))""")
        c.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, judged INTEGER,
            created_at TEXT DEFAULT (datetime('now')))""")


def save_answer(config, qid, question, answer, sources, row: dict, run_id="run1"):
    def g(k):
        return row.get(k)
    with _conn() as c:
        c.execute("""INSERT INTO answers
            (run_id, config, question_id, category, chunk_size, top_k,
             question, answer, sources, answer_accuracy, context_relevance,
             context_precision_at_k, evidence_recall_at_k, context_f1_at_k,
             faithfulness, is_out_of_scope, correct_refusal,
             retrieval_ms, generation_s, total_s, n_chunks, ctx_tokens, ans_tokens)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id, config, question_id) DO UPDATE SET
             answer=excluded.answer,
             answer_accuracy=excluded.answer_accuracy,
             context_relevance=excluded.context_relevance,
             context_precision_at_k=excluded.context_precision_at_k,
             evidence_recall_at_k=excluded.evidence_recall_at_k,
             context_f1_at_k=excluded.context_f1_at_k,
             faithfulness=excluded.faithfulness,
             is_out_of_scope=excluded.is_out_of_scope,
             correct_refusal=excluded.correct_refusal,
             retrieval_ms=excluded.retrieval_ms,
             generation_s=excluded.generation_s, total_s=excluded.total_s,
             n_chunks=excluded.n_chunks, ctx_tokens=excluded.ctx_tokens,
             ans_tokens=excluded.ans_tokens""",
            (run_id, config, qid, g("category"), row.get("_chunk_size"),
             row.get("_top_k"), question, answer, sources,
             g("answer_accuracy"), g("context_relevance"),
             g("context_precision_at_k"), g("evidence_recall_at_k"),
             g("context_f1_at_k"), g("faithfulness"),
             1 if g("is_out_of_scope") else 0, g("correct_refusal"),
             g("retrieval_ms"), g("generation_s"), g("total_s"),
             g("n_chunks"), g("ctx_tokens"), g("ans_tokens")))


def save_result(config, chunk_size, top_k, agg: dict, n: int,
                run_id="run1", judged=False):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO runs (run_id, judged) VALUES (?,?)",
                  (run_id, 1 if judged else 0))
        c.execute("DELETE FROM results WHERE run_id=? AND config=?", (run_id, config))
        c.execute("""INSERT INTO results
            (run_id, config, chunk_size, top_k, answer_accuracy, context_relevance,
             context_precision_at_k, evidence_recall_at_k, context_f1_at_k,
             faithfulness, n_questions)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, config, chunk_size, top_k,
             agg.get("answer_accuracy"), agg.get("context_relevance"),
             agg.get("context_precision_at_k"), agg.get("evidence_recall_at_k"),
             agg.get("context_f1_at_k"), agg.get("faithfulness"), n))


def load_results(run_id: str = None) -> pd.DataFrame:
    if not os.path.exists(DB):
        return pd.DataFrame()
    with _conn() as c:
        try:
            if run_id:
                return pd.read_sql_query(
                    "SELECT * FROM results WHERE run_id=? ORDER BY config",
                    c, params=(run_id,))
            return pd.read_sql_query("SELECT * FROM results ORDER BY run_id, config", c)
        except Exception:
            return pd.DataFrame()


def load_answers(config: str = None, run_id: str = None) -> pd.DataFrame:
    if not os.path.exists(DB):
        return pd.DataFrame()
    q, params = "SELECT * FROM answers WHERE 1=1", []
    if config:
        q += " AND config=?"; params.append(config)
    if run_id:
        q += " AND run_id=?"; params.append(run_id)
    q += " ORDER BY config, question_id"
    with _conn() as c:
        try:
            return pd.read_sql_query(q, c, params=params)
        except Exception:
            return pd.DataFrame()


def load_runs() -> pd.DataFrame:
    if not os.path.exists(DB):
        return pd.DataFrame()
    with _conn() as c:
        try:
            return pd.read_sql_query("SELECT * FROM runs ORDER BY created_at", c)
        except Exception:
            return pd.DataFrame()