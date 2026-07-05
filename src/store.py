"""
store.py — SQLite persistence for evaluation results.
======================================================
Two tables:
  answers  — every generated answer (audit trail)
  results  — aggregated metric scores per configuration

Note: Streamlit Cloud wipes local files on redeploy. Collect data
locally and use the CSV export on the Results page as the permanent
record for the dissertation.
"""

import os, sqlite3, json
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "experiments.db")


def _conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    return sqlite3.connect(DB)


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config      TEXT,
            question_id INTEGER,
            question    TEXT,
            answer      TEXT,
            sources     TEXT,
            metrics     TEXT,
            UNIQUE(config, question_id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS results (
            config            TEXT PRIMARY KEY,
            chunk_size        INTEGER,
            top_k             INTEGER,
            answer_accuracy   REAL,
            faithfulness      REAL,
            context_relevance REAL,
            precision_at_k    REAL,
            recall_at_k       REAL,
            mrr               REAL,
            rq_accuracy       REAL,
            rq_relevance      REAL,
            rq_faithfulness   REAL,
            overall           REAL,
            n_questions       INTEGER,
            created_at        TEXT DEFAULT (datetime('now')))""")


def save_answer(config, qid, question, answer, sources, metrics: dict):
    with _conn() as c:
        c.execute("""INSERT INTO answers
            (config, question_id, question, answer, sources, metrics)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(config, question_id) DO UPDATE SET
              answer=excluded.answer, sources=excluded.sources,
              metrics=excluded.metrics""",
            (config, qid, question, answer, sources, json.dumps(metrics)))


def save_result(config, chunk_size, top_k, agg: dict, n: int):
    with _conn() as c:
        c.execute("DELETE FROM results WHERE config=?", (config,))
        c.execute("""INSERT INTO results
            (config, chunk_size, top_k, answer_accuracy, faithfulness,
             context_relevance, precision_at_k, recall_at_k, mrr,
             rq_accuracy, rq_relevance, rq_faithfulness, overall, n_questions)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (config, chunk_size, top_k,
             agg["answer_accuracy"], agg["faithfulness"], agg["context_relevance"],
             agg["precision_at_k"], agg["recall_at_k"], agg["mrr"],
             agg["rq_accuracy"], agg["rq_relevance"], agg["rq_faithfulness"],
             agg["overall"], n))


def load_results() -> pd.DataFrame:
    if not os.path.exists(DB):
        return pd.DataFrame()
    with _conn() as c:
        try:
            return pd.read_sql_query("SELECT * FROM results ORDER BY config", c)
        except Exception:
            return pd.DataFrame()


def load_answers(config: str) -> pd.DataFrame:
    if not os.path.exists(DB):
        return pd.DataFrame()
    with _conn() as c:
        try:
            return pd.read_sql_query(
                "SELECT * FROM answers WHERE config=? ORDER BY question_id",
                c, params=(config,))
        except Exception:
            return pd.DataFrame()
