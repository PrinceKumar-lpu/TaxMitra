"""
User Feedback and Active Learning Service for TaxMitra.
Interview "Why":
- Why a feedback loop in SQLite? Real-world user interactions reveal edge cases not covered by initial RAG docs
  (e.g., obscure deductions, confusing ITR questions, or edge-case salaries). Storing feedback directly in SQLite
  creates an audit trail without external database dependencies.
- Why export to evals/candidates.json? Automatically turns downvoted queries into actionable test candidates for
  continuous evaluation and iterative doc expansion (RLAIF / Active Learning workflow).
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

FEEDBACK_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "feedback.db"


class FeedbackItem(BaseModel):
    session_id: Optional[str] = None
    query: str
    bot_answer: str
    rating: str = Field(description="'up' or 'down'")
    comment: Optional[str] = None
    retrieval_mode: Optional[str] = "hybrid+rerank"


def get_db_connection() -> sqlite3.Connection:
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FEEDBACK_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db():
    """Creates feedback table if it does not exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT NOT NULL,
                bot_answer TEXT NOT NULL,
                rating TEXT NOT NULL,
                comment TEXT,
                retrieval_mode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def record_feedback(item: FeedbackItem) -> int:
    """Stores a feedback entry in SQLite."""
    init_feedback_db()
    with get_db_connection() as conn:
        cur = conn.execute("""
            INSERT INTO feedback (session_id, query, bot_answer, rating, comment, retrieval_mode)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (item.session_id, item.query, item.bot_answer, item.rating, item.comment, item.retrieval_mode))
        conn.commit()
        return cur.lastrowid


def get_all_feedback(limit: int = 100) -> List[Dict[str, Any]]:
    init_feedback_db()
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_downvoted_feedback() -> List[Dict[str, Any]]:
    init_feedback_db()
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM feedback WHERE rating = 'down' ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]
