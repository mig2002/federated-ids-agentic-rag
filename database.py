"""
SQLite database persistence for Agentic RAG chat history.
Stores user and assistant messages, sources, and reasoning traces in db.sqlite3.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db.sqlite3")


def get_db_connection():
    """Returns a connection to the SQLite database with row_factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database and creates the chat_messages table if not exists."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                trace_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_session 
            ON chat_messages(session_id, created_at);
            """
        )
        conn.commit()


def save_message(
    role: str,
    content: str,
    source: Optional[str] = None,
    trace: Optional[Dict[str, Any]] = None,
    session_id: str = "default",
) -> int:
    """Saves a message (user or assistant) along with source attribution and trace."""
    init_db()

    # Serialize trace safely (convert Document objects if present)
    trace_json = None
    if trace:
        try:
            safe_trace = {}
            for k, v in trace.items():
                if k == "kb_docs" and isinstance(v, list):
                    safe_trace[k] = [
                        {
                            "page_content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                            "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                        }
                        for doc in v
                    ]
                else:
                    safe_trace[k] = v
            trace_json = json.dumps(safe_trace)
        except Exception as e:
            trace_json = json.dumps({"error": f"Failed to serialize trace: {e}"})

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, source, trace_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, source, trace_json),
        )
        conn.commit()
        return cursor.lastrowid


def load_chat_messages(session_id: str = "default") -> List[Dict[str, Any]]:
    """Loads all chat messages for a given session for rendering in Streamlit."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, content, source, trace_json, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = cursor.fetchall()

    messages = []
    for r in rows:
        trace = None
        if r["trace_json"]:
            try:
                trace = json.loads(r["trace_json"])
            except Exception:
                trace = None

        messages.append(
            {
                "role": r["role"],
                "content": r["content"],
                "source": r["source"],
                "trace": trace,
                "created_at": r["created_at"],
            }
        )
    return messages


def load_recent_history_for_llm(session_id: str = "default", max_turns: int = 6) -> List[Dict[str, str]]:
    """
    Returns the most recent N conversational turns for feeding into the LLM context.
    Format: [{"role": "user"|"assistant", "content": "..."}]
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, max_turns),
        )
        rows = cursor.fetchall()

    # Reverse to keep chronological order
    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def clear_chat_history(session_id: str = "default"):
    """Deletes all messages for the specified session from db.sqlite3."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.commit()


def count_chat_messages(session_id: str = "default") -> int:
    """Returns total count of stored messages."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chat_messages WHERE session_id = ?", (session_id,))
        count = cursor.fetchone()[0]
    return count
