"""Tiny SQLite layer: users and the list of uploaded documents."""
import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash BLOB NOT NULL,
    team          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    is_active     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    team         TEXT NOT NULL,
    filename     TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    uploaded_by  TEXT NOT NULL,
    uploaded_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (team, sha256)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def create_user(username: str, password_hash: bytes, team: str, role: str = "user") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, team, role) VALUES (?, ?, ?, ?)",
            (username, password_hash, team, role),
        )


def get_user(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def add_document(team: str, filename: str, sha256: str, uploaded_by: str) -> int:
    """Raises sqlite3.IntegrityError if this team already uploaded identical content."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (team, filename, sha256, uploaded_by) VALUES (?, ?, ?, ?)",
            (team, filename, sha256, uploaded_by),
        )
        return cur.lastrowid


def delete_document(doc_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def list_documents(team: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT filename, uploaded_by, uploaded_at FROM documents "
            "WHERE team = ? ORDER BY uploaded_at DESC",
            (team,),
        ).fetchall()
        return [dict(r) for r in rows]