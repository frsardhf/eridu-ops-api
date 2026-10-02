"""SQLite connection and schema initialization for durable feedback."""

import os
import sqlite3
from pathlib import Path

_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("FEEDBACK_DB_PATH", str(_DIR / "data" / "feedback.sqlite"))
SCHEMA_PATH = _DIR / "schema.sql"


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db() -> None:
    """Create the feedback tables if missing."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"[feedback] schema initialized at {DB_PATH}")
