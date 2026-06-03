"""SQLite connection + schema init for the Bond 100 Hall service.

The DB file lives in ./data/bond100.sqlite by default (gitignored). Override
with the BOND100_DB_PATH env var (e.g. /opt/eridu-ops-api/var/bond100.sqlite
in production). Bridge model: the DB is just a small cache (bond100_meta blobs +
bond100_refresh_log), so init is a plain executescript.
"""
import os
import sqlite3
from pathlib import Path

_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("BOND100_DB_PATH", str(_DIR / "data" / "bond100.sqlite"))
SCHEMA_PATH = _DIR / "schema.sql"

# Canonical server regions (matches the frontend Bond100ServerRegion union).
SERVER_REGIONS = ("global_na", "global_eu", "global_asia", "global_tw", "global_kr")


# Linked-student pairs (mirror of frontend linkedStudents.ts LINKED_STUDENT_PAIRS).
# Some students exist as two SchaleDB ids for one in-game unit (e.g. Hoshino
# Armed 10098/10099). sync_arona collapses the secondary into the primary so the
# wall shows one tile. Keep this in sync with the FE.
_SECONDARY_TO_PRIMARY = {
    10099: 10098,
}


def primary_student_id(student_id: int) -> int:
    """Resolve a (possibly secondary) student id to its canonical primary id."""
    return _SECONDARY_TO_PRIMARY.get(student_id, student_id)


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL lets multiple gunicorn workers read concurrently while one writes.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db() -> None:
    """Create the cache tables if missing. Idempotent."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"[bond100] schema initialized at {DB_PATH}")
