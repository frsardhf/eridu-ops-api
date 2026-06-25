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

# arona's per-server ids, used both inside `_info` payloads and as the `server`
# param for /refresh and /findRank. NOT the same as arona's friend-interface
# "server group" docs (1=China, 3=Japan, 4=International) -- a live probe
# against a real Global/Asia friend code returned 3009 (rejected) for group 4
# and 200 (accepted) only for the specific id below. Keys match SERVER_REGIONS.
GLOBAL_SERVER_IDS = {
    "global_eu": 5,
    "global_tw": 6,
    "global_kr": 7,
    "global_asia": 8,
    "global_na": 9,
}


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


def linked_partner_ids(primary_id: int) -> list[int]:
    """Secondary ids that collapse into this primary (e.g. [10099] for 10098).
    The /rank sweep queries the primary AND its secondaries, since a player can
    be bond-100 on either alt style; results merge under the primary, deduped by
    player key so one player counts once for the unit."""
    return [sec for sec, prim in _SECONDARY_TO_PRIMARY.items() if prim == primary_id]


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
