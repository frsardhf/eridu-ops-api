"""Read queries for the public Bond 100 endpoints.

Bridge model: arona is the single source of truth. sync_arona.py caches the
aggregated wall + per-student entries as JSON blobs in bond100_meta, and these
two readers serve them. Nothing about players is stored locally.
"""
import json

from db import get_connection


def _meta(conn, keys: tuple[str, ...]) -> dict[str, str]:
    qs = ",".join("?" * len(keys))
    return {
        r["key"]: r["value"]
        for r in conn.execute(f"SELECT key, value FROM bond100_meta WHERE key IN ({qs})", keys)
    }


def get_summary() -> dict:
    """{ total, snapshotDate?, students: [{ studentId, count, byServer }] } — from the cached blob."""
    conn = get_connection()
    try:
        m = _meta(conn, ("wall_summary", "snapshot_date"))
    finally:
        conn.close()

    result = json.loads(m["wall_summary"]) if m.get("wall_summary") else {"total": 0, "students": []}
    if m.get("snapshot_date"):
        result["snapshotDate"] = m["snapshot_date"]
    return result


def get_student_entries(student_id: int) -> dict:
    """{ studentId, entries: [{ serverRegion, playerName }] } from the cached blob."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM bond100_meta WHERE key = 'entries'").fetchone()
    finally:
        conn.close()

    all_entries = json.loads(row["value"]) if row else {}
    return {"studentId": student_id, "entries": all_entries.get(str(student_id), [])}
