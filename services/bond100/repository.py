"""Read queries for the public Bond 100 endpoints.

Bridge model: arona is the single source of truth. wall_store.assemble_wall()
caches the aggregated wall + per-student entries as JSON blobs in bond100_meta
(rebuilt from bond100_student_rank), and these two readers serve them. Nothing
about players is stored locally.
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
    """{ total, snapshotDate?, students: [{ studentId, count, byServer, fetchedAt }] }.

    fetchedAt is per-student: with the rolling /rank sweep each student refreshes
    on its own cadence, so the wall can show per-tile freshness rather than one
    global snapshot. snapshotDate is kept as the wall-wide 'as of' fallback."""
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
    """{ studentId, fetchedAt?, entries: [{ serverRegion, playerName }] }.

    fetchedAt is read from the summary blob; the entries blob stays a plain list
    so the sync's canonical_wall read path is unaffected."""
    conn = get_connection()
    try:
        m = _meta(conn, ("entries", "wall_summary"))
    finally:
        conn.close()

    all_entries = json.loads(m["entries"]) if m.get("entries") else {}
    result = {"studentId": student_id, "entries": all_entries.get(str(student_id), [])}

    if m.get("wall_summary"):
        for s in json.loads(m["wall_summary"]).get("students", []):
            if s["studentId"] == student_id:
                if s.get("fetchedAt"):
                    result["fetchedAt"] = s["fetchedAt"]
                break
    return result
