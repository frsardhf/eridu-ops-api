"""Per-student bond-100 store + wall assembly.

Bridge-model evolution. Previously one rank_by_max_favor_user_info ("_info")
call clobbered the wall blobs each sync. arona's _info cache froze for 3+ weeks
(2026-06 incident), so the wall now accumulates per-student: each student's
bond-100 entries live as a row in bond100_student_rank, written by whatever
source last fetched it (the 'info' baseline, or the per-student 'rank' sweep
against friends/rank, which is real-time). assemble_wall() rebuilds the
bond100_meta blobs (wall_summary + entries) from those rows, so repository.py
and the frontend keep reading the exact same shape.

This module owns the table and the assembly. The _info sync and the /rank sweep
both write rows via upsert_student() then call assemble_wall().
"""
import argparse
import json
import sys
from datetime import date

from db import DB_PATH, get_connection, init_db, primary_student_id
from sync_arona import canonical_wall


def upsert_student(conn, student_id: int, count: int, entries: list, source: str,
                   fetched_at: str) -> None:
    """Write one student's bond-100 row. `entries` is the display list already
    deduped by the caller: [{serverRegion, playerName}]. Keyed (and collapsed)
    to the primary id, so a 'rank' fetch fully replaces an 'info' seed."""
    sid = primary_student_id(student_id)
    conn.execute(
        "INSERT INTO bond100_student_rank (student_id, count, entries, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(student_id) DO UPDATE SET "
        "count = excluded.count, entries = excluded.entries, "
        "source = excluded.source, fetched_at = excluded.fetched_at",
        (sid, count, json.dumps(entries, ensure_ascii=False, separators=(",", ":")),
         source, fetched_at),
    )


def read_all_students(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT student_id, count, entries, source, fetched_at FROM bond100_student_rank"
    ).fetchall()
    return [
        {
            "student_id": r["student_id"],
            "count": r["count"],
            "entries": json.loads(r["entries"]),
            "source": r["source"],
            "fetched_at": r["fetched_at"],
        }
        for r in rows
    ]


def build_wall(students: list[dict]) -> tuple[dict, dict]:
    """Pure: rows -> (wall_summary, entries_blob), the same shape sync_arona's
    aggregate() produced. The displayed count is len(entries) (== byServer sum)
    so the wall is always internally consistent with the names it shows; the
    stored `count` column is kept for the sweep scheduler / validation. With
    size=200 fetches no student's bond-100 block is capped, so the two agree."""
    summary_students = []
    entries_blob: dict[str, list] = {}
    for s in students:
        by_server: dict[str, int] = {}
        public_entries = []
        for e in s["entries"]:
            region = e["serverRegion"]
            by_server[region] = by_server.get(region, 0) + 1
            public_entries.append({"serverRegion": region, "playerName": e["playerName"]})
        sid = s["student_id"]
        summary_students.append({
            "studentId": sid,
            "count": sum(by_server.values()),
            "byServer": by_server,
            "fetchedAt": s["fetched_at"],
        })
        entries_blob[str(sid)] = public_entries

    summary_students.sort(key=lambda x: (-x["count"], x["studentId"]))
    summary = {"total": sum(s["count"] for s in summary_students), "students": summary_students}
    return summary, entries_blob


def _strip_freshness(summary: dict) -> dict:
    """Drop per-student fetchedAt so the data (counts/byServer) can be compared
    against the old _info wall, which predates the freshness field."""
    return {
        "total": summary["total"],
        "students": [
            {k: v for k, v in s.items() if k != "fetchedAt"} for s in summary["students"]
        ],
    }


def _read_blobs(conn) -> tuple[dict | None, dict | None, str | None]:
    rows = {
        r["key"]: r["value"]
        for r in conn.execute(
            "SELECT key, value FROM bond100_meta "
            "WHERE key IN ('wall_summary', 'entries', 'snapshot_date')"
        )
    }
    summary = json.loads(rows["wall_summary"]) if rows.get("wall_summary") else None
    entries = json.loads(rows["entries"]) if rows.get("entries") else None
    return summary, entries, rows.get("snapshot_date")


def _write_blobs(conn, summary: dict, entries: dict) -> None:
    for key, value in (
        ("wall_summary", json.dumps(summary, ensure_ascii=False, separators=(",", ":"))),
        ("entries", json.dumps(entries, ensure_ascii=False, separators=(",", ":"))),
    ):
        conn.execute(
            "INSERT INTO bond100_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def assemble_wall() -> tuple[dict, dict]:
    """Rebuild the bond100_meta wall blobs from bond100_student_rank. Returns the
    (summary, entries) written. snapshot_date is left untouched here (the sweep
    owns freshness via per-student fetched_at)."""
    init_db()
    conn = get_connection()
    try:
        summary, entries = build_wall(read_all_students(conn))
        _write_blobs(conn, summary, entries)
        conn.commit()
    finally:
        conn.close()
    return summary, entries


def seed_from_info(default_fetched_at: str) -> int:
    """One-time: explode the current _info wall (bond100_meta blobs) into
    per-student rows so the served wall stays identical while the /rank sweep
    gradually overwrites rows. source='info'; fetched_at = the _info snapshot_date
    (or `default_fetched_at` if unset) so seeded rows sort as stalest and are
    swept first. Idempotent (upsert); a later 'rank' row replaces its seed."""
    init_db()
    conn = get_connection()
    try:
        summary, entries, snapshot = _read_blobs(conn)
        if entries is None:
            return 0
        fetched = snapshot or default_fetched_at
        counts = {s["studentId"]: s["count"] for s in (summary or {}).get("students", [])}
        # Never clobber a fresher /rank row with the stale _info seed: once the
        # sweep has fetched a student, re-running the seed leaves it alone.
        rank_rows = {
            r["student_id"]
            for r in conn.execute("SELECT student_id FROM bond100_student_rank WHERE source = 'rank'")
        }
        seeded = 0
        for sid_str, ent_list in entries.items():
            sid = int(sid_str)
            if sid in rank_rows:
                continue
            upsert_student(conn, sid, counts.get(sid, len(ent_list)), ent_list, "info", fetched)
            seeded += 1
        conn.commit()
    finally:
        conn.close()
    return seeded


def _verify(default_fetched_at: str) -> bool:
    """Seed the table from the live _info wall, assemble a fresh wall from the
    table, and confirm the two are canonically identical. Writes the seed rows
    (a new table, not yet read by anything) but NOT the wall blobs, so serving
    is untouched. Returns True on match."""
    print(f"cache db: {DB_PATH}")
    conn = get_connection()
    try:
        live_summary, live_entries, snapshot = _read_blobs(conn)
    finally:
        conn.close()
    if live_summary is None or live_entries is None:
        print("verify: no live _info wall in bond100_meta; nothing to compare against.")
        return False

    seeded = seed_from_info(default_fetched_at)
    conn = get_connection()
    try:
        assembled_summary, assembled_entries = build_wall(read_all_students(conn))
    finally:
        conn.close()

    print(f"verify: seeded {seeded} students; "
          f"live total={live_summary['total']} assembled total={assembled_summary['total']}")
    # Compare on data only; the assembled wall carries fetchedAt the _info wall lacks.
    if canonical_wall(_strip_freshness(assembled_summary), assembled_entries) == \
            canonical_wall(live_summary, live_entries):
        print(f"verify: MATCH — assembled wall is identical to the live _info wall "
              f"(snapshot {snapshot}). Safe to --commit.")
        return True

    print("verify: MISMATCH — assembled wall differs from the live wall. Diff:")
    live_counts = {s["studentId"]: s["count"] for s in live_summary["students"]}
    asm_counts = {s["studentId"]: s["count"] for s in assembled_summary["students"]}
    for sid in sorted(live_counts.keys() | asm_counts.keys()):
        if live_counts.get(sid, 0) != asm_counts.get(sid, 0):
            print(f"  student {sid}: live={live_counts.get(sid, 0)} assembled={asm_counts.get(sid, 0)}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-student bond-100 store: seed from the _info wall and assemble the served wall.")
    ap.add_argument("--default-fetched-at", default=date.today().isoformat(),
                    help="fetched_at for seeded rows when the _info snapshot_date is unset (YYYY-MM-DD).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true",
                   help="Seed the table + assemble in memory + confirm it matches the live wall. "
                        "Writes seed rows but NOT the served blobs.")
    g.add_argument("--commit", action="store_true",
                   help="Seed the table, then assemble and write the served wall blobs.")
    args = ap.parse_args()

    if args.verify:
        ok = _verify(args.default_fetched_at)
        sys.exit(0 if ok else 1)

    if args.commit:
        if not _verify(args.default_fetched_at):
            sys.exit("commit aborted: verify did not match; refusing to overwrite the served wall.")
        summary, entries = assemble_wall()
        print(f"commit: wrote assembled wall (total={summary['total']} "
              f"students={len(summary['students'])}) to bond100_meta.")


if __name__ == "__main__":
    main()
