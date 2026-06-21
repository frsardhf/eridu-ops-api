"""Bridge sync: pull arona's rank_by_max_favor_user_info, aggregate, cache.

Bridge model: arona is the single source of truth. We make ONE call, aggregate
the global servers into a wall summary + per-student entries, and cache both as
JSON blobs in bond100_meta. No per-entry rows, no dedup, no moderation storage.

    python3 sync_arona.py --from-file userinfo_full_s9.json   # local, no network
    ARONA_TOKEN='<email>:<token>' python3 sync_arona.py        # live fetch
    python3 sync_arona.py --dry-run                            # report, write nothing
    python3 sync_arona.py --server 8                           # raw per-student counts for ONE server (cross-check vs arona.icu)

The endpoint's `server` param is ignored (it always returns the full cross-server
cache, refreshed every 4h on arona's side), so we request once and filter to the
five global servers ourselves.

If arona serves a wall identical to what we already cached (its upstream ranking
cache has frozen for a week+ before; 2026-06 incident), the sync skips the write
and keeps snapshot_date, so the frontend reports when the data last actually
changed rather than when we last fetched it.
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import DB_PATH, get_connection, init_db, primary_student_id  # noqa: E402

API_URL = "https://api.arona.icu/api/friends/rank_by_max_favor_user_info"
GLOBAL_SERVERS = {5: "global_eu", 6: "global_tw", 7: "global_kr", 8: "global_asia", 9: "global_na"}


def fetch_live() -> dict:
    token = os.environ.get("ARONA_TOKEN")
    if not token:
        sys.exit("ARONA_TOKEN env var required for a live fetch")
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"server": 4}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"ba-token {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def aggregate(data: list) -> tuple[dict, dict]:
    """(wall_summary, entries_by_student). Global servers only; one player
    contributes to each student in its studentIds list."""
    counts: dict[int, dict[str, int]] = {}
    entries: dict[int, list] = {}
    for p in data:
        region = GLOBAL_SERVERS.get(p.get("server"))
        if region is None:
            continue
        nick = (p.get("nickname") or "").strip()
        if not nick:
            continue
        for sid in (p.get("studentIds") or []):
            sid = primary_student_id(sid)
            counts.setdefault(sid, {}).setdefault(region, 0)
            counts[sid][region] += 1
            entries.setdefault(sid, []).append({"serverRegion": region, "playerName": nick})

    students = sorted(
        ({"studentId": s, "count": sum(by.values()), "byServer": by} for s, by in counts.items()),
        key=lambda x: (-x["count"], x["studentId"]),
    )
    summary = {"total": sum(s["count"] for s in students), "students": students}
    entries_json = {str(k): v for k, v in entries.items()}
    return summary, entries_json


def inspect_server(data: list, server: int, student: int | None = None) -> None:
    """Diagnostic: per-student bond-100 counts for ONE server, straight from
    arona's payload. Unlike aggregate(), this does NOT collapse linked students
    or drop blank-nickname players, so the numbers line up with what arona.icu
    shows for that single server. Writes nothing; for cross-checking the _info
    endpoint against the website when a server's counts look stuck."""
    region = GLOBAL_SERVERS.get(server, f"server_{server}")
    counts: dict[int, int] = {}
    names: dict[int, list[str]] = {}
    players = blank = 0
    for p in data:
        if p.get("server") != server:
            continue
        players += 1
        nick = (p.get("nickname") or "").strip()
        if not nick:
            blank += 1
        for sid in (p.get("studentIds") or []):
            counts[sid] = counts.get(sid, 0) + 1
            names.setdefault(sid, []).append(nick or "(no nickname)")

    total = sum(counts.values())
    print(f"server {server} ({region}): players={players} blank_nickname={blank} "
          f"bond100_slots={total} students={len(counts)}")
    if student is not None:
        print(f"  student {student}: count={counts.get(student, 0)}")
        for n in sorted(names.get(student, [])):
            print(f"    {n}")
        return
    for sid in sorted(counts, key=lambda s: (-counts[s], s)):
        print(f"  {sid}: {counts[sid]}")


def read_cache() -> tuple[dict | None, dict | None, str | None]:
    """(wall_summary, entries, snapshot_date) currently cached; Nones when empty."""
    init_db()
    conn = get_connection()
    try:
        rows = {
            r["key"]: r["value"]
            for r in conn.execute(
                "SELECT key, value FROM bond100_meta "
                "WHERE key IN ('wall_summary', 'entries', 'snapshot_date')"
            )
        }
    finally:
        conn.close()
    summary = json.loads(rows["wall_summary"]) if rows.get("wall_summary") else None
    entries = json.loads(rows["entries"]) if rows.get("entries") else None
    return summary, entries, rows.get("snapshot_date")


def canonical_wall(summary: dict, entries: dict) -> str:
    """Order-independent fingerprint of the wall. arona re-serves its cache in a
    different list order run to run, so byte-comparing the blobs reports false
    changes; normalize entry order and dict key order before comparing."""
    norm_entries = {
        sid: sorted((e["serverRegion"], e["playerName"]) for e in lst)
        for sid, lst in entries.items()
    }
    return json.dumps({"summary": summary, "entries": norm_entries},
                      ensure_ascii=False, sort_keys=True)


def diff_counts(old_summary: dict | None, summary: dict) -> list[str]:
    """Per-student count changes, for journald and --dry-run vetting."""
    old = {s["studentId"]: s["count"] for s in (old_summary or {}).get("students", [])}
    new = {s["studentId"]: s["count"] for s in summary["students"]}
    return [
        f"  student {sid}: {old.get(sid, 0)} -> {new.get(sid, 0)}"
        for sid in sorted(old.keys() | new.keys())
        if old.get(sid, 0) != new.get(sid, 0)
    ]


def write_cache(summary: dict, entries: dict, snapshot_date: str) -> None:
    init_db()
    conn = get_connection()
    try:
        for key, value in (
            ("wall_summary", json.dumps(summary, ensure_ascii=False, separators=(",", ":"))),
            ("entries", json.dumps(entries, ensure_ascii=False, separators=(",", ":"))),
            ("snapshot_date", snapshot_date),
        ):
            conn.execute(
                "INSERT INTO bond100_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync the Bond 100 wall from arona's user-info endpoint.")
    ap.add_argument("--from-file", help="Read a saved userinfo dump instead of fetching (no network).")
    ap.add_argument("--snapshot-date", default=date.today().isoformat(),
                    help="Snapshot date YYYY-MM-DD (only written when the wall actually changed).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + aggregate + report what would change, but write nothing.")
    ap.add_argument("--server", type=int,
                    help="Diagnostic: print raw per-student bond-100 counts for ONE server id "
                         "(e.g. 8 = global_asia) straight from arona, no aggregation/collapsing. "
                         "Writes nothing. Pair with --student to list that student's player names.")
    ap.add_argument("--student", type=int,
                    help="With --server, list the player names at bond 100 for this student id.")
    args = ap.parse_args()

    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = fetch_live()

    if d.get("code") != 200 or not isinstance(d.get("data"), list):
        sys.exit(f"unexpected response: code={d.get('code')} message={d.get('message')!r}")

    if args.server is not None:
        inspect_server(d["data"], args.server, args.student)
        return

    summary, entries = aggregate(d["data"])
    old_summary, old_entries, old_snapshot = read_cache()
    unchanged = (
        old_summary is not None and old_entries is not None
        and canonical_wall(summary, entries) == canonical_wall(old_summary, old_entries)
    )
    old_total = old_summary["total"] if old_summary else 0
    fetched = (f"total={summary['total']} students={len(summary['students'])} "
               f"entry_lists={len(entries)}")
    # Always state which cache file this run reads/writes. Outside systemd,
    # BOND100_DB_PATH is unset and this falls back to the dev data/ DB, so the
    # comparison silently runs against the wrong (often empty) cache.
    print(f"cache db: {DB_PATH}")

    if args.dry_run:
        cached = f"total={old_summary['total']} snapshot={old_snapshot}" if old_summary else "empty"
        print(f"[dry-run] fetched: {fetched}")
        print(f"[dry-run] cached:  {cached}")
        print(f"[dry-run] total: {old_total} -> {summary['total']} (delta {summary['total'] - old_total:+d})")
        if unchanged:
            print(f"[dry-run] wall unchanged; a real run would keep snapshot_date={old_snapshot}")
        else:
            for line in diff_counts(old_summary, summary)[:40]:
                print(f"[dry-run]{line}")
            print(f"[dry-run] wall changed; a real run would write snapshot_date={args.snapshot_date}")
        return

    if unchanged:
        # arona served the same wall again. Keep the old snapshot_date so the
        # frontend shows when the data last actually changed, and journald gets
        # an explicit staleness trail instead of a fake fresh sync.
        print(f"unchanged: {fetched}; wall identical to cache "
              f"(total stays {summary['total']}), snapshot stays {old_snapshot}")
        return

    write_cache(summary, entries, args.snapshot_date)
    changed = len(diff_counts(old_summary, summary)) if old_summary else len(summary["students"])
    print(f"synced: {fetched} (total {old_total} -> {summary['total']}) "
          f"snapshot={args.snapshot_date} changed_students={changed}")


if __name__ == "__main__":
    main()
