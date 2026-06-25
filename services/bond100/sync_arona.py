"""_info baseline sync: pull arona's rank_by_max_favor_user_info into the store.

The _info endpoint is one call for every student, but a DELAYED cache (it froze
for 3+ weeks in the 2026-06 incident), so it's now the BASELINE, not the master:
this sync writes its aggregate as source='info' rows in bond100_student_rank and
reassembles the wall. The live /rank sweep (sweep_rank.py) overwrites those rows
student-by-student, and write_info_rows NEVER clobbers a fresher 'rank' row, so a
re-sync can't undo the sweep's real-time data.

    python3 sync_arona.py --from-file userinfo_full_s9.json   # local, no network
    ARONA_TOKEN='<email>:<token>' python3 sync_arona.py        # live fetch
    python3 sync_arona.py --dry-run                            # report, write nothing
    python3 sync_arona.py --server 8                           # raw per-student counts for ONE server (cross-check vs arona.icu)

The endpoint's `server` param is ignored (it always returns the full cross-server
cache), so we request once and filter to the five global servers ourselves.

Post-cutover its systemd timer is disabled (the sweep owns the wall); this stays
as a manual re-seed / diagnostic tool.
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget  # noqa: E402
import wall_store  # noqa: E402
from db import DB_PATH, GLOBAL_SERVER_IDS, get_connection, init_db, primary_student_id  # noqa: E402

API_URL = "https://api.arona.icu/api/friends/rank_by_max_favor_user_info"
GLOBAL_SERVERS = {v: k for k, v in GLOBAL_SERVER_IDS.items()}


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the Bond 100 _info baseline rows from arona's user-info endpoint.")
    ap.add_argument("--from-file", help="Read a saved userinfo dump instead of fetching (no network).")
    ap.add_argument("--snapshot-date", default=date.today().isoformat(),
                    help="fetched_at stamped on the _info baseline rows (YYYY-MM-DD).")
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
    counts = {s["studentId"]: s["count"] for s in summary["students"]}
    fetched = (f"total={summary['total']} students={len(summary['students'])} "
               f"entry_lists={len(entries)}")
    # Always state which cache file this run reads/writes. Outside systemd,
    # BOND100_DB_PATH is unset and this falls back to the dev data/ DB.
    print(f"cache db: {DB_PATH}")

    init_db()
    conn = get_connection()
    try:
        # A live fetch spent one arona call (even on a dry-run); record it.
        if not args.from_file:
            budget.record_call(conn, "info")
        rank_held = conn.execute(
            "SELECT COUNT(*) AS c FROM bond100_student_rank WHERE source = 'rank'"
        ).fetchone()["c"]

        if args.dry_run:
            print(f"[dry-run] fetched: {fetched}")
            print(f"[dry-run] would refresh {len(entries) - min(rank_held, len(entries))}+ "
                  f"info rows; {rank_held} fresher /rank rows kept untouched. Writes nothing.")
            return

        written = wall_store.write_info_rows(conn, entries, counts, args.snapshot_date)
        conn.commit()
    finally:
        conn.close()

    pub, _ = wall_store.assemble_wall()
    print(f"synced _info baseline: {fetched}; refreshed {written} info rows "
          f"({rank_held} /rank rows kept); published wall total={pub['total']}")


if __name__ == "__main__":
    main()
