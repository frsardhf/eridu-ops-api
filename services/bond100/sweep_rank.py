"""Rolling /rank sweep: refresh the bond-100 wall student-by-student.

arona's _info endpoint (one call, all students) is a delayed cache that froze for
weeks (2026-06 incident); friends/rank is live but per-student. This sweep walks
the global roster a slice at a time, staying within the shared daily budget
(budget.py), so the whole wall refreshes every ~week while leaving a reserved
slice for user submissions.

Each run:
  1. roster = SchaleDB JP students.json (the id superset) filtered to
     IsReleased[Global], minus the linked secondaries (swept with their primary).
  2. order stalest-first: never-fetched students lead, so a newly released student
     with real bond-100 players (which the frozen _info missed, e.g. 10135) gets
     picked up within one cycle; then by oldest fetched_at.
  3. fetch each via rank_client, record the calls against the shared budget, upsert
     the row, until the per-run limit or the sweep budget is spent.

It writes rows ONLY; it does not reassemble the served wall. That serving cutover
is wired separately (Phase 5) so the live Hall stays stable until we flip it.

    python3 sweep_rank.py --dry-run     # roster + what would be fetched, no calls
    python3 sweep_rank.py --limit 3     # fetch 3 stalest (live; spends budget)
    python3 sweep_rank.py               # fetch up to the sweep budget
"""
import argparse
import json
import logging
import os
import sys
import urllib.request
from datetime import date

import budget
import rank_client
import wall_store
from db import get_connection, init_db, primary_student_id

log = logging.getLogger("bond100")

ROSTER_URL = "https://schaledb.com/data/jp/students.json"   # JP = the id superset
GLOBAL_RELEASE_IDX = 1   # IsReleased = [JP, Global, CN]


def fetch_roster(timeout: int = 30) -> list[int]:
    """Global-released PRIMARY student ids from SchaleDB. Secondaries are dropped;
    they're swept as part of their primary's fetch. Not an arona call (SchaleDB
    CDN), so it costs no token budget."""
    req = urllib.request.Request(ROSTER_URL, headers={"User-Agent": "eridu-ops-bond100"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    roster = []
    for sid_str, s in data.items():
        rel = s.get("IsReleased")
        if not (isinstance(rel, list) and len(rel) > GLOBAL_RELEASE_IDX and rel[GLOBAL_RELEASE_IDX]):
            continue
        sid = int(sid_str)
        if primary_student_id(sid) != sid:   # a linked secondary -> swept via primary
            continue
        roster.append(sid)
    return roster


def order_stalest_first(conn, roster: list[int]) -> list[int]:
    """Roster ordered stalest-first. Never-fetched students sort before any
    fetched one ('' < any ISO date), so 0-count newcomers _info missed lead;
    then by oldest fetched_at, tie-broken by id."""
    fetched = {
        r["student_id"]: r["fetched_at"]
        for r in conn.execute("SELECT student_id, fetched_at FROM bond100_student_rank")
    }
    return sorted(roster, key=lambda sid: (fetched.get(sid, ""), sid))


def run_sweep(limit: int, dry_run: bool) -> None:
    init_db()
    token = os.environ.get("ARONA_TOKEN")
    if not token and not dry_run:
        sys.exit("ARONA_TOKEN required for a live sweep")

    roster = fetch_roster()
    conn = get_connection()
    try:
        order = order_stalest_first(conn, roster)
        budget_left = budget.sweep_budget_left(conn)
        print(f"roster: {len(roster)} global-released students; "
              f"sweep budget left: {budget_left}; per-run limit: {limit}")

        if dry_run:
            n = min(limit, budget_left)
            preview = order[:n]
            print(f"[dry-run] would fetch {len(preview)} stalest: "
                  f"{preview[:20]}{' ...' if len(preview) > 20 else ''}")
            return

        today = date.today().isoformat()
        fetched = changed = 0
        for sid in order:
            if fetched >= limit or budget.sweep_budget_left(conn) <= 0:
                break
            try:
                count, entries, calls = rank_client.fetch_student(sid, token)
            except Exception as e:  # noqa: BLE001 - skip a bad student, keep sweeping
                log.warning("sweep: student %s fetch failed: %s", sid, type(e).__name__)
                continue
            budget.record_call(conn, "rank", calls)
            row = conn.execute(
                "SELECT count FROM bond100_student_rank WHERE student_id = ?", (sid,)
            ).fetchone()
            old = row["count"] if row else None
            wall_store.upsert_student(conn, sid, count, entries, "rank", today)
            conn.commit()
            fetched += 1
            if old != count:
                changed += 1
                print(f"  {sid}: {old if old is not None else '-'} -> {count}")
        print(f"sweep done: fetched {fetched} students ({changed} changed); "
              f"budget now {budget.calls_in_window(conn)}/{budget.CEILING}. "
              f"Run wall_store.py to publish (Phase 5 cutover).")
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling per-student bond-100 sweep via arona friends/rank.")
    ap.add_argument("--limit", type=int, default=budget.SWEEP_LIMIT,
                    help=f"Max students to fetch this run (default {budget.SWEEP_LIMIT}; also bounded by the shared budget).")
    ap.add_argument("--dry-run", action="store_true", help="Show roster + what would be fetched; write nothing, no calls.")
    args = ap.parse_args()
    run_sweep(args.limit, args.dry_run)


if __name__ == "__main__":
    main()
