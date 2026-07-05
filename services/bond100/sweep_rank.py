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


def run_sweep(limit: int, dry_run: bool, publish: bool = True,
              only_ids: list[int] | None = None) -> None:
    init_db()
    token = os.environ.get("ARONA_TOKEN")
    if not token and not dry_run:
        sys.exit("ARONA_TOKEN required for a live sweep")

    conn = get_connection()
    fetched = 0
    try:
        if only_ids:
            # Targeted mode: fetch exactly these ids now, bypassing the roster
            # (no SchaleDB call), the stalest ordering, and the per-run limit /
            # sweep-budget gate. Still records the arona calls so the budget stays
            # accurate. For a handful of deliberate ids.
            order = [primary_student_id(s) for s in only_ids]
            gated = False
            print(f"targeted fetch: {order}; "
                  f"budget now {budget.calls_in_window(conn)}/{budget.CEILING}")
            if dry_run:
                print(f"[dry-run] would fetch {order}; writes nothing, no calls.")
                return
        else:
            roster = fetch_roster()
            order = order_stalest_first(conn, roster)
            gated = True
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
        changed = 0
        for sid in order:
            if gated and (fetched >= limit or budget.sweep_budget_left(conn) <= 0):
                break
            try:
                count, entries, calls = rank_client.fetch_student(sid, token)
            except Exception as e:  # noqa: BLE001 - skip a bad student, keep sweeping
                # Log the full reason (arona code/message, e.g. a 500 for alt-style
                # ids), not just the exception type, so failures are diagnosable.
                log.warning("sweep: student %s fetch failed: %s", sid, e)
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
            # Roster sweep logs only changes (keeps 40-student runs quiet);
            # targeted mode always shows the result.
            if old != count or not gated:
                print(f"  {sid}: {old if old is not None else '-'} -> {count}")
        print(f"sweep done: fetched {fetched} students ({changed} changed); "
              f"budget now {budget.calls_in_window(conn)}/{budget.CEILING}.")
    finally:
        conn.close()

    # Publish: rebuild the served wall blobs from the table (swept + seeded rows).
    # Skipped on a no-op run (nothing fetched) and when --no-publish is passed.
    if publish and fetched > 0:
        psummary, _ = wall_store.assemble_wall()
        print(f"published: served wall total={psummary['total']} "
              f"({len(psummary['students'])} students)")


def run_global(publish: bool = True) -> None:
    """Full global bond-100 refresh: one /rank stream (no studentId) -> aggregate
    per student -> ATOMICALLY replace the table -> publish. ~ceil(bond100/50) calls.

    Atomic: the whole bond-100 block is fetched + validated (collected count ==
    arona's `extension`) in memory before any DB write, and the replace runs in a
    single transaction. A cut, failed, or budget-limited fetch commits nothing, so
    the served wall is never left partial."""
    init_db()
    token = os.environ.get("ARONA_TOKEN")
    if not token:
        sys.exit("ARONA_TOKEN required for a global fetch")

    conn = get_connection()
    try:
        budget_left = budget.sweep_budget_left(conn)
        try:
            records, extension, calls, complete = rank_client.fetch_all_bond100(token, max_calls=budget_left)
        except Exception as e:  # noqa: BLE001 - network/parse; nothing written, retry next run
            print(f"global fetch failed mid-stream: {e}; served wall unchanged.")
            return
        budget.record_call(conn, "rank", calls)
        if not complete:
            print(f"global fetch incomplete: collected {len(records)}/{extension} in {calls} calls "
                  f"(sweep budget left was {budget_left}); served wall unchanged.")
            conn.commit()
            return

        by_student: dict[int, list] = {}
        for r in records:
            by_student.setdefault(r["studentId"], []).append(
                {"serverRegion": r["serverRegion"], "playerName": r["playerName"]})
        today = date.today().isoformat()
        conn.execute("DELETE FROM bond100_student_rank")
        for sid, entries in by_student.items():
            wall_store.upsert_student(conn, sid, len(entries), entries, "rank", today)
        conn.commit()
        print(f"global fetch: {len(records)} bond-100 across {len(by_student)} students in {calls} calls; "
              f"budget now {budget.calls_in_window(conn)}/{budget.CEILING}")
    finally:
        conn.close()

    if publish:
        summary, _ = wall_store.assemble_wall()
        print(f"published: served wall total={summary['total']} ({len(summary['students'])} students)")


def report_store(limit: int) -> None:
    """Read-only: list the stalest students, split by whether they have bond-100
    entries. No arona calls, just the local store, for watching sweep coverage."""
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT student_id, count, fetched_at, source FROM bond100_student_rank "
            "ORDER BY fetched_at ASC, student_id ASC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("store is empty.")
        return

    dist: dict[str, int] = {}
    for r in rows:
        dist[r["fetched_at"]] = dist.get(r["fetched_at"], 0) + 1
    print(f"store: {len(rows)} students | by fetched_at: "
          + ", ".join(f"{d}={n}" for d, n in sorted(dist.items())))

    for title, subset in (
        ("WITH entries (count>0)", [r for r in rows if r["count"] > 0]),
        ("NO entries (count==0)", [r for r in rows if r["count"] == 0]),
    ):
        shown = subset[:limit]
        more = len(subset) - len(shown)
        print(f"\n{title} - {len(subset)} students, stalest first"
              + (f" (showing {limit})" if more > 0 else "") + ":")
        for r in shown:
            print(f"  {r['fetched_at']}  {r['student_id']:>6}  count={r['count']:>3}  {r['source']}")
        if more > 0:
            print(f"  ... {more} more")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling per-student bond-100 sweep via arona friends/rank.")
    ap.add_argument("--limit", type=int, default=budget.SWEEP_LIMIT,
                    help=f"Max students to fetch this run (default {budget.SWEEP_LIMIT}; also bounded by the shared budget).")
    ap.add_argument("--dry-run", action="store_true", help="Show roster + what would be fetched; write nothing, no calls.")
    ap.add_argument("--no-publish", action="store_false", dest="publish",
                    help="Fetch + write rows but do NOT rebuild the served wall (validation only).")
    ap.add_argument("--student", type=int, action="append", metavar="ID",
                    help="Fetch specific student id(s) now (repeatable), bypassing the roster, "
                         "stalest ordering, and per-run limit. Still records the call + publishes.")
    ap.add_argument("--report", action="store_true",
                    help="Read-only: list the stalest students split by entries / no entries "
                         "(rows shown per group capped by --limit). No arona calls.")
    ap.add_argument("--global", dest="global_fetch", action="store_true",
                    help="Full global bond-100 refresh via one /rank stream (no studentId): "
                         "atomically replaces the whole wall. ~ceil(bond100/50) calls.")
    args = ap.parse_args()
    if args.report:
        report_store(args.limit)
        return
    if args.global_fetch:
        run_global(args.publish)
        return
    run_sweep(args.limit, args.dry_run, args.publish, only_ids=args.student)


if __name__ == "__main__":
    main()
