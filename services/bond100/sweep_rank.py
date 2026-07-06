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


def run_global(publish: bool = True, force: bool = False) -> None:
    """Full global bond-100 refresh: one /rank stream (no studentId) -> aggregate
    per student -> ATOMICALLY replace the table -> publish. ~ceil(bond100/50) calls.

    Atomic: the whole bond-100 block is fetched + validated (collected count ==
    arona's `extension`) in memory before any DB write, and the replace runs in a
    single transaction. A cut, failed, or budget-limited fetch commits nothing, so
    the served wall is never left partial.

    force=True bypasses OUR sweep-budget gate (for a manual run when the window is
    spent). It can't bypass arona's own daily limit: if arona rate-limits mid-fetch
    the fetch just fails atomically and the wall is left unchanged."""
    init_db()
    token = os.environ.get("ARONA_TOKEN")
    if not token:
        sys.exit("ARONA_TOKEN required for a global fetch")

    conn = get_connection()
    try:
        budget_left = budget.sweep_budget_left(conn)
        max_calls = 100 if force else budget_left
        if force:
            print(f"--force: ignoring the sweep budget gate "
                  f"(sweep budget left={budget_left}, window {budget.calls_in_window(conn)}/{budget.CEILING}).")
        try:
            res = rank_client.fetch_all_bond100(token, max_calls=max_calls)
        except Exception as e:  # noqa: BLE001 - page 1 down / network; nothing written, retry next run
            print(f"global fetch failed on page 1: {e}; served wall unchanged.")
            return
        records, extension, calls = res["records"], res["extension"], res["calls"]
        lost, poisoned, complete = res["lost"], res["poisoned"], res["complete"]
        budget.record_call(conn, "rank", calls)
        if poisoned:
            print(f"global fetch: {len(poisoned)} page(s) {poisoned} 500'd (arona's broken-record bug); "
                  f"recovered innocents at finer size, lost {lost} record(s) to the poison.")
        if not complete:
            print(f"global fetch incomplete: {len(records)} fetched + {lost} lost != {extension} "
                  f"in {calls} calls (sweep budget left was {budget_left}); served wall unchanged.")
            conn.commit()
            return

        by_student: dict[int, list] = {}
        for r in records:
            by_student.setdefault(r["studentId"], []).append(
                {"serverRegion": r["serverRegion"], "playerName": r["playerName"],
                 "key": r["key"], "rut": r["rut"]})
        today = date.today().isoformat()
        conn.execute("DELETE FROM bond100_student_rank")
        for sid, entries in by_student.items():
            wall_store.upsert_student(conn, sid, len(entries), entries, "rank", today)
        # rank_extension = arona's bond-100 count; the tail-fetch reads it to know
        # where the new records start (positions stored_extension+1 .. new).
        conn.execute(
            "INSERT INTO bond100_meta (key, value) VALUES ('rank_extension', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (str(extension),))
        conn.commit()
        print(f"global fetch: {len(records)} bond-100"
              + (f" (+{lost} lost to arona's broken records)" if lost else "")
              + f" across {len(by_student)} students in {calls} calls; "
              f"budget now {budget.calls_in_window(conn)}/{budget.CEILING}")
    finally:
        conn.close()

    if publish:
        summary, _ = wall_store.assemble_wall()
        print(f"published: served wall total={summary['total']} ({len(summary['students'])} students)")


def run_tail(publish: bool = True, force: bool = False) -> None:
    """Cheap incremental refresh. New bond-100 achievers have the newest
    rankUpdateTime, so they land at the tail of the stably-sorted block: read the
    current count (page 1), and if it GREW, fetch only the tail pages holding the new
    records and merge them by rankUpdateTime (stable across the refreshes that fill the
    overlap region; `key` is not). ~2-4 calls vs a full ~48. A SHRINK (a
    player dropped out) can't be localized, so it tells you to run --global for a
    full resync. Requires a prior --global (rank_extension seeded)."""
    init_db()
    token = os.environ.get("ARONA_TOKEN")
    if not token:
        sys.exit("ARONA_TOKEN required for a tail fetch")

    conn = get_connection()
    did_change = False
    try:
        row = conn.execute("SELECT value FROM bond100_meta WHERE key = 'rank_extension'").fetchone()
        if not row:
            sys.exit("no rank_extension yet; run --global once to seed the store first.")
        stored_e = int(row["value"])

        budget_left = 100 if force else budget.sweep_budget_left(conn)
        if budget_left < 2:
            print(f"tail: no sweep budget (window {budget.calls_in_window(conn)}/{budget.CEILING}); skipped.")
            return

        try:
            data = rank_client._fetch_global_page(1, token)
        except Exception as e:  # noqa: BLE001 - page 1 down; nothing written
            print(f"tail: page 1 failed: {e}; wall unchanged.")
            return
        budget.record_call(conn, "rank", 1)
        new_e = data.get("extension") or 0

        if new_e < stored_e:
            print(f"tail: bond-100 count dropped {stored_e} -> {new_e} (a player left); a removal can't "
                  f"be localized -> run --global for a full resync. Wall unchanged.")
            return
        if new_e == stored_e:
            print(f"tail: count unchanged ({new_e}); no new bond-100. Wall unchanged.")
            return

        recs, calls, lost = rank_client.fetch_tail(stored_e, new_e, token, max_calls=budget_left - 1)
        budget.record_call(conn, "rank", calls)

        # Merge new records into the store, deduped by rankUpdateTime (`rut`, arona's
        # stable "recorded time") within each student. The tail's overlap region is
        # full of refreshers, and a refresh regenerates `key` while leaving `rut`
        # fixed, so a key-dedup would re-add every refresher (the old +30 over-count);
        # rut recognizes them. Needs a store re-seeded by a --global that wrote `rut`.
        existing: dict[int, tuple[list, set]] = {}
        for r in conn.execute("SELECT student_id, entries FROM bond100_student_rank"):
            ents = json.loads(r["entries"])
            existing[r["student_id"]] = (ents, {e.get("rut") for e in ents})
        today = date.today().isoformat()
        touched: set = set()
        added = 0
        for rec in recs:
            ents, ruts = existing.setdefault(rec["studentId"], ([], set()))
            if rec["rut"] in ruts:
                continue
            ents.append({"serverRegion": rec["serverRegion"], "playerName": rec["playerName"],
                         "key": rec["key"], "rut": rec["rut"]})
            ruts.add(rec["rut"])
            touched.add(rec["studentId"])
            added += 1
        for sid in touched:
            ents, _ = existing[sid]
            wall_store.upsert_student(conn, sid, len(ents), ents, "rank", today)
        conn.execute("UPDATE bond100_meta SET value = ? WHERE key = 'rank_extension'", (str(new_e),))
        conn.commit()
        did_change = added > 0
        print(f"tail: {stored_e} -> {new_e} (+{added} new bond-100 across {len(touched)} students"
              + (f", {lost} lost to poison" if lost else "")
              + f") in {calls + 1} calls; budget now {budget.calls_in_window(conn)}/{budget.CEILING}")
    finally:
        conn.close()

    if publish and did_change:
        summary, _ = wall_store.assemble_wall()
        print(f"published: served wall total={summary['total']} ({len(summary['students'])} students)")


def diagnose_page(page: int) -> None:
    """Pinpoint arona's broken record(s) on a poisoned page: re-fetch its 50-record
    range at size 5 to find the 500-ing chunk, then at size 1 to isolate the exact
    position that 500s (the genuine broken record) and print the innocent players
    recoverable around it. ~15 arona calls. Positions are global bond-100 ranks."""
    token = os.environ.get("ARONA_TOKEN")
    if not token:
        sys.exit("ARONA_TOKEN required")
    size50 = rank_client.GLOBAL_PAGE_SIZE
    base = (page - 1) * size50
    print(f"page {page} = global bond-100 positions {base + 1}..{base + size50}; scanning at size 5...")
    per5 = size50 // 5
    first5 = (page - 1) * per5 + 1
    bad = []
    for q in range(first5, first5 + per5):
        try:
            rank_client._fetch_global_page(q, token, size=5)
        except Exception:  # noqa: BLE001
            bad.append(q)
            print(f"  size-5 chunk page {q} (positions {(q - 1) * 5 + 1}..{q * 5}): 500")
    if not bad:
        print("  no chunk 500'd now (transient?); nothing to isolate.")
        return
    for q in bad:
        lo = (q - 1) * 5 + 1
        print(f"  isolating positions {lo}..{lo + 4} at size 1:")
        for pos in range(lo, lo + 5):
            try:
                d = rank_client._fetch_global_page(pos, token, size=1)
                rec = (d.get("records") or [None])[0]
                if not rec:
                    print(f"    pos {pos}: OK (empty)")
                    continue
                a = (rec.get("assistInfoList") or [{}])[0]
                print(f"    pos {pos}: OK  student={primary_student_id(a.get('uniqueId'))} "
                      f"favorRank={a.get('favorRank')} server={rec.get('server')} nick={rec.get('nickname')!r}")
            except Exception:  # noqa: BLE001
                print(f"    pos {pos}: 500  <-- BROKEN record (arona can't serve this one)")


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
    ap.add_argument("--tail", action="store_true",
                    help="Cheap incremental refresh: fetch only the newly-added bond-100 records "
                         "at the tail and merge them. ~2-4 calls. Needs a prior --global.")
    ap.add_argument("--force", action="store_true",
                    help="With --global: bypass OUR sweep-budget gate (manual run when the window "
                         "is spent). Cannot bypass arona's own daily limit.")
    ap.add_argument("--diagnose-page", type=int, metavar="N",
                    help="Drill poisoned page N (size 5 -> size 1) to pinpoint the broken record(s) "
                         "and show the innocent players around it. ~15 arona calls.")
    args = ap.parse_args()
    if args.report:
        report_store(args.limit)
        return
    if args.diagnose_page:
        diagnose_page(args.diagnose_page)
        return
    if args.global_fetch:
        run_global(args.publish, force=args.force)
        return
    if args.tail:
        run_tail(args.publish, force=args.force)
        return
    run_sweep(args.limit, args.dry_run, args.publish, only_ids=args.student)


if __name__ == "__main__":
    main()
