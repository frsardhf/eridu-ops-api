"""Shared daily arona call budget for the Bond 100 service.

arona's token allows ~60 calls/day, shared by every path that hits arona: the
submission /refresh flow, the rolling /rank sweep, and the daily _info sync. A
single ledger (bond100_api_log) records each call so the paths can't collectively
overrun the cap. Two guardrails:

  * CEILING (55) keeps a safety margin under arona's ~60.
  * REFRESH_RESERVE (15) is held back for user-facing submissions, so a full
    sweep can never starve "add me". The sweep self-limits to SWEEP_LIMIT (40);
    /refresh may use the whole ceiling.

Observed submission volume is low single digits/day (see bond100_refresh_log), so
the 15-call reserve comfortably covers real demand while the sweep takes the rest.
"""
import argparse
from datetime import datetime, timedelta, timezone

from db import get_connection, init_db

CEILING = 55           # stay safely under arona's ~60/day
REFRESH_RESERVE = 15   # always free for user-facing submissions
SWEEP_LIMIT = CEILING - REFRESH_RESERVE   # 40: the sweep never pushes past this
# Rolling window for the shared cap. MUST be shorter than the daily sweep cadence
# (24h): with a 24h window, a full ~40-call sweep is still "in window" when the
# next daily run fires ~24h later, so that run sees 0 budget and fetches nothing
# (coverage then crawls, alternating 40/0 per day). 20h leaves a clear margin so
# each daily run starts with a fresh sweep budget.
WINDOW_HOURS = 20


def record_call(conn, kind: str, count: int = 1) -> None:
    """Log `count` arona calls of one kind (a linked-pair /rank fetch spends 2).
    Commit immediately so concurrent gunicorn workers and the sweep see an
    up-to-date count."""
    if count <= 0:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO bond100_api_log (kind, called_at) VALUES (?, ?)",
        [(kind, now)] * count,
    )
    conn.commit()


def calls_in_window(conn, hours: int = WINDOW_HOURS) -> int:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return conn.execute(
        "SELECT COUNT(*) AS c FROM bond100_api_log WHERE called_at > ?", (since,)
    ).fetchone()["c"]


def refresh_allowed(conn) -> bool:
    """User-facing /refresh may use the full ceiling (including the reserve)."""
    return calls_in_window(conn) < CEILING


def sweep_budget_left(conn) -> int:
    """How many more calls the rolling sweep may make right now, leaving the
    refresh reserve untouched."""
    return max(0, SWEEP_LIMIT - calls_in_window(conn))


def usage(conn) -> dict:
    used = calls_in_window(conn)
    by_kind = {
        r["kind"]: r["c"]
        for r in conn.execute(
            "SELECT kind, COUNT(*) AS c FROM bond100_api_log WHERE called_at > ? GROUP BY kind",
            ((datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat(),),
        )
    }
    return {
        "used": used,
        "byKind": by_kind,
        "ceiling": CEILING,
        "refreshAllowed": used < CEILING,
        "sweepBudgetLeft": max(0, SWEEP_LIMIT - used),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=f"Show the shared arona call budget (rolling {WINDOW_HOURS}h).")
    ap.add_argument("--hours", type=int, default=WINDOW_HOURS, help="Window size in hours.")
    ap.parse_args()
    init_db()
    conn = get_connection()
    try:
        u = usage(conn)
    finally:
        conn.close()
    print(f"arona calls (last {WINDOW_HOURS}h): {u['used']}/{u['ceiling']}  by kind: {u['byKind']}")
    print(f"refresh allowed: {u['refreshAllowed']}   sweep budget left: {u['sweepBudgetLeft']}")


if __name__ == "__main__":
    main()
