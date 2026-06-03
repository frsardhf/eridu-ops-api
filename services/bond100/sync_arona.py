"""Bridge sync: pull arona's rank_by_max_favor_user_info, aggregate, cache.

Bridge model: arona is the single source of truth. We make ONE call, aggregate
the global servers into a wall summary + per-student entries, and cache both as
JSON blobs in bond100_meta. No per-entry rows, no dedup, no moderation storage.

    python3 sync_arona.py --from-file userinfo_full_s9.json   # local, no network
    ARONA_TOKEN='<email>:<token>' python3 sync_arona.py        # live fetch

The endpoint's `server` param is ignored (it always returns the full cross-server
cache, refreshed every 4h on arona's side), so we request once and filter to the
five global servers ourselves.
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_connection, init_db, primary_student_id  # noqa: E402

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
    ap.add_argument("--snapshot-date", default=date.today().isoformat(), help="Snapshot date YYYY-MM-DD.")
    args = ap.parse_args()

    if args.from_file:
        with open(args.from_file, encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = fetch_live()

    if d.get("code") != 200 or not isinstance(d.get("data"), list):
        sys.exit(f"unexpected response: code={d.get('code')} message={d.get('message')!r}")

    summary, entries = aggregate(d["data"])
    write_cache(summary, entries, args.snapshot_date)
    print(f"synced: total={summary['total']} students={len(summary['students'])} "
          f"entry_lists={len(entries)} snapshot={args.snapshot_date}")


if __name__ == "__main__":
    main()
