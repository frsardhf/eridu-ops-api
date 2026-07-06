"""arona friends/rank client: one student's LIVE bond-100 entries.

Per-student leaderboard, the real-time counterpart to rank_by_max_favor_user_info
("_info"). _info is one call for every student but a delayed cache (it froze for
3+ weeks in the 2026-06 incident); friends/rank is live but per-student, so the
rolling sweep (Phase 4) spreads it across the roster over several days within
arona's daily token budget.

Response shape (probed live against arona):
  data.records[]:
    key        -- per-SNAPSHOT record id; REGENERATES when a player refreshes their
                  arona data, so it's safe for intra-fetch dedup but NOT across time
    server     -- per-server id; we keep the global ones (5-9)
    nickname   -- player name shown on the wall
    assistInfoList[]: { uniqueId, favorRank, rankUpdateTime, ... }  -- per-student
                  bond level. rankUpdateTime is arona's "Recorded Time": fixed when
                  the entry was first recorded and STABLE across refreshes, so it's
                  the cross-time (tail-merge) dedup anchor. The friend code is NOT here.
  data.extension -- bond-100 count for the queried student (the site's 百绊 N badge)
  data.lastPage  -- bool

Notes baked in from the probes:
- The `server` request param is ignored (always returns all servers).
- Bond comes from the per-student assist's favorRank, NOT record.maxFavorRank
  (which is the player's highest bond across ALL their students).
- Records are sorted bond-descending, so bond-100 players are the top prefix and
  `extension` tells us exactly how many to expect (size=200 covers any student in
  one page today; we still paginate defensively).
"""
import argparse
import json
import logging
import os
import sys
import urllib.request

from db import GLOBAL_SERVER_IDS, linked_partner_ids, primary_student_id

log = logging.getLogger("bond100")

API_URL = "https://api.arona.icu/api/friends/rank"
# arona's per-server id -> our region (5->global_eu ... 9->global_na). Inverse of
# db.GLOBAL_SERVER_IDS; records on any other server (China/Japan) are dropped.
ID_TO_REGION = {v: k for k, v in GLOBAL_SERVER_IDS.items()}
PAGE_SIZE = 200       # one page covers any student's bond-100 block (max << 200 today)
GLOBAL_PAGE_SIZE = 50  # arona caps friends/rank at 50 records/page regardless of request
TARGET_BOND = 100
MAX_PAGES = 10        # safety net; a single page is the norm with size=200


def _fetch_page(student_id: int, page: int, token: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"page": page, "size": PAGE_SIZE, "studentId": student_id, "server": 4}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"ba-token {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    if body.get("code") != 200 or not isinstance(body.get("data"), dict):
        raise RuntimeError(f"arona /rank code={body.get('code')} message={body.get('message')!r}")
    return body["data"]


def _target_assist(rec: dict, target_ids: set) -> dict | None:
    """The assist for the queried unit. Match by uniqueId (handles the linked pair,
    where a player's assist is the secondary id); fall back to index 0, which for a
    single-student query is always the queried student."""
    assists = rec.get("assistInfoList") or []
    return next((a for a in assists if a.get("uniqueId") in target_ids),
                assists[0] if assists else None)


def _bond100_entries(records: list, target_ids: set) -> dict:
    """{key: {serverRegion, playerName}} for records at bond 100 on the global
    servers. Keyed by player `key` so callers dedup across pages / linked ids."""
    out: dict[str, dict] = {}
    for rec in records:
        region = ID_TO_REGION.get(rec.get("server"))
        if region is None:
            continue
        assist = _target_assist(rec, target_ids)
        if not assist or assist.get("favorRank") != TARGET_BOND:
            continue
        name = (rec.get("nickname") or "").strip()
        key = rec.get("key")
        if not name or not key:
            continue
        out.setdefault(key, {"serverRegion": region, "playerName": name})
    return out


def _page_dipped(records: list, target_ids: set) -> bool:
    """True once a page's last row drops below bond 100 — the (descending) bond-100
    prefix has ended, so no later page can hold more."""
    if not records:
        return True
    last = _target_assist(records[-1], target_ids)
    return last is not None and (last.get("favorRank") or 0) < TARGET_BOND


def _fetch_one_id(student_id: int, target_ids: set, token: str) -> tuple[dict, int | None, int]:
    """All bond-100 entries for a single studentId, paginating until the bond-100
    prefix ends. Returns ({key: entry}, extension, pages_fetched)."""
    merged: dict[str, dict] = {}
    extension: int | None = None
    page = 1
    while page <= MAX_PAGES:
        data = _fetch_page(student_id, page, token)
        if extension is None and isinstance(data.get("extension"), int):
            extension = data["extension"]
        records = data.get("records") or []
        merged.update(_bond100_entries(records, target_ids))
        if data.get("lastPage") or _page_dipped(records, target_ids):
            break
        page += 1
    return merged, extension, page


def fetch_student(student_id: int, token: str | None = None) -> tuple[int, list, int]:
    """Live bond-100 entries for a student (pass the PRIMARY id). Queries the
    primary plus any linked secondary ids and dedups players by `key`, so one
    player counts once for the unit even if they maxed both alt styles (this
    differs from _info, which double-counts such players). Returns
    (count, entries, calls_made) with entries = [{serverRegion, playerName}]
    sorted stably and count == len(entries). calls_made is the number of arona
    requests issued, for the caller to record against the shared budget. Raises
    on a non-200 arona response."""
    token = token or os.environ.get("ARONA_TOKEN")
    if not token:
        raise RuntimeError("ARONA_TOKEN required for a /rank fetch")

    ids = [student_id, *linked_partner_ids(student_id)]
    target_ids = set(ids)
    merged: dict[str, dict] = {}
    ext_sum = 0
    calls = 0
    for sid in ids:
        per_id, extension, pages = _fetch_one_id(sid, target_ids, token)
        merged.update(per_id)
        calls += pages
        if isinstance(extension, int):
            ext_sum += extension

    entries = sorted(merged.values(), key=lambda e: (e["serverRegion"], e["playerName"]))
    count = len(entries)
    # extension is arona's own count; for a single id it should equal our count.
    # For a linked pair ext_sum can exceed count (dual-style players dedup to one),
    # so only flag the unlinked mismatch, where it signals a parsing drift.
    if len(ids) == 1 and ext_sum and ext_sum != count:
        log.warning("rank %s: extension=%d but parsed %d bond-100 entries", student_id, ext_sum, count)
    return count, entries, calls


def _fetch_global_page(page: int, token: str, size: int = GLOBAL_PAGE_SIZE, timeout: int = 30) -> dict:
    """One page of the GLOBAL (no studentId) bond ranking, sorted bond-descending."""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"page": page, "size": size, "server": 4}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"ba-token {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    if body.get("code") != 200 or not isinstance(body.get("data"), dict):
        raise RuntimeError(f"arona /rank (global) page={page} size={size} "
                           f"code={body.get('code')} message={body.get('message')!r}")
    return body["data"]


def _collect_bond100(records: list, out: list, seen: set) -> bool:
    """Append bond-100 records to `out` (student collapsed to primary, non-global/blank
    dropped), deduped within a fetch by (key, studentId). Each record also carries
    `rut` (rankUpdateTime, arona's stable "recorded time"), which the cross-time tail
    merge dedups on instead: `key` regenerates on refresh, so it can't recognize a
    refreshed player across two fetches, but `rut` stays fixed. Return True once a
    record drops below bond 100 (the sorted block has ended)."""
    for rec in records:
        assists = rec.get("assistInfoList") or []
        assist = assists[0] if assists else None
        fr = assist.get("favorRank") if assist else None
        if fr is None:
            continue
        if fr < TARGET_BOND:
            return True
        region = ID_TO_REGION.get(rec.get("server"))
        if region is None:
            continue
        name = (rec.get("nickname") or "").strip()
        key = rec.get("key")
        rut = assist.get("rankUpdateTime")
        uid = assist.get("uniqueId")
        if not name or not key or rut is None or uid is None:
            continue
        sid = primary_student_id(uid)
        if (key, sid) in seen:
            continue
        seen.add((key, sid))
        out.append({"key": key, "rut": rut, "serverRegion": region, "playerName": name, "studentId": sid})
    return False


def _recover_page(page: int, token: str, out: list, seen: set, budget_calls: int,
                  sub_size: int = 5) -> tuple[int, int]:
    """Salvage a poisoned size-50 page. Re-fetch its exact record range at `sub_size`;
    any sub-chunk that STILL 500s is drilled at size 1, so we recover every innocent
    record and lose ONLY the genuinely-broken position(s) (the ones that 500 even
    alone). Stops if budget_calls is reached (caller then sees an incomplete result
    and won't publish). Returns (lost, calls): lost = records that 500 at size 1.
    Positions align across sizes because the block is stably sorted by rankUpdateTime."""
    per = GLOBAL_PAGE_SIZE // sub_size          # sub-pages covering one size-50 page
    first = (page - 1) * per + 1
    lost = calls = 0
    for q in range(first, first + per):
        if calls >= budget_calls:
            break
        try:
            data = _fetch_global_page(q, token, size=sub_size)
            calls += 1
        except Exception:  # noqa: BLE001 - poisoned chunk; drill it at size 1
            calls += 1
            lo = (q - 1) * sub_size + 1
            for pos in range(lo, lo + sub_size):
                if calls >= budget_calls:
                    break
                try:
                    d1 = _fetch_global_page(pos, token, size=1)
                    calls += 1
                    _collect_bond100(d1.get("records") or [], out, seen)
                except Exception:  # noqa: BLE001 - the genuinely broken record
                    calls += 1
                    lost += 1
            continue
        _collect_bond100(data.get("records") or [], out, seen)
    return lost, calls


def fetch_all_bond100(token: str | None = None, max_calls: int = 200, recover_size: int = 5) -> dict:
    """Fetch the ENTIRE global bond-100 block from friends/rank (no studentId,
    server=4), paging until favorRank drops below 100, then salvaging any page that
    500'd by re-fetching it at `recover_size` (arona's broken-record bug poisons a
    whole 50-page over one record). Accumulates fully in memory; the caller commits
    nothing unless the result is `complete`.

    Returns a dict:
      records   = [{key, rut, serverRegion, playerName, studentId}] bond-100 only.
      extension = arona's reported bond-100 count (page 1).
      calls     = arona requests issued.
      lost      = records that 500 even at size 1 (the genuinely broken records;
                  innocents in a poisoned chunk are drilled out and recovered).
      poisoned  = the size-50 pages that 500'd.
      complete  = len(records) + lost == extension, i.e. every record is accounted
                  for (fetched or genuinely unfetchable) with no budget cutoff.

    Page 1 failing raises. If page 1 shows the block needs more pages than max_calls,
    it stops after page 1 (complete=False)."""
    token = token or os.environ.get("ARONA_TOKEN")
    if not token:
        raise RuntimeError("ARONA_TOKEN required for a /rank fetch")

    out: list = []
    seen: set = set()

    # Page 1 gives extension; if it fails the endpoint is down -> raise.
    data = _fetch_global_page(1, token)
    calls = 1
    extension = data.get("extension") or 0
    needed = -(-extension // GLOBAL_PAGE_SIZE) if extension else 1   # ceil(extension/50)
    if needed > max_calls:
        return {"records": out, "extension": extension, "calls": calls,
                "lost": 0, "poisoned": [], "complete": False}

    poisoned: list = []
    dipped = _collect_bond100(data.get("records") or [], out, seen)
    page = 2
    while not dipped and page <= needed + 2 and calls < max_calls:  # +2 slack for the boundary
        try:
            data = _fetch_global_page(page, token)
        except Exception:  # noqa: BLE001 - a poisoned page; note it, recover it below
            poisoned.append(page)
            calls += 1
            page += 1
            continue
        calls += 1
        dipped = _collect_bond100(data.get("records") or [], out, seen)
        page += 1

    # Recovery pass: salvage each poisoned page (size sub -> size 1). Bounded by the
    # remaining budget; a page truncated by budget leaves the result incomplete, so
    # the caller won't publish (accounting stays exact: collected + lost == extension
    # only when every record was fetched or isolated as genuinely broken).
    lost = 0
    for p in poisoned:
        if calls >= max_calls:
            break
        pl, pc = _recover_page(p, token, out, seen, max_calls - calls, recover_size)
        lost += pl
        calls += pc

    return {"records": out, "extension": extension, "calls": calls,
            "lost": lost, "poisoned": poisoned, "complete": len(out) + lost == extension}


def fetch_tail(stored_extension: int, new_extension: int, token: str,
               max_calls: int = 20) -> tuple[list, int, int]:
    """Fetch the bond-100 records added since `stored_extension`. New achievers have
    the newest rankUpdateTime, so they sit at the TAIL (positions stored+1..new). We
    re-fetch from the page holding stored+1 through the page holding `new`, recovering
    any poisoned tail page. Returns (records, calls, lost); the caller merges records
    into the store deduped by rankUpdateTime (`rut`), which survives the refreshes that
    the overlap region is full of (a refresher keeps its rut but gets a new key)."""
    out: list = []
    seen: set = set()
    lost = calls = 0
    first_page = stored_extension // GLOBAL_PAGE_SIZE + 1
    last_page = -(-new_extension // GLOBAL_PAGE_SIZE)   # ceil(new_extension/50)
    for p in range(first_page, last_page + 1):
        if calls >= max_calls:
            break
        try:
            data = _fetch_global_page(p, token)
            calls += 1
        except Exception:  # noqa: BLE001 - poisoned tail page; recover it
            calls += 1
            pl, pc = _recover_page(p, token, out, seen, max_calls - calls)
            lost += pl
            calls += pc
            continue
        _collect_bond100(data.get("records") or [], out, seen)
    return out, calls, lost


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch one student's live bond-100 entries from arona friends/rank.")
    ap.add_argument("--student", type=int, required=True, help="Primary SchaleDB student id (e.g. 10135).")
    ap.add_argument("--names", action="store_true", help="List player names, not just the per-server tally.")
    args = ap.parse_args()

    count, entries, calls = fetch_student(args.student)
    by_region: dict[str, int] = {}
    for e in entries:
        by_region[e["serverRegion"]] = by_region.get(e["serverRegion"], 0) + 1
    print(f"student {args.student}: bond100={count} byServer={by_region} (arona calls: {calls})")
    if args.names:
        for e in entries:
            print(f"  {e['serverRegion']:>12}  {e['playerName']}")


if __name__ == "__main__":
    main()
