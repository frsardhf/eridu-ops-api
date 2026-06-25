"""arona /refresh client + abuse limiting for the submission ("add me") flow.

A submission triggers arona to pull the player's bond-100 data; it then shows up
in the next daily sync. We never store the friend code, only a salted hash for
the cooldown / rate-limit bookkeeping.

Abuse limits, in layers (FE limits are cosmetic; these are the real guards):
  * nginx              -- per-IP request rate (edge; deploy/eridu-api.nginx.conf)
  * per-code cooldown  -- skip if this code was refreshed within arona's 4h cache
  * shared daily budget-- the bond100_api_log ledger (budget.py); /refresh shares
                          arona's ~60/day token with the /rank sweep + _info sync
  * format validation  -- reject junk friend codes before calling arona
"""
import hashlib
import json
import logging
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone

import budget
from db import GLOBAL_SERVER_IDS, get_connection

log = logging.getLogger("bond100")

REFRESH_URL = "https://api.arona.icu/api/friends/refresh"
# arona's docs describe a coarse "server group" for the friend interfaces
# (1=China, 3=Japan, 4=International), but a live probe disproved it for a
# real Global/Asia friend code: group 4 got 3009 (rejected), only the specific
# per-server id (8, i.e. GLOBAL_SERVER_IDS["global_asia"]) got 200. So /refresh
# wants the same per-server id as `_info` and /findRank, not the group value.
# arona made `server` required on /refresh in 2026-06; omitting it gets
# rejected outright.
FRIEND_CODE_RE = re.compile(r"^[A-Za-z0-9]{4,20}$")
COOLDOWN = timedelta(hours=6)       # > arona's 4h cache; sooner is wasted quota
# The global daily cap now lives in the shared ledger (budget.py): /refresh is
# allowed while the combined arona call count is under budget.CEILING, and the
# sweep self-limits below that so submissions keep a reserved slice.


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_hash(server: str, friend_code: str) -> str:
    return hashlib.sha256(f"{server}|{friend_code}".encode("utf-8")).hexdigest()


def valid_friend_code(code) -> bool:
    return isinstance(code, str) and bool(FRIEND_CODE_RE.match(code.strip()))


def _check_rate(conn, code_hash: str) -> tuple[bool, str]:
    """(allowed, reason). DB-backed so it's shared across gunicorn workers. The
    per-code cooldown lives in bond100_refresh_log; the global daily cap is the
    shared arona budget (counts every call kind, not just refreshes)."""
    now = datetime.now(timezone.utc)
    row = conn.execute(
        "SELECT refreshed_at FROM bond100_refresh_log WHERE code_hash = ?", (code_hash,)
    ).fetchone()
    if row and (now - datetime.fromisoformat(row["refreshed_at"])) < COOLDOWN:
        return False, "cooldown"
    if not budget.refresh_allowed(conn):
        return False, "global_rate"
    return True, ""


def _record(conn, code_hash: str, server: str) -> None:
    conn.execute(
        "INSERT INTO bond100_refresh_log (code_hash, server, refreshed_at) VALUES (?, ?, ?) "
        "ON CONFLICT(code_hash) DO UPDATE SET refreshed_at = excluded.refreshed_at, server = excluded.server",
        (code_hash, server, _now_iso()),
    )
    conn.commit()


def _call_refresh(conn, friend_code: str, server_id: int, timeout: int = 15) -> tuple[bool, str]:
    token = os.environ.get("ARONA_TOKEN")
    if not token:
        log.error("arona /refresh skipped: ARONA_TOKEN not set")
        return False, "no_token"
    req = urllib.request.Request(
        REFRESH_URL,
        data=json.dumps({"friend": friend_code, "assistType": 0, "server": server_id}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"ba-token {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - network/parse errors all map to "try later"
        log.warning("arona /refresh transport error: %s", type(e).__name__)
        return False, f"arona_error:{type(e).__name__}"
    # The request reached arona and spent a token call (even on a non-200 code),
    # so it counts against the shared daily budget.
    budget.record_call(conn, "refresh")
    code = d.get("code")
    if code != 200:
        log.warning("arona /refresh non-success: code=%s message=%s", code, d.get("message"))
    return (code == 200), str(code)


def submit_refresh(server: str, friend_code: str) -> tuple[dict, int]:
    """Validate -> rate-limit -> call arona /refresh. Returns (json_body, http_status)."""
    code = (friend_code or "").strip()
    if not valid_friend_code(code):
        log.info("submission rejected: invalid_friend_code server=%s", server)
        return {"ok": False, "error": "invalid_friend_code"}, 400

    code_hash = _code_hash(server, code)
    ref = code_hash[:10]  # correlation id for logs — never the raw friend code
    conn = get_connection()
    try:
        allowed, reason = _check_rate(conn, code_hash)
        if not allowed:
            # A recent submission already queued this code — treat as success
            # (idempotent from the user's view), but spend no quota.
            if reason == "cooldown":
                log.info("submission deduped (cooldown) server=%s ref=%s", server, ref)
                return {"ok": True, "queued": False, "message": "already_submitted"}, 200
            log.warning("submission rate_limited (global daily cap) server=%s ref=%s", server, ref)
            return {"ok": False, "error": "rate_limited"}, 429

        ok, detail = _call_refresh(conn, code, GLOBAL_SERVER_IDS[server])
        if not ok:
            log.warning("submission refresh_failed server=%s ref=%s detail=%s", server, ref, detail)
            return {"ok": False, "error": "refresh_failed"}, 502
        _record(conn, code_hash, server)
        log.info("submission queued server=%s ref=%s", server, ref)
    finally:
        conn.close()
    return {"ok": True, "queued": True}, 201
