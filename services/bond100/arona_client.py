"""arona /refresh client + abuse limiting for the submission ("add me") flow.

A submission triggers arona to pull the player's bond-100 data; it then shows up
in the next daily sync. We never store the friend code, only a salted hash for
the cooldown / rate-limit bookkeeping.

Abuse limits, in layers (FE limits are cosmetic; these are the real guards):
  * nginx              -- per-IP request rate (edge; deploy/eridu-api.nginx.conf)
  * per-code cooldown  -- skip if this code was refreshed within arona's 4h cache
  * global hourly cap  -- protect the shared arona quota / token
  * format validation  -- reject junk friend codes before calling arona
"""
import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone

from db import get_connection

REFRESH_URL = "https://api.arona.icu/api/friends/refresh"
FRIEND_CODE_RE = re.compile(r"^[A-Za-z0-9]{4,20}$")
COOLDOWN = timedelta(hours=6)       # > arona's 4h cache; sooner is wasted quota
MAX_REFRESH_PER_HOUR = 60           # global cap across all users


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_hash(server: str, friend_code: str) -> str:
    return hashlib.sha256(f"{server}|{friend_code}".encode("utf-8")).hexdigest()


def valid_friend_code(code) -> bool:
    return isinstance(code, str) and bool(FRIEND_CODE_RE.match(code.strip()))


def _check_rate(conn, code_hash: str) -> tuple[bool, str]:
    """(allowed, reason). DB-backed so it's shared across gunicorn workers."""
    now = datetime.now(timezone.utc)
    row = conn.execute(
        "SELECT refreshed_at FROM bond100_refresh_log WHERE code_hash = ?", (code_hash,)
    ).fetchone()
    if row and (now - datetime.fromisoformat(row["refreshed_at"])) < COOLDOWN:
        return False, "cooldown"
    hour_ago = (now - timedelta(hours=1)).isoformat()
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM bond100_refresh_log WHERE refreshed_at > ?", (hour_ago,)
    ).fetchone()["c"]
    if n >= MAX_REFRESH_PER_HOUR:
        return False, "global_rate"
    return True, ""


def _record(conn, code_hash: str, server: str) -> None:
    conn.execute(
        "INSERT INTO bond100_refresh_log (code_hash, server, refreshed_at) VALUES (?, ?, ?) "
        "ON CONFLICT(code_hash) DO UPDATE SET refreshed_at = excluded.refreshed_at, server = excluded.server",
        (code_hash, server, _now_iso()),
    )
    conn.commit()


def _call_refresh(friend_code: str, timeout: int = 15) -> tuple[bool, str]:
    token = os.environ.get("ARONA_TOKEN")
    if not token:
        return False, "no_token"
    req = urllib.request.Request(
        REFRESH_URL,
        data=json.dumps({"friend": friend_code, "assistType": 0}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"ba-token {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - network/parse errors all map to "try later"
        return False, f"arona_error:{type(e).__name__}"
    return (d.get("code") == 200), str(d.get("code"))


def submit_refresh(server: str, friend_code: str) -> tuple[dict, int]:
    """Validate -> rate-limit -> call arona /refresh. Returns (json_body, http_status)."""
    code = (friend_code or "").strip()
    if not valid_friend_code(code):
        return {"ok": False, "error": "invalid_friend_code"}, 400

    code_hash = _code_hash(server, code)
    conn = get_connection()
    try:
        allowed, reason = _check_rate(conn, code_hash)
        if not allowed:
            # A recent submission already queued this code — treat as success
            # (idempotent from the user's view), but spend no quota.
            if reason == "cooldown":
                return {"ok": True, "queued": False, "message": "already_submitted"}, 200
            return {"ok": False, "error": "rate_limited"}, 429

        ok, _detail = _call_refresh(code)
        if not ok:
            return {"ok": False, "error": "refresh_failed"}, 502
        _record(conn, code_hash, server)
    finally:
        conn.close()
    return {"ok": True, "queued": True}, 201
