"""Bond 100 Hall API — lightweight SQLite-backed Flask service.

Runs as its own process (port 5002) separate from the inventory_parser, behind
the same nginx. Route paths include the /bond100 prefix; nginx proxies the
prefix through without stripping.

Bridge model: arona.icu is the single source. sync_arona.py caches the wall, and
these endpoints serve it. Submissions trigger an arona /refresh; removal is
handled on arona's side (the frontend shows guidelines), so there's no removal
endpoint or moderation queue here.

    GET  /bond100/summary
    GET  /bond100/students/<id>/entries
    GET  /bond100/health
    POST /bond100/submissions
"""
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Submission outcomes (incl. arona /refresh failure reasons) log through the
# "bond100" logger; INFO+ to stderr so journald captures it under the service.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Load .env from the repo root (two dirs up: services/bond100/app.py).
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SERVICE_DIR))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from db import init_db, SERVER_REGIONS  # noqa: E402
import arona_client  # noqa: E402
import repository  # noqa: E402

app = Flask(__name__)
CORS(app, origins=["https://eriduops.com", "http://localhost:5173"])

init_db()

MAX_TEXT = 300     # friend code length guard


# ── Public reads ─────────────────────────────────────────────────────────────

@app.get("/bond100/summary")
def summary():
    return jsonify(repository.get_summary())


@app.get("/bond100/students/<int:student_id>/entries")
def student_entries(student_id: int):
    return jsonify(repository.get_student_entries(student_id))


@app.get("/bond100/health")
def health():
    return jsonify({"ok": True})


# ── Submission ("add me") — triggers a rate-limited arona /refresh ───────────

@app.post("/bond100/submissions")
def create_submission():
    """A player asks to be listed. We trigger an arona /refresh for their account
    (rate-limited in arona_client); they appear after the next sync. Nothing is
    stored locally except the rate-limit bookkeeping (a salted hash)."""
    d = request.get_json(silent=True) or {}
    if d.get("serverRegion") not in SERVER_REGIONS:
        return jsonify({"error": "invalid serverRegion"}), 400
    fc = d.get("friendCode")
    if not isinstance(fc, str) or not fc.strip() or len(fc) > MAX_TEXT:
        return jsonify({"error": "friendCode required"}), 400

    result, status = arona_client.submit_refresh(d["serverRegion"], fc)
    return jsonify(result), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
