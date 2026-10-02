"""Anonymous Eridu Ops feedback API.

Runs as its own lightweight process on port 5003, behind the shared nginx.
The public surface is write-only except for a liveness endpoint:

    GET  /feedback/health
    POST /feedback/submissions
"""

import logging
import re

from flask import Flask, jsonify, request
from flask_cors import CORS

from db import init_db
import repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = Flask(__name__)
CORS(app, origins=["https://eriduops.com", "http://localhost:5173"])
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

init_db()

MAX_MESSAGE_LENGTH = 2000
MAX_PAGE_LENGTH = 200
LOCALES = ("en", "jp", "kr")


def _clean_text(value: str) -> str:
    """Remove terminal control characters while preserving normal whitespace."""
    return "".join(char for char in value if char in "\n\t" or ord(char) >= 32).strip()


@app.get("/feedback/health")
def health():
    return jsonify({"ok": True})


@app.post("/feedback/submissions")
def create_submission():
    """Store an anonymous product-feedback submission."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid JSON body"}), 400

    honeypot = data.get("website", "")
    if not isinstance(honeypot, str):
        return jsonify({"error": "invalid website"}), 400
    if honeypot.strip():
        return jsonify({"ok": True}), 201

    category = data.get("category")
    if category not in repository.CATEGORIES:
        return jsonify({"error": "invalid category"}), 400

    message = data.get("message")
    if not isinstance(message, str):
        return jsonify({"error": "message required"}), 400
    message = _clean_text(message)
    if not message or len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": "message must be 1 to 2000 characters"}), 400

    page = data.get("page")
    if not isinstance(page, str) or len(page) > MAX_PAGE_LENGTH:
        return jsonify({"error": "invalid page"}), 400
    if not re.fullmatch(r"/[A-Za-z0-9/_-]*", page):
        return jsonify({"error": "invalid page"}), 400

    locale = data.get("locale")
    if locale not in LOCALES:
        return jsonify({"error": "invalid locale"}), 400

    submission_id = repository.create_submission(category, message, page, locale)
    logging.getLogger("feedback").info(
        "stored submission id=%s category=%s page=%s locale=%s",
        submission_id,
        category,
        page,
        locale,
    )
    return jsonify({"ok": True}), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
