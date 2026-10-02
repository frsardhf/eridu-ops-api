"""Integration tests for anonymous feedback submission and persistence."""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

_TEMP_DIR = tempfile.TemporaryDirectory()
os.environ["FEEDBACK_DB_PATH"] = str(Path(_TEMP_DIR.name) / "feedback.sqlite")

from app import app  # noqa: E402


class FeedbackSubmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        with sqlite3.connect(os.environ["FEEDBACK_DB_PATH"]) as conn:
            conn.execute("DELETE FROM feedback_submissions")

    def _count(self) -> int:
        with sqlite3.connect(os.environ["FEEDBACK_DB_PATH"]) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM feedback_submissions").fetchone()[0])

    def test_health(self) -> None:
        response = self.client.get("/feedback/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})

    def test_valid_submission_is_stored_without_contact_data(self) -> None:
        response = self.client.post(
            "/feedback/submissions",
            json={
                "category": "suggestion",
                "message": "  Add a planner shortcut.  ",
                "page": "/students",
                "locale": "en",
                "website": "",
            },
        )

        self.assertEqual(response.status_code, 201)
        with sqlite3.connect(os.environ["FEEDBACK_DB_PATH"]) as conn:
            row = conn.execute(
                "SELECT category, message, page, locale, status FROM feedback_submissions"
            ).fetchone()
        self.assertEqual(row, ("suggestion", "Add a planner shortcut.", "/students", "en", "new"))

    def test_honeypot_returns_success_without_storing(self) -> None:
        response = self.client.post(
            "/feedback/submissions",
            json={
                "category": "other",
                "message": "Spam",
                "page": "/",
                "locale": "en",
                "website": "https://spam.example",
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(self._count(), 0)

    def test_rejects_invalid_fields(self) -> None:
        cases = (
            {"category": "unknown", "message": "Hello", "page": "/", "locale": "en"},
            {"category": "other", "message": "", "page": "/", "locale": "en"},
            {"category": "other", "message": "Hello", "page": "home", "locale": "en"},
            {"category": "other", "message": "Hello", "page": "/", "locale": "id"},
        )

        for payload in cases:
            with self.subTest(payload=payload):
                response = self.client.post("/feedback/submissions", json=payload)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self._count(), 0)

    def test_rejects_oversized_message(self) -> None:
        response = self.client.post(
            "/feedback/submissions",
            json={
                "category": "bug_report",
                "message": "x" * 2001,
                "page": "/hall",
                "locale": "en",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self._count(), 0)


if __name__ == "__main__":
    unittest.main()
