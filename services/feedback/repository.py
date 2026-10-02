"""Persistence helpers for feedback submissions and the review CLI."""

from typing import Literal, TypedDict

from db import get_connection

FeedbackCategory = Literal["suggestion", "bug_report", "incorrect_data", "other"]
FeedbackStatus = Literal["new", "reviewed", "archived"]

CATEGORIES: tuple[FeedbackCategory, ...] = (
    "suggestion",
    "bug_report",
    "incorrect_data",
    "other",
)
STATUSES: tuple[FeedbackStatus, ...] = ("new", "reviewed", "archived")


class FeedbackSubmission(TypedDict):
    id: int
    category: str
    message: str
    page: str
    locale: str
    status: str
    created_at: str


def create_submission(
    category: FeedbackCategory,
    message: str,
    page: str,
    locale: str,
) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO feedback_submissions (category, message, page, locale)
            VALUES (?, ?, ?, ?)
            """,
            (category, message, page, locale),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def list_submissions(
    status: FeedbackStatus | None = "new",
    limit: int = 20,
) -> list[FeedbackSubmission]:
    conn = get_connection()
    try:
        if status is None:
            rows = conn.execute(
                """
                SELECT id, category, message, page, locale, status, created_at
                FROM feedback_submissions
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, category, message, page, locale, status, created_at
                FROM feedback_submissions
                WHERE status = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        return [FeedbackSubmission(**dict(row)) for row in rows]
    finally:
        conn.close()


def get_submission(submission_id: int) -> FeedbackSubmission | None:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, category, message, page, locale, status, created_at
            FROM feedback_submissions
            WHERE id = ?
            """,
            (submission_id,),
        ).fetchone()
        return FeedbackSubmission(**dict(row)) if row else None
    finally:
        conn.close()


def set_status(submission_id: int, status: FeedbackStatus) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "UPDATE feedback_submissions SET status = ? WHERE id = ?",
            (status, submission_id),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()
