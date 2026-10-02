"""Read and triage feedback without exposing a public admin endpoint."""

import argparse

from db import init_db
from repository import (
    STATUSES,
    FeedbackSubmission,
    get_submission,
    list_submissions,
    set_status,
)


def _print_submission(submission: FeedbackSubmission) -> None:
    print(
        f"#{submission['id']} [{submission['status']}] {submission['category']} "
        f"{submission['created_at']}"
    )
    print(f"page={submission['page']} locale={submission['locale']}")
    print(submission["message"])
    print()


def _list(args: argparse.Namespace) -> None:
    status = None if args.status == "all" else args.status
    submissions = list_submissions(status=status, limit=args.limit)
    if not submissions:
        print("No matching feedback submissions.")
        return
    for submission in submissions:
        _print_submission(submission)


def _show(args: argparse.Namespace) -> None:
    submission = get_submission(args.id)
    if submission is None:
        raise SystemExit(f"Feedback #{args.id} was not found.")
    _print_submission(submission)


def _status(args: argparse.Namespace) -> None:
    changed = set_status(args.id, args.status)
    if not changed:
        raise SystemExit(f"Feedback #{args.id} was not found.")
    print(f"Feedback #{args.id} marked {args.status}.")


def _bounded_limit(value: str) -> int:
    limit = int(value)
    if not 1 <= limit <= 200:
        raise argparse.ArgumentTypeError("limit must be between 1 and 200")
    return limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Eridu Ops feedback submissions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List recent submissions.")
    list_parser.add_argument("--status", choices=(*STATUSES, "all"), default="new")
    list_parser.add_argument("--limit", type=_bounded_limit, default=20)
    list_parser.set_defaults(handler=_list)

    show_parser = subparsers.add_parser("show", help="Show one submission.")
    show_parser.add_argument("id", type=int)
    show_parser.set_defaults(handler=_show)

    status_parser = subparsers.add_parser("status", help="Change a submission status.")
    status_parser.add_argument("id", type=int)
    status_parser.add_argument("status", choices=STATUSES)
    status_parser.set_defaults(handler=_status)

    args = parser.parse_args()
    init_db()
    args.handler(args)


if __name__ == "__main__":
    main()
