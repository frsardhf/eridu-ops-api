# Feedback API

Small, independent Flask service for anonymous Eridu Ops product feedback. It
runs on `127.0.0.1:5003` behind the shared nginx and owns its own durable SQLite
database.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/feedback/health` | Liveness check |
| POST | `/feedback/submissions` | Store anonymous `{category, message, page, locale}` feedback |

There is no public read endpoint. Review submissions with `cli.py` on the VPS.

## Local development

```bash
cd services/feedback
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python db.py
python app.py
```

The database defaults to `data/feedback.sqlite`. Production sets
`FEEDBACK_DB_PATH=/opt/eridu-ops-api/var/feedback.sqlite`.

## Review CLI

```bash
python cli.py list
python cli.py show 12
python cli.py status 12 reviewed
```

Categories are `suggestion`, `bug_report`, `incorrect_data`, and `other`.
Statuses are `new`, `reviewed`, and `archived`. The database stores no contact
field or raw IP address and must be included in backups and VPS migrations.
