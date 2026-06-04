# Bond 100 Hall

Lightweight SQLite-backed Flask service powering the `/hall` page on the frontend — a community wall of how many players have reached **Bond 100** with each student, on the Global servers.

Runs as its own gunicorn process at `127.0.0.1:5002` (two preloaded workers), behind the same nginx as the inventory parser.

## Bridge model

[arona.icu](https://arona.icu) is the single source of truth. We never store player data per-row — a daily sync hits arona's `rank_by_max_favor_user_info` endpoint **once**, aggregates the five global servers into a wall summary + per-student entry lists, and caches both as JSON blobs in `bond100_meta`. The SQLite file is therefore a **regenerable cache** — a re-sync rebuilds it from scratch.

Submissions ("add me") trigger arona's `/refresh` for the supplied friend code so it shows up in the next sync. The friend code is never stored — only a salted hash for abuse limiting.

Removal is handled on arona's side; the frontend links out to arona's guidelines.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/bond100/summary` | Wall counts per student, plus per-server breakdown and the snapshot date |
| GET | `/bond100/students/<id>/entries` | Player names at bond 100 for one student |
| GET | `/bond100/health` | Liveness check |
| POST | `/bond100/submissions` | "Add me" — triggers an arona `/refresh` for the given `{serverRegion, friendCode}` |

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask entrypoint — 4 routes, CORS, no warmup |
| `sync_arona.py` | Daily sync: pulls arona's user-info endpoint, aggregates global servers, writes the wall blobs to `bond100_meta` |
| `arona_client.py` | `/refresh` client + abuse limiting (per-code 4h cooldown, global hourly cap, friend-code format validation) |
| `repository.py` | Read helpers — `get_summary()` and `get_student_entries(id)` over the cached blobs |
| `db.py` | SQLite connection (WAL mode), schema init, server regions, linked-student pair map |
| `schema.sql` | Two tables: `bond100_meta` (cached wall) and `bond100_refresh_log` (cooldown ledger keyed by `sha256(server|friend_code)`) |
| `gunicorn.conf.py` | `127.0.0.1:5002`, 2 workers, `preload_app=True`, 30s timeout |
| `requirements.txt` | Flask, flask-cors, gunicorn, python-dotenv — no ML deps (kept separate from the parser venv) |
| `data/bond100.sqlite` | Local dev cache (gitignored). Production lives at `/opt/eridu-ops-api/var/bond100.sqlite` via `BOND100_DB_PATH`. |

## Environment

| Variable | Required? | Notes |
|---|---|---|
| `ARONA_TOKEN` | yes for sync + submissions | `'<email>:<token>'` from arona. **Binds to the first IP that calls arona** — set on the VPS only, and never let a laptop make the first call. |
| `BOND100_DB_PATH` | optional | Defaults to `./data/bond100.sqlite`. Production overrides to the regenerable cache under `var/`. |

## Local development

```bash
cd services/bond100
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Initialise the SQLite file from schema.sql (creates ./data/bond100.sqlite)
python db.py

# Local sync without hitting arona — replay a saved JSON dump
python sync_arona.py --from-file userinfo_full_s9.json

# Live sync (requires ARONA_TOKEN — see env table above)
ARONA_TOKEN='<email>:<token>' python sync_arona.py

# Run dev server
python app.py
# API at http://localhost:5002
```

Quick smoke check:

```bash
curl http://localhost:5002/bond100/health
curl http://localhost:5002/bond100/summary | jq '. | {students: (.entries|length), snapshotDate}'
```

## Deployment

Production setup, systemd units, daily sync timer, token binding, and the regenerable-cache backup story all live in [`../../deploy/README.md`](../../deploy/README.md). Key bits:

- `eridu-bond100.service` — the gunicorn process
- `eridu-bond100-sync.{service,timer}` — daily wall refresh
- `var/bond100.sqlite` — the cache (outside the code tree so `git pull` doesn't touch it)
