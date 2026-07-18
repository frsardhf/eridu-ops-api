# Bond 100 Hall

Lightweight SQLite-backed Flask service powering the `/hall` page on the frontend — a community wall of how many players have reached **Bond 100** with each student, on the Global servers.

Runs as its own gunicorn process at `127.0.0.1:5002` (two preloaded workers), behind the same nginx as the inventory parser.

## Bridge model

[arona.icu](https://arona.icu) is the single source of truth. We never store player data per-row. The scheduled daily job is a **rolling `/rank` sweep** (`sweep_rank.py --global`): it pages arona's `friends/rank` global ranking (server 4, no `studentId`, sorted bond-descending), aggregates it per student into the `bond100_student_rank` table, and **atomically** reassembles + publishes the wall (`wall_store.py`). The SQLite file is therefore a **regenerable cache**: a re-sweep rebuilds it from arona's ground truth every run, so there is no accumulated drift.

An older one-call `_info` baseline (`sync_arona.py`, arona's `rank_by_max_favor_user_info`) seeded all students at once. It is now retired to a manual re-seed (its timer is off) and writes through the same per-student table without clobbering a fresher `/rank` row.

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
| `app.py` | Flask entrypoint: 4 routes, CORS, no warmup |
| `sweep_rank.py` | **The scheduled daily job** (`--global --force`): rolling `/rank` sweep that pages arona's global bond-100 ranking, aggregates per student, and atomically rebuilds + publishes the wall via `wall_store.py`. Poisoned-page recovery (re-fetch at size 5, drill the bad chunk at size 1) so a partial fetch never regresses the wall. Also `--student` / `--report` / `--diagnose-page`, and a retired `--tail` kept debug-only |
| `wall_store.py` | Per-student store + wall assembly: writes one student's row (`source` `rank` or `info`), reassembles the summary + entries blob, and `--commit` publishes it. Preserves per-student `fetched_at` when that student's roster is unchanged, so freshness reflects real roster changes, not the daily rebuild |
| `rank_client.py` | arona `friends/rank` client: fetches one student's live bond-100 entries (the per-student sweep path) |
| `sync_arona.py` | Retired `_info` baseline re-seed (manual/diagnostic; its timer is off). Pulls arona's `rank_by_max_favor_user_info` in one call and writes through the per-student table as `source='info'`, never clobbering a fresher `/rank` row; `--dry-run` previews without writing |
| `budget.py` | Shared arona call budget: one ledger (`bond100_api_log`) over a 20h window caps total calls across `/refresh`, the sweep, and the `_info` sync, holding `REFRESH_RESERVE` back for user submissions |
| `arona_client.py` | `/refresh` client + abuse limiting (per-code 6h cooldown, longer than arona's 4h refresh cache, plus friend-code format validation). The global daily cap now lives in the shared budget ledger (`budget.py`) |
| `repository.py` | Read helpers: `get_summary()` and `get_student_entries(id)` over the published wall |
| `db.py` | SQLite connection (WAL mode), schema init, server regions, linked-student pair map |
| `schema.sql` | Four tables: `bond100_meta` (published wall blob), `bond100_student_rank` (per-student rows the wall is assembled from), `bond100_refresh_log` (per-code cooldown ledger keyed by `sha256(server|friend_code)`), `bond100_api_log` (shared call-budget ledger) |
| `gunicorn.conf.py` | `127.0.0.1:5002`, 2 workers, `preload_app=True`, 30s timeout |
| `requirements.txt` | Flask, flask-cors, gunicorn, python-dotenv (no ML deps, kept separate from the parser venv) |
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

# _info baseline re-seed without hitting arona (replay a saved JSON dump)
python sync_arona.py --from-file userinfo_full_s9.json

# The daily job: rolling /rank sweep (requires ARONA_TOKEN, see env table above).
# --force bypasses our local 20h budget gate for an off-window manual run.
ARONA_TOKEN='<email>:<token>' python sweep_rank.py --global --force

# Read-only report of the current wall (no arona calls)
python sweep_rank.py --report --limit 60

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

Production setup, systemd units, the daily sweep timer, token binding, and the regenerable-cache backup story all live in [`../../deploy/README.md`](../../deploy/README.md). Key bits:

- `eridu-bond100.service`: the gunicorn process
- `eridu-bond100-sweep.{service,timer}`: the daily job, `sweep_rank.py --global --force`
- `eridu-bond100-sync.{service,timer}`: the retired `_info` baseline (timer disabled; re-enable only if arona's `_info` cache un-freezes)
- `var/bond100.sqlite`: the cache (outside the code tree so `git pull` doesn't touch it)
