# VPS Cheatsheet

```
ssh root@109.123.250.18
```

Two services behind one nginx (see [README.md](README.md)):

- **eridu-parser** — inventory screenshot parser (gunicorn `127.0.0.1:5001`)
- **eridu-bond100** — Bond 100 Hall API + daily arona sweep (gunicorn `127.0.0.1:5002`)

## Full reinstall / first-time setup

```bash
bash /opt/eridu-ops-api/deploy/setup.sh
```

## Service management

```bash
systemctl status  eridu-parser
systemctl restart eridu-parser                     # after parser code changes
systemctl status  eridu-bond100                    # the wall API (serves /bond100/*)
systemctl restart eridu-bond100                    # after bond100 SERVING code changes
systemctl list-timers eridu-bond100-sweep.timer    # daily sweep schedule (repo default 05:30 Asia/Jakarta)
systemctl status  eridu-bond100-sweep.service      # last sweep run result
```

## Logs

```bash
journalctl -u eridu-parser -f
journalctl -u eridu-bond100 -f                             # wall API live
journalctl -u eridu-bond100-sweep.service -n 20 --no-pager # daily sweep outcome
journalctl -u eridu-parser --since "1 hour ago"
```

## Code update (most common)

```bash
# Parser:
cd /opt/eridu-ops-api && git pull && systemctl restart eridu-parser

# Bond100 SERVING endpoints (app.py etc.):
cd /opt/eridu-ops-api && git pull && systemctl restart eridu-bond100

# Bond100 sweep LOGIC only (sweep_rank.py / rank_client.py / wall_store.py):
cd /opt/eridu-ops-api && git pull            # oneshot; next timer picks it up, no restart

# Bond100 unit file changed (deploy/eridu-bond100-sweep.service):
cd /opt/eridu-ops-api && git pull
sudo cp deploy/eridu-bond100-sweep.service /etc/systemd/system/eridu-bond100-sweep.service
sudo systemctl daemon-reload
```

## Bond 100 sweep (arona /rank)

Token: `ARONA_TOKEN=<email>:<token>` in `/opt/eridu-ops-api/.env` (VPS-only, binds to this IP).

All sweep commands **must run from the bond100 dir** — `.venv/bin/python` is a relative path, so running from `~` gives "Permission denied". `BOND100_DB_PATH` must be set too (it lives in the systemd unit, not `.env`).

```bash
cd /opt/eridu-ops-api/services/bond100

# Wrapper — swap FLAGS below:
sudo -u eridu bash -c 'set -a; source /opt/eridu-ops-api/.env; BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite; set +a; .venv/bin/python sweep_rank.py FLAGS'
```

| FLAGS | What it does |
|---|---|
| `--global --force` | The daily job. Atomic full rebuild + publish (~48 calls). Run to fix / catch up the wall right now. |
| `--report --limit 60` | Read-only store coverage, stalest first. No arona calls. |
| `--diagnose-page N` | Drill poisoned page N (size 5 -> size 1) to pinpoint arona's broken record (~15 calls). Clean = `no chunk 500'd now`. |
| `--tail --dry-run --force` | DEBUG ONLY: preview the retired incremental; writes nothing. |

Why the daily job is `--global --force`, not an incremental `--tail`: arona regenerates **both** the record `key` AND `rankUpdateTime` on every player refresh, so no per-record anchor dedups reliably across time, and an append-only tail can't see removals anyway. `--global` rebuilds from arona's ground truth each run (no drift, ghosts, or dupes); `--force` guarantees it runs even if the shared budget window was spent by manual runs.

Budget usage (read-only):

```bash
sudo -u eridu bash -c 'BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite .venv/bin/python budget.py'
```

## Bond 100 health / print

Served wall total — should equal arona.icu's 百绊 badge (minus 1 if a record is poisoned):

```bash
curl -s https://api.eriduops.com/bond100/summary \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print("total",d.get("total"),"students",len(d.get("students",[])))'
```

Local store audit — dupes (rut/name), missing rut, total (no arona calls):

```bash
sudo -u eridu python3 - <<'PY'
import json, sqlite3
db = sqlite3.connect("/opt/eridu-ops-api/var/bond100.sqlite"); db.row_factory = sqlite3.Row
total = miss = dupe = 0
for r in db.execute("SELECT student_id, entries FROM bond100_student_rank"):
    ents = json.loads(r["entries"]); total += len(ents); seen = {}
    for e in ents:
        if e.get("rut") is None: miss += 1
        seen.setdefault((e.get("serverRegion"), e.get("playerName")), []).append(e.get("rut"))
    dupe += sum(len(v) - 1 for v in seen.values() if len(v) > 1)
print("store total:", total, "| missing rut:", miss, "| dupes:", dupe)
PY
```

Invariant: **served total == arona badge** (minus 1 per poisoned record) = healthy. A `dupes` count above 0 means drift crept in; a `--global --force` clears it.

## Update when new game items drop (parser)

```bash
cd /opt/eridu-ops-api && git pull
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && python download_icons.py && python embed.py items && python embed.py equipment"
systemctl restart eridu-parser
```

## Parser health check

```bash
curl -X POST https://api.eriduops.com/inventory/parse \
  -F "image=@/path/to/screenshot.png" \
  -F "inventoryType=items"
```

## System

```bash
free -h    # RAM headroom
df -h      # disk space
```

## Nginx

```bash
nginx -t                     # test config before reloading
systemctl reload nginx       # reload without downtime
systemctl status nginx
```

## SSL cert (auto-renews)

```bash
certbot renew --dry-run
```

## pip deps update (if requirements.txt changed)

```bash
# Parser:
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && pip install -r requirements.txt"
systemctl restart eridu-parser

# Bond100:
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/bond100 && source .venv/bin/activate && pip install -r requirements.txt"
systemctl restart eridu-bond100
```
