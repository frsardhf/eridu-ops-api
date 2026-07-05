# Deployment

One-shot installer for the inventory parser API on a fresh Ubuntu 24.04 VPS.

## Quick start

DNS first: add an A record `api.eriduops.com` → VPS IP in Cloudflare (DNS only, grey cloud).

Then on the VPS as root:

```bash
curl -fsSL https://raw.githubusercontent.com/frsardhf/eridu-ops-api/master/deploy/setup.sh | bash
```

Takes ~5 minutes (apt install ~2 min, pip install ~2 min, icon fetch ~1 min).

## Environment overrides

```bash
DOMAIN=api.example.com EMAIL=you@example.com bash setup.sh
```

## What gets installed

| Component | Location |
|---|---|
| Code | `/opt/eridu-ops-api` (owned by `eridu` user) |
| Python venv | `/opt/eridu-ops-api/services/inventory_parser/.venv` |
| Systemd unit | `/etc/systemd/system/eridu-parser.service` |
| Bond100 venv | `/opt/eridu-ops-api/services/bond100/.venv` |
| Bond100 cache | `/opt/eridu-ops-api/var/bond100.sqlite` (regenerable cache — survives `git pull`) |
| Systemd units | `eridu-parser.service`, `eridu-bond100.service`, `eridu-bond100-sync.{service,timer}`, `eridu-bond100-sweep.{service,timer}` |
| Nginx site | `/etc/nginx/sites-available/eridu-api` |
| SSL cert | `/etc/letsencrypt/live/<DOMAIN>/` (auto-renews via certbot timer) |

Two independent services behind one nginx:
- **Parser** — gunicorn `127.0.0.1:5001`, 3 workers, rate-limited 5 req/min/IP on `/inventory/parse`.
- **Bond100** — gunicorn `127.0.0.1:5002`, 2 workers. Reads (`/bond100/summary`, `/bond100/students/<id>/entries`) at 2 req/s/IP; the one write (`/bond100/submissions`, which triggers a rate-limited arona `/refresh`) at 10 req/min/IP. A daily `eridu-bond100-sync` timer refreshes the cached wall. No admin/moderation queue (bridge model).

## Bond 100 Hall (bridge model)

arona.icu is the single source of truth. The service makes one daily call to
arona's `rank_by_max_favor_user_info` endpoint, aggregates the five global
servers into a wall summary + per-student entries, and caches both as JSON blobs
in `bond100_meta`. The SQLite file is therefore a **regenerable cache**, not
primary data — a re-sync rebuilds it from scratch.

### Set the API token (required for sync + submissions)

The token is `'<email>:<token>'` from arona. It binds to the **first IP that
calls arona**, so it must live on the VPS only and the first call must originate
here — never from a laptop:

```bash
echo 'ARONA_TOKEN=<email>:<token>' >> /opt/eridu-ops-api/.env
```

### Seed + schedule the wall

```bash
# First run seeds the wall AND binds the token to this VPS's static IP.
systemctl start eridu-bond100-sync.service
journalctl -u eridu-bond100-sync.service -n 20      # expect: synced: total=… students=…
# The timer (installed by setup.sh) then refreshes daily.
systemctl list-timers eridu-bond100-sync.timer
```

Local testing never needs the token: `python sync_arona.py --from-file <dump>`
replays a saved response with no network call.

### If the wall looks frozen

The sync only rewrites the cache (and `snapshot_date`) when arona's wall
actually changed; an identical payload logs `unchanged: ...; snapshot stays
<date>` instead, so the frontend's snapshot date tracks real changes. arona's
upstream ranking cache has frozen for a week+ before (2026-06 incident: a
crash in their cache processor), so a stuck wall usually means arona is stuck,
not our timer. To inspect what a sync would do without writing (costs one
arona request):

```bash
cd /opt/eridu-ops-api/services/bond100
sudo -u eridu bash -c 'set -a; source /opt/eridu-ops-api/.env; BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite; set +a; .venv/bin/python sync_arona.py --dry-run'
```

`BOND100_DB_PATH` is required: it lives in the systemd unit, not `.env`, so a
plain shell run falls back to the dev `data/bond100.sqlite` and compares against
the wrong (usually empty) cache, which makes every dry-run report "wall changed"
no matter what arona returns. The sync now prints `cache db: <path>` so you can
confirm it's pointed at `var/bond100.sqlite`.

To pause ingestion entirely (e.g. arona warns their data is dirty until a
cleanup lands): `systemctl disable --now eridu-bond100-sync.timer`; re-enable
later with `systemctl enable --now eridu-bond100-sync.timer`.

### Daily /rank refresh (whole wall in one stream)

`_info` is one call for every student but a delayed cache (it froze for 3+ weeks
in 2026-06). `friends/rank` is live; queried **without** a `studentId` (server=4)
it returns the whole global ranking sorted bond-descending, so the daily job
(`sweep_rank.py --global`) pages the top bond-100 block, aggregates it per student,
and **atomically** replaces + publishes the wall. ~`ceil(bond100/50)` pages (~33 at
1600 bond-100) plus recovery, one shot.

The recovery: arona's `/rank` 500s ("服务器内部错误") on pages holding certain broken
student records (the same ids that 500 on per-student queries, e.g. Rio Armed
10133); one bad record poisons its whole 50-record page. The fetch re-fetches a
poisoned page at a finer size to salvage the innocents, losing only the few in the
broken record's chunk. It publishes only when everything is accounted for
(`fetched + lost == extension`), so a cut/partial fetch never regresses the wall.

```bash
cd /opt/eridu-ops-api/services/bond100
# the scheduled job, run manually:
sudo -u eridu bash -c 'set -a; source /opt/eridu-ops-api/.env; BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite; set +a; .venv/bin/python sweep_rank.py --global'
# --force bypasses OUR budget gate (not arona's) for an off-window manual run.
# read-only coverage report: students split by with-entries / no-entries + a
# fetched_at distribution. No arona calls. --limit caps rows shown per group.
sudo -u eridu bash -c 'BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite python3 sweep_rank.py --report --limit 60'
# debug one student via the per-student endpoint (bypasses the global stream):
sudo -u eridu bash -c 'set -a; source /opt/eridu-ops-api/.env; BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite; set +a; .venv/bin/python sweep_rank.py --student 10135'
```

The older per-student roster sweep (`sweep_rank.py` with no `--global`, plus
`--limit`/`--student`/`--dry-run`) is kept for debugging but is no longer the
scheduled job.

#### Cutover (one-time): make the rolling sweep the wall's source

The wall used to come straight from the `_info` sync; the sweep now owns it. To
flip a VPS that's still serving the old `_info` blob:

```bash
cd /opt/eridu-ops-api/services/bond100
# 1) publish the current table (208 seeded + any swept rows) to the served wall
sudo -u eridu bash -c 'BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite python3 wall_store.py --commit'
# 2) start the daily rolling refresh
systemctl enable --now eridu-bond100-sweep.timer
# 3) retire the _info sync timer. _info is frozen + superseded by the sweep, and
#    re-stamping its stale data daily would misrepresent the wall's freshness.
systemctl disable --now eridu-bond100-sync.timer
```

After this the wall is published by the sweep's assembly. `sync_arona.py` now
also writes through the per-student table (source='info') and reassembles, never
clobbering a fresher `/rank` row, so it's safe to run manually as a re-seed /
diagnostic tool. Its timer stays off unless arona's `_info` cache un-freezes (a
fast one-call baseline for all students); re-enable it then.

### arona call budget (shared)

Every path that hits arona, `/refresh` submissions, the `/rank` sweep, and the
`_info` sync, shares arona's ~60/day token via one ledger (`bond100_api_log`).
`budget.py` holds a `CEILING` of 55 (safety margin) and reserves 15 calls for
user submissions, so the sweep self-limits to 40 and can never starve "add me".
Check current usage:

```bash
sudo -u eridu bash -c 'BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite python3 budget.py'
```

### Submissions ("add me")

`POST /bond100/submissions {serverRegion, friendCode}` triggers an arona
`/refresh` for that account, so it appears in the next sync. It's rate-limited in
three layers — nginx edge, a per-code 6h cooldown, and the shared daily budget
above (`/refresh` may use the full 55 ceiling; the ledger counts every call kind)
— and the friend code is never stored (only a salted hash for the cooldown).
Removal is handled on arona's side; the frontend links out to arona's guidelines.

### Backup

The cache is regenerable, so backups are optional — `systemctl start
eridu-bond100-sync.service` rebuilds it from arona. To snapshot anyway (WAL-safe):

```bash
sqlite3 /opt/eridu-ops-api/var/bond100.sqlite ".backup '/root/bond100-$(date +%F).sqlite'"
```

## Update flow (after pushing new code)

```bash
ssh root@<vps-ip>
cd /opt/eridu-ops-api && git pull
chown -R eridu:eridu /opt/eridu-ops-api          # git pull as root leaves new files root-owned
systemctl restart eridu-parser eridu-bond100     # bond100 cache in var/ is untouched
```

The `chown` matters: a root-run `git pull` makes new files root-owned, which
breaks the `eridu`-user services and venv creation (`Permission denied` on
`.venv`). Always re-chown after pulling.

**If you re-install the nginx conf**, note `deploy/eridu-api.nginx.conf` is a bare
`listen 80;` template — overwriting the live file wipes certbot's SSL (443) block,
so HTTPS goes down until you re-run certbot to re-inject it:

```bash
install -m 644 deploy/eridu-api.nginx.conf /etc/nginx/sites-available/eridu-api
sed -i 's/__DOMAIN__/api.eriduops.com/g' /etc/nginx/sites-available/eridu-api
certbot --nginx -d api.eriduops.com --non-interactive --agree-tos -m frsardhafa@gmail.com --redirect
```

(Re-running `setup.sh` avoids this — it runs certbot right after installing the conf.)

If the parser's `requirements.txt` changed:

```bash
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && pip install -r requirements.txt"
systemctl restart eridu-parser
```

If the bond100 service's `requirements.txt` changed (separate venv):

```bash
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/bond100 && source .venv/bin/activate && pip install -r requirements.txt"
systemctl restart eridu-bond100
```

If new game items were added (re-fetch icon sprites — the matcher reads them directly, no rebuild step):

```bash
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && python download_icons.py"
systemctl restart eridu-parser
```

One-time after the CLIP→template-matching migration, reclaim ~2 GB on the VPS (torch/transformers are no longer in `requirements.txt` but stay installed until removed):

```bash
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && pip uninstall -y torch transformers tokenizers safetensors huggingface-hub && rm -f cache/icon_embeddings_*"
sudo -u eridu rm -rf ~eridu/.cache/huggingface
```

## Logs

```bash
journalctl -u eridu-parser -f          # live (parser)
journalctl -u eridu-bond100 -f         # live (bond100)
journalctl -u eridu-parser -n 100      # last 100 lines
```

## Health check

```bash
systemctl status eridu-parser
curl -X POST https://api.eriduops.com/inventory/parse \
  -F image=@screenshot.png -F inventoryType=items
free -h    # confirm RAM headroom
```

## Migration to a different VPS provider

Both services are now effectively stateless — the parser holds no data and the
**bond100 DB is a regenerable cache** — so there's nothing to carry over. Just
re-sync on the new box.

1. Provision new VPS, get IP
2. Run setup.sh on new VPS
3. Add `ARONA_TOKEN` to `/opt/eridu-ops-api/.env`, then seed:
   `systemctl start eridu-bond100-sync.service` (the first call binds the token
   to the new IP). If the token is locked to the old IP, ask arona to rebind it.
4. Update Cloudflare A record to new IP (TTL 5 min → traffic switches in ~5 min)
5. Confirm new VPS is serving (check logs, hit endpoints)
6. Cancel old VPS subscription
