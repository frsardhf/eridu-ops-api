# Deployment

One-shot installer for the inventory parser API on a fresh Ubuntu 24.04 VPS.

## Quick start

DNS first: add an A record `api.eriduops.com` → VPS IP in Cloudflare (DNS only, grey cloud).

Then on the VPS as root:

```bash
curl -fsSL https://raw.githubusercontent.com/frsardhf/eridu-ops-api/master/deploy/setup.sh | bash
```

Takes ~10 minutes (apt install ~2 min, pip install ~3 min, embed.py ×2 ~5 min).

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
| Systemd units | `eridu-parser.service`, `eridu-bond100.service`, `eridu-bond100-sync.{service,timer}` |
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
sudo -u eridu bash -c 'set -a; source /opt/eridu-ops-api/.env; set +a; .venv/bin/python sync_arona.py --dry-run'
```

To pause ingestion entirely (e.g. arona warns their data is dirty until a
cleanup lands): `systemctl disable --now eridu-bond100-sync.timer`; re-enable
later with `systemctl enable --now eridu-bond100-sync.timer`.

### Submissions ("add me")

`POST /bond100/submissions {serverRegion, friendCode}` triggers an arona
`/refresh` for that account, so it appears in the next sync. It's rate-limited in
three layers — nginx edge, a per-code 6h cooldown, and a global daily cap (45/day, under arona's ~60 req/day token budget shared with the sync) — and
the friend code is never stored (only a salted hash for the cooldown). Removal is
handled on arona's side; the frontend links out to arona's guidelines.

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

If new game items were added (re-fetch icons + rebuild embeddings):

```bash
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && python download_icons.py && python embed.py items && python embed.py equipment"
systemctl restart eridu-parser
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
