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
| Nginx site | `/etc/nginx/sites-available/eridu-api` |
| SSL cert | `/etc/letsencrypt/live/<DOMAIN>/` (auto-renews via certbot timer) |

Gunicorn binds to `127.0.0.1:5001` with 3 workers. Nginx terminates SSL on 443 and rate-limits to 5 requests/min per IP.

## Update flow (after pushing new code)

```bash
ssh root@<vps-ip>
cd /opt/eridu-ops-api && git pull
systemctl restart eridu-parser
```

If `requirements.txt` changed:

```bash
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && pip install -r requirements.txt"
systemctl restart eridu-parser
```

If new game items were added (re-fetch icons + rebuild embeddings):

```bash
sudo -u eridu bash -c "cd /opt/eridu-ops-api/services/inventory_parser && source .venv/bin/activate && python download_icons.py && python embed.py items && python embed.py equipment"
systemctl restart eridu-parser
```

## Logs

```bash
journalctl -u eridu-parser -f          # live
journalctl -u eridu-parser -n 100      # last 100 lines
journalctl -u eridu-parser --since "1 hour ago"
```

## Health check

```bash
systemctl status eridu-parser
curl -X POST https://api.eriduops.com/inventory/parse \
  -F image=@screenshot.png -F inventoryType=items
free -h    # confirm RAM headroom
```

## Migration to a different VPS provider

The setup is fully stateless (no DB, no user uploads retained). To migrate:

1. Provision new VPS, get IP
2. Run setup.sh on new VPS
3. Update Cloudflare A record to new IP (TTL 5 min → traffic switches in ~5 min)
4. Confirm new VPS is serving (check logs, hit endpoint)
5. Cancel old VPS subscription
