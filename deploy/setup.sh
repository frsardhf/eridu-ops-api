#!/usr/bin/env bash
# One-shot installer for the inventory parser API.
# Run on a fresh Ubuntu 24.04 VPS as root:
#
#   curl -fsSL https://raw.githubusercontent.com/frsardhf/eridu-ops-api/master/deploy/setup.sh | bash
#
# Override defaults via env vars:
#   DOMAIN=api.example.com EMAIL=you@example.com bash setup.sh
#
# Idempotent — safe to re-run after edits.

set -euo pipefail

DOMAIN="${DOMAIN:-api.eriduops.com}"
EMAIL="${EMAIL:-frsardhafa@gmail.com}"
APP_DIR="/opt/eridu-ops-api"
SVC_DIR="$APP_DIR/services/inventory_parser"
SVC_BOND_DIR="$APP_DIR/services/bond100"
VAR_DIR="$APP_DIR/var"   # stateful data (bond100 SQLite) — outside the code tree
REPO_URL="https://github.com/frsardhf/eridu-ops-api.git"
USER_NAME="eridu"

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: run as root (sudo bash setup.sh)" >&2
  exit 1
fi

echo "==> System packages"
apt update
apt install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx git ufw curl libgl1

echo "==> Service user"
id "$USER_NAME" &>/dev/null || useradd -m -s /bin/bash "$USER_NAME"

echo "==> Clone / update repo"
git config --global --add safe.directory "$APP_DIR" 2>/dev/null || true
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull
else
  git clone "$REPO_URL" "$APP_DIR"
fi
chown -R "$USER_NAME:$USER_NAME" "$APP_DIR"

echo "==> Python venv + deps (inventory parser)"
sudo -u "$USER_NAME" bash -c "
  set -e
  cd '$SVC_DIR'
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
"

echo "==> Python venv + deps (bond100)"
sudo -u "$USER_NAME" bash -c "
  set -e
  cd '$SVC_BOND_DIR'
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
"

echo "==> Bond100 cache dir (persists across git pull)"
install -d -o "$USER_NAME" -g "$USER_NAME" "$VAR_DIR"
# Initialize the cache tables. The wall is populated by sync_arona.py once
# ARONA_TOKEN is set in /opt/eridu-ops-api/.env — see deploy/README.md.
sudo -u "$USER_NAME" bash -c "
  set -e
  cd '$SVC_BOND_DIR'
  BOND100_DB_PATH='$VAR_DIR/bond100.sqlite' .venv/bin/python db.py
"

echo "==> Download icons + build icon index (fetches from schaledb.com)"
sudo -u "$USER_NAME" bash -c "
  set -e
  cd '$SVC_DIR'
  source .venv/bin/activate
  python download_icons.py
"

echo "==> Nginx site"
install -m 644 "$APP_DIR/deploy/eridu-api.nginx.conf" /etc/nginx/sites-available/eridu-api
sed -i "s/__DOMAIN__/$DOMAIN/g" /etc/nginx/sites-available/eridu-api
ln -sf /etc/nginx/sites-available/eridu-api /etc/nginx/sites-enabled/eridu-api
# Disable default site if it exists, to avoid conflicts on port 80
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

echo "==> Systemd services"
install -m 644 "$APP_DIR/deploy/eridu-parser.service" /etc/systemd/system/eridu-parser.service
install -m 644 "$APP_DIR/deploy/eridu-bond100.service" /etc/systemd/system/eridu-bond100.service
# _info baseline sync (one arona call for all students; frozen/delayed cache).
install -m 644 "$APP_DIR/deploy/eridu-bond100-sync.service" /etc/systemd/system/eridu-bond100-sync.service
install -m 644 "$APP_DIR/deploy/eridu-bond100-sync.timer" /etc/systemd/system/eridu-bond100-sync.timer
# Rolling /rank sweep (live per-student counts) — the wall's primary source.
install -m 644 "$APP_DIR/deploy/eridu-bond100-sweep.service" /etc/systemd/system/eridu-bond100-sweep.service
install -m 644 "$APP_DIR/deploy/eridu-bond100-sweep.timer" /etc/systemd/system/eridu-bond100-sweep.timer
systemctl daemon-reload
systemctl enable --now eridu-parser
systemctl enable --now eridu-bond100
# The sweep owns the wall; enable its timer. The _info sync is the superseded
# baseline (frozen upstream), so its timer stays OFF — re-enable only if arona
# fixes _info. Both unit files are installed either way.
systemctl enable --now eridu-bond100-sweep.timer

# NOTE: the sync + submissions need ARONA_TOKEN ('<email>:<token>') in
# /opt/eridu-ops-api/.env. The token binds to the first IP that calls arona, so
# the first run MUST happen here on the VPS (its static IP), never from a laptop:
#   echo 'ARONA_TOKEN=<email>:<token>' >> /opt/eridu-ops-api/.env
#   # bootstrap the table from _info once, then publish + let the sweep roll:
#   sudo -u eridu BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite python3 \
#     services/bond100/sync_arona.py        # seeds baseline rows + binds the IP
#   sudo -u eridu BOND100_DB_PATH=/opt/eridu-ops-api/var/bond100.sqlite python3 \
#     services/bond100/wall_store.py --commit   # publish the served wall

echo "==> Firewall"
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

echo "==> SSL (certbot)"
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect

echo "==> Done. Service status:"
systemctl status eridu-parser --no-pager -l | head -20

echo
echo "Test from your local machine:"
echo "  curl -X POST https://$DOMAIN/inventory/parse \\"
echo "    -F image=@screenshot.png -F inventoryType=items"
