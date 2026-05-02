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

echo "==> Python venv + deps"
sudo -u "$USER_NAME" bash -c "
  set -e
  cd '$SVC_DIR'
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
"

echo "==> Build CLIP embeddings (downloads ~600 MB on first run, ~5 min each)"
sudo -u "$USER_NAME" bash -c "
  set -e
  cd '$SVC_DIR'
  source .venv/bin/activate
  python embed.py items
  python embed.py equipment
"

echo "==> Nginx site"
install -m 644 "$APP_DIR/deploy/eridu-api.nginx.conf" /etc/nginx/sites-available/eridu-api
sed -i "s/__DOMAIN__/$DOMAIN/g" /etc/nginx/sites-available/eridu-api
ln -sf /etc/nginx/sites-available/eridu-api /etc/nginx/sites-enabled/eridu-api
# Disable default site if it exists, to avoid conflicts on port 80
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

echo "==> Systemd service"
install -m 644 "$APP_DIR/deploy/eridu-parser.service" /etc/systemd/system/eridu-parser.service
systemctl daemon-reload
systemctl enable --now eridu-parser

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
