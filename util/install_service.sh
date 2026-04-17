#!/bin/bash
# Install omniai.service as a systemd unit.
# Run on the Pi: ~/omniai/util/install_service.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="$PROJECT_DIR/util/omniai.service"
UNIT_DST="/etc/systemd/system/omniai.service"

if [ ! -f "$UNIT_SRC" ]; then
    echo "ERROR: $UNIT_SRC not found" >&2
    exit 1
fi

echo "Installing $UNIT_SRC -> $UNIT_DST"
sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo chmod 644 "$UNIT_DST"
sudo systemctl daemon-reload
sudo systemctl enable omniai.service
sudo systemctl restart omniai.service

echo
echo "Service installed. Useful commands:"
echo "  sudo systemctl status omniai"
echo "  sudo systemctl restart omniai"
echo "  journalctl -u omniai -f"
echo "  journalctl -u omniai --since '10 min ago'"
