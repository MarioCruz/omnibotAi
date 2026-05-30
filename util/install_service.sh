#!/bin/bash
# Install the OmniAI systemd units.
#   ~/omniai/util/install_service.sh                # dashboard + clean-URL redirect
#   ~/omniai/util/install_service.sh --no-redirect  # dashboard only (no 80/443 redirect)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

WITH_REDIRECT=1
for arg in "$@"; do
    case "$arg" in
        --no-redirect) WITH_REDIRECT=0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

install_unit() {
    local name="$1"
    local src="$PROJECT_DIR/util/$name"
    local dst="/etc/systemd/system/$name"
    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found" >&2
        exit 1
    fi
    echo "Installing $src -> $dst"
    sudo cp "$src" "$dst"
    sudo chmod 644 "$dst"
}

# Dashboard service (port 8080).
install_unit omniai.service

# Clean-URL redirect (ports 80/443 -> :8080) unless opted out.
if [ "$WITH_REDIRECT" -eq 1 ]; then
    install_unit omniai-redirect.service
fi

sudo systemctl daemon-reload
sudo systemctl enable omniai.service
sudo systemctl restart omniai.service

if [ "$WITH_REDIRECT" -eq 1 ]; then
    sudo systemctl enable omniai-redirect.service
    sudo systemctl restart omniai-redirect.service
fi

echo
echo "Service(s) installed. Useful commands:"
echo "  sudo systemctl status omniai"
echo "  sudo systemctl restart omniai"
echo "  journalctl -u omniai -f"
echo "  journalctl -u omniai --since '10 min ago'"
if [ "$WITH_REDIRECT" -eq 1 ]; then
    echo
    echo "Clean URL active: https://omniai.local (redirects to :8080)"
    echo "  sudo systemctl status omniai-redirect"
fi
