#!/bin/bash
# Start the OmniBot AI Dashboard
# Can be run from anywhere: ~/omniai/util/start.sh

# Navigate to project root (one level up from util/)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==================================="
echo "  OmniBot AI Dashboard"
echo "==================================="
echo "  Project: $PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[Setup] Creating virtual environment with system site-packages..."
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
    pip install flask flask-cors flask-socketio requests ollama websocket-client \
        python-socketio opencv-python sounddevice st7735 gpiodevice
else
    source venv/bin/activate
fi

# Pre-launch smoke test — aborts startup if imports or hardware are broken
# so systemd doesn't flap a half-working dashboard in a restart loop.
echo "[Smoke] Running pre-launch checks..."
if ! python3 util/smoke_test.py; then
    echo "[Smoke] FAILED — not launching dashboard. See output above."
    exit 1
fi

# Start the dashboard (uses rule-based navigation — no LLM needed)
echo "[Navigation] Rule-based (no LLM required)"
echo "[Start] Launching dashboard on port 8080..."
exec python3 dashboard.py "$@"
