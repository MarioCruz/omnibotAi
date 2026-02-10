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

# Load .env if present
if [ -f ".env" ]; then
    echo "[Config] Found .env file"
fi

# Check Groq API key
if [ -n "$GROQ_API_KEY" ] || grep -q "GROQ_API_KEY" .env 2>/dev/null; then
    echo "[LLM] Groq cloud API configured"
else
    echo "[LLM] Warning: No GROQ_API_KEY found. Set it in .env for cloud LLM."
    # Check Ollama as fallback
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[LLM] Ollama available as fallback"
    else
        echo "[LLM] No LLM available - AI commands will use rule-based fallback"
    fi
fi

# Start the dashboard
echo "[Start] Launching dashboard on port 8080..."
python3 dashboard.py "$@"
