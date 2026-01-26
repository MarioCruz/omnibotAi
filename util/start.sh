#!/bin/bash
# Start the OmniBot AI Dashboard

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with system site-packages..."
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
    pip install flask flask-cors flask-socketio requests ollama websocket-client python-socketio opencv-python sounddevice
else
    source venv/bin/activate
fi

# Start Ollama if not already running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 3  # Give Ollama time to start
fi

# Check if Ollama is responding
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is running"
else
    echo "Warning: Ollama not responding. LLM features will use rule-based fallback."
fi

# Start the dashboard
echo "Starting OmniBot AI Dashboard on port 8080..."
python3 dashboard.py
