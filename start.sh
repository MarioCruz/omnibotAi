#!/bin/bash
# Start the HTTPS camera streaming server

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if certificates exist
if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    echo "SSL certificates not found. Generating..."
    ./generate_certs.sh
fi

# Start the server
echo "Starting camera stream server..."
python3 app.py
