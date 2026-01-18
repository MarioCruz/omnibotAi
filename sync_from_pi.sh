#!/bin/bash
# Sync OmniAI project from Raspberry Pi to local backup
# Run periodically to protect against SD card corruption

PI_HOST="admin@omniai.local"  # Change to your Pi's hostname or IP
PI_PATH="/home/admin/omniai/"
LOCAL_PATH="/Users/mariocruz/Documents/GitHub/omnibotAi/"
LOG_FILE="$LOCAL_PATH/sync.log"

# Timestamp
echo "=== Sync started: $(date) ===" >> "$LOG_FILE"

# Check if Pi is reachable
if ! ping -c 1 -W 2 omniai.local &>/dev/null; then
    echo "ERROR: Pi not reachable" >> "$LOG_FILE"
    exit 1
fi

# Sync from Pi (excluding temp files and caches)
rsync -avz --delete \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pem' \
    --exclude='.DS_Store' \
    --exclude='._*' \
    --exclude='models/' \
    --exclude='.claude/' \
    --exclude='*.log' \
    "$PI_HOST:$PI_PATH" "$LOCAL_PATH" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "SUCCESS: Sync completed" >> "$LOG_FILE"

    # Auto-commit if there are changes
    cd "$LOCAL_PATH"
    if [[ -n $(git status --porcelain) ]]; then
        git add -A
        git commit -m "Auto-sync from Pi: $(date '+%Y-%m-%d %H:%M')"
        echo "Git commit created" >> "$LOG_FILE"
    else
        echo "No changes to commit" >> "$LOG_FILE"
    fi
else
    echo "ERROR: rsync failed" >> "$LOG_FILE"
fi

echo "=== Sync ended: $(date) ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
