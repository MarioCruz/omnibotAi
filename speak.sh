#!/bin/bash

# Robot Speech Script for Tomy Omnibot
# Plays Speaker On tone, speaks text, then plays Speaker Off tone

# Frequencies
SPKON_FREQ=1422
SPKOFF_FREQ=4650
TONE_DURATION=0.3

# Check if sox is installed
if ! command -v play &> /dev/null; then
    echo "Error: sox is not installed. Install with: brew install sox"
    exit 1
fi

# Check if text was provided
if [ -z "$1" ]; then
    echo "Usage: ./speak.sh \"Your message here\""
    echo "Example: ./speak.sh \"Hello, I am Tomy Omnibot\""
    exit 1
fi

TEXT="$1"

# Play Speaker On tone
echo "Speaker On..."
play -n synth $TONE_DURATION sine $SPKON_FREQ gain -10 2>/dev/null

# Small pause
sleep 0.2

# Speak the text using macOS say command with a robotic voice
echo "Speaking: $TEXT"
say -v Zarvox "$TEXT" 2>/dev/null || say -v Fred "$TEXT" 2>/dev/null || say "$TEXT"

# Small pause
sleep 0.3

# Play Speaker Off tone
echo "Speaker Off..."
play -n synth $TONE_DURATION sine $SPKOFF_FREQ gain -10 2>/dev/null

echo "Done!"
