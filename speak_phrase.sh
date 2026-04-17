#!/bin/bash
# Play a pre-recorded phrase WAV file with speaker on/off tones
# Usage: ./speak_phrase.sh <phrase_name>
# Phrases: hello, yes, no, thanks, omnibot

SPKON_FREQ=1422
SPKOFF_FREQ=4650
# 0.3s is the minimum the robot's audio relay reliably detects — don't
# shorten this or speaker_on/off transitions get missed.
TONE_DURATION=0.3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHRASE_DIR="$SCRIPT_DIR/audio_phrases"

if [ -z "$1" ]; then
    echo "Usage: ./speak_phrase.sh <phrase_name>"
    echo "Available: hello, yes, no, thanks, omnibot"
    exit 1
fi

PHRASE="$1"
WAV_FILE="$PHRASE_DIR/${PHRASE}.wav"

if [ ! -f "$WAV_FILE" ]; then
    echo "Phrase not found: $WAV_FILE"
    echo "Available phrases:"
    ls "$PHRASE_DIR"/*.wav 2>/dev/null | xargs -I{} basename {} .wav
    # Fall back to text-to-speech
    echo "Falling back to TTS..."
    exec "$SCRIPT_DIR/speak_pi.sh" "$PHRASE"
fi

# Play Speaker On tone
sox -n -t wav - synth $TONE_DURATION sine $SPKON_FREQ gain -5 2>/dev/null | pw-play -
sleep 0.2

# Play the WAV file
echo "Playing: $PHRASE"
pw-play "$WAV_FILE"

sleep 0.3

# Play Speaker Off tone
sox -n -t wav - synth $TONE_DURATION sine $SPKOFF_FREQ gain -5 2>/dev/null | pw-play -
