#!/bin/bash
# Robot Speech Script for Pi (uses PipeWire for Bluetooth audio)

SPKON_FREQ=1422
SPKOFF_FREQ=4650
# 0.3s is the minimum duration the robot's audio relay reliably detects.
# Shortening below this causes speaker_on/off transitions to miss, which
# makes speech play through PipeWire but not route to the robot speaker.
TONE_DURATION=0.3

if [ -z "$1" ]; then
    echo "Usage: ./speak_pi.sh \"Your message here\""
    exit 1
fi

TEXT="$1"

# First send speaker off to reset state
sox -n -t wav - synth $TONE_DURATION sine $SPKOFF_FREQ gain -5 2>/dev/null | pw-play -
sleep 0.3

# Play Speaker On tone via PipeWire
echo "Speaker On..."
sox -n -t wav - synth $TONE_DURATION sine $SPKON_FREQ gain -5 2>/dev/null | pw-play -

sleep 0.2

# Speak using espeak-ng via PipeWire
echo "Speaking: $TEXT"
espeak-ng --stdout -a 200 -- "$TEXT" | pw-play -

sleep 0.3

# Play Speaker Off tone via PipeWire
echo "Speaker Off..."
sox -n -t wav - synth $TONE_DURATION sine $SPKOFF_FREQ gain -5 2>/dev/null | pw-play -

echo "Done!"
