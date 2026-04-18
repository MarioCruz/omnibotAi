#!/bin/bash
# Generate pre-recorded phrase WAVs on the Pi using espeak-ng.
# Run this once on the Pi, then commit the generated WAVs back to the repo.
#
# Voice params (-a 200) match speak_pi.sh so generated phrases sound identical
# to the live TTS path. Existing WAVs are not overwritten — delete the file
# first if you want to regenerate it.
#
# Usage: ~/omniai/util/generate_phrases.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PHRASE_DIR="$SCRIPT_DIR/audio_phrases"
mkdir -p "$PHRASE_DIR"

# name|spoken text
PHRASES=(
    "ready|Ready to go"
    "goodbye|Goodbye"
    "found_it|I found it"
    "oops|Oops"
    "sorry|Sorry"
    "okay|Okay"
)

for entry in "${PHRASES[@]}"; do
    name="${entry%%|*}"
    text="${entry#*|}"
    out="$PHRASE_DIR/$name.wav"
    if [ -f "$out" ]; then
        echo "skip  $name.wav (exists)"
        continue
    fi
    echo "gen   $name.wav  <- \"$text\""
    espeak-ng -a 200 -w "$out" -- "$text"
done

echo
echo "Done. Commit new WAVs with:"
echo "  git add audio_phrases/*.wav && git commit -m 'Add phrase WAVs'"
