#!/usr/bin/env python3
"""
Audio Commander for Tomy Omnibot
Generates audio frequency tones to control the robot
"""

import numpy as np
import time

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[AudioCommander] sounddevice not available - install with: pip install sounddevice")


class AudioCommander:
    """
    Generates audio tones to control Tomy Omnibot.

    The Omnibot uses specific frequencies for each command:
    - Forward: 1614 Hz
    - Backward: 2013 Hz
    - Left: 2208 Hz
    - Right: 1811 Hz
    - Speaker On: 1422 Hz
    - Speaker Off: 4650 Hz
    """

    # Omnibot control frequencies (Hz)
    FREQUENCIES = {
        'forward': 1614,
        'backward': 2013,
        'left': 2208,
        'right': 1811,
        'speaker_on': 1422,
        'speaker_off': 4650,
    }

    def __init__(self, volume: float = 0.5, sample_rate: int = 44100):
        """
        Initialize the audio commander.

        Args:
            volume: Output volume (0.0 to 1.0)
            sample_rate: Audio sample rate in Hz
        """
        self.volume = max(0.0, min(1.0, volume))
        self.sample_rate = sample_rate
        self.is_playing = False

        if not SOUNDDEVICE_AVAILABLE:
            print("[AudioCommander] Warning: sounddevice not available, audio disabled")

    def _generate_tone(self, frequency: float, duration_ms: int) -> np.ndarray:
        """Generate a sine wave tone."""
        duration_s = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(self.sample_rate * duration_s), False)
        tone = np.sin(2 * np.pi * frequency * t) * self.volume
        return tone.astype(np.float32)

    def _play_tone(self, frequency: float, duration_ms: int) -> bool:
        """Play a tone at the specified frequency and duration. Returns True if successful."""
        if not SOUNDDEVICE_AVAILABLE:
            print(f"[AudioCommander] Would play {frequency}Hz for {duration_ms}ms (sounddevice not available)")
            return False

        try:
            self.is_playing = True
            tone = self._generate_tone(frequency, duration_ms)
            sd.play(tone, self.sample_rate)
            # Wait for playback with timeout protection
            # sd.wait() doesn't accept timeout, so we poll with a deadline
            timeout_s = (duration_ms / 1000.0) + 1.0
            deadline = time.time() + timeout_s
            while sd.get_stream() and sd.get_stream().active:
                if time.time() > deadline:
                    print("[AudioCommander] Playback timeout, stopping")
                    sd.stop()
                    break
                time.sleep(0.01)
            self.is_playing = False
            return True
        except Exception as e:
            print(f"[AudioCommander] Error playing tone: {e}")
            self.is_playing = False
            return False

    def forward(self, duration_ms: int = 500):
        """Move forward."""
        self._play_tone(self.FREQUENCIES['forward'], duration_ms)

    def backward(self, duration_ms: int = 500):
        """Move backward."""
        self._play_tone(self.FREQUENCIES['backward'], duration_ms)

    def left(self, duration_ms: int = 500):
        """Turn left."""
        self._play_tone(self.FREQUENCIES['left'], duration_ms)

    def right(self, duration_ms: int = 500):
        """Turn right."""
        self._play_tone(self.FREQUENCIES['right'], duration_ms)

    def speaker_on(self, duration_ms: int = 200):
        """Turn speaker on."""
        self._play_tone(self.FREQUENCIES['speaker_on'], duration_ms)

    def speaker_off(self, duration_ms: int = 200):
        """Turn speaker off."""
        self._play_tone(self.FREQUENCIES['speaker_off'], duration_ms)

    def stop(self):
        """Stop any playing audio."""
        if SOUNDDEVICE_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass
        self.is_playing = False

    def speak(self, text: str):
        """
        Speak text using espeak (with speaker on/off tones).

        Args:
            text: Text to speak (sanitized to alphanumeric, spaces, and basic punctuation)
        """
        import subprocess
        import re

        # Sanitize text to prevent command injection
        # Only allow alphanumeric, spaces, and basic punctuation
        sanitized = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)
        if not sanitized:
            print("[AudioCommander] No valid text to speak after sanitization")
            return

        # Turn speaker on
        if not self.speaker_on(200):
            print("[AudioCommander] Failed to turn speaker on")
            return
        time.sleep(0.1)

        # Use espeak to speak the text (text passed as list element, not shell-parsed)
        # Note: Don't use capture_output=True as it suppresses audio output
        try:
            subprocess.run(['espeak', '--', sanitized], timeout=10)
        except FileNotFoundError:
            print("[AudioCommander] espeak not found - install with: sudo apt install espeak")
        except subprocess.TimeoutExpired:
            print("[AudioCommander] espeak timed out")
        except Exception as e:
            print(f"[AudioCommander] espeak error: {e}")

        time.sleep(0.1)
        # Turn speaker off
        self.speaker_off(200)


# For testing
if __name__ == '__main__':
    print("Testing AudioCommander...")

    commander = AudioCommander(volume=0.5)

    print("Testing forward tone (1614 Hz)...")
    commander.forward(500)
    time.sleep(0.5)

    print("Testing left tone (2208 Hz)...")
    commander.left(500)
    time.sleep(0.5)

    print("Testing right tone (1811 Hz)...")
    commander.right(500)
    time.sleep(0.5)

    print("Testing backward tone (2013 Hz)...")
    commander.backward(500)
    time.sleep(0.5)

    print("Test complete!")
