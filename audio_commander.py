#!/usr/bin/env python3
"""
Audio Commander for Tomy Omnibot
Generates audio frequency tones to control the robot
"""

import numpy as np
import time
import subprocess
import tempfile
import wave
import os

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[AudioCommander] sounddevice not available - will try system audio players")


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
        self.audio_device = None
        self.audio_available = False
        self.use_system_player = False
        self.system_player = None

        # Try sounddevice first
        if SOUNDDEVICE_AVAILABLE:
            self._init_audio_device()

        # If sounddevice didn't work, try system audio players (for PipeWire/PulseAudio/Bluetooth)
        if not self.audio_available:
            self._init_system_player()

    def _init_audio_device(self):
        """Initialize and validate sounddevice audio output."""
        try:
            devices = sd.query_devices()
            default_output = sd.default.device[1]

            if default_output is not None and default_output >= 0:
                try:
                    device_info = sd.query_devices(default_output, 'output')
                    if device_info['max_output_channels'] > 0:
                        self.audio_device = default_output
                        self.audio_available = True
                        print(f"[AudioCommander] Using sounddevice: {device_info['name']}")
                        return
                except Exception:
                    pass

            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    try:
                        sd.check_output_settings(device=i, samplerate=self.sample_rate)
                        self.audio_device = i
                        self.audio_available = True
                        print(f"[AudioCommander] Using sounddevice: {dev['name']}")
                        return
                    except Exception:
                        continue

        except Exception as e:
            print(f"[AudioCommander] sounddevice init error: {e}")

    def _init_system_player(self):
        """Initialize system audio player fallback (pw-play, paplay, or aplay)."""
        # Try players in order of preference for PipeWire/Bluetooth support
        players = ['pw-play', 'paplay', 'aplay']

        for player in players:
            try:
                # Check if player exists
                result = subprocess.run(['which', player], capture_output=True, timeout=2)
                if result.returncode == 0:
                    self.system_player = player
                    self.use_system_player = True
                    self.audio_available = True
                    print(f"[AudioCommander] Using system player: {player}")
                    return
            except Exception:
                continue

        print("[AudioCommander] Warning: No audio output available")

    def _generate_wav_data(self, frequency: float, duration_ms: int) -> bytes:
        """Generate WAV file data for a tone."""
        duration_s = duration_ms / 1000.0
        n_samples = int(self.sample_rate * duration_s)
        t = np.linspace(0, duration_s, n_samples, False)
        tone = np.sin(2 * np.pi * frequency * t) * self.volume

        # Convert to 16-bit PCM
        audio_data = (tone * 32767).astype(np.int16)

        # Create WAV in memory
        import io
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return buffer.getvalue()

    def _generate_tone(self, frequency: float, duration_ms: int) -> np.ndarray:
        """Generate a sine wave tone."""
        duration_s = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(self.sample_rate * duration_s), False)
        tone = np.sin(2 * np.pi * frequency * t) * self.volume
        return tone.astype(np.float32)

    def _play_tone(self, frequency: float, duration_ms: int) -> bool:
        """Play a tone at the specified frequency and duration. Returns True if successful."""
        if not self.audio_available:
            return False

        # Use system player (pw-play/paplay/aplay) for PipeWire/Bluetooth support
        if self.use_system_player:
            return self._play_tone_system(frequency, duration_ms)

        # Use sounddevice if available
        if not SOUNDDEVICE_AVAILABLE:
            return False

        try:
            self.is_playing = True
            tone = self._generate_tone(frequency, duration_ms)
            sd.play(tone, self.sample_rate, device=self.audio_device)
            timeout_s = (duration_ms / 1000.0) + 1.0
            deadline = time.time() + timeout_s
            while sd.get_stream() and sd.get_stream().active:
                if time.time() > deadline:
                    sd.stop()
                    break
                time.sleep(0.01)
            self.is_playing = False
            return True
        except Exception as e:
            print(f"[AudioCommander] Error playing tone: {e}")
            self.is_playing = False
            return False

    def _play_tone_system(self, frequency: float, duration_ms: int) -> bool:
        """Play tone using system audio player (pw-play/paplay/aplay)."""
        try:
            self.is_playing = True

            # Generate WAV data
            wav_data = self._generate_wav_data(frequency, duration_ms)

            # Stream directly to player via stdin (faster than temp files)
            timeout_s = (duration_ms / 1000.0) + 2.0

            # pw-play and paplay can read from stdin with '-'
            if self.system_player in ['pw-play', 'paplay']:
                proc = subprocess.Popen(
                    [self.system_player, '-'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                proc.communicate(input=wav_data, timeout=timeout_s)
            else:
                # aplay needs explicit format flags for stdin
                proc = subprocess.Popen(
                    ['aplay', '-f', 'S16_LE', '-r', str(self.sample_rate), '-c', '1', '-'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # For aplay, send raw PCM (skip WAV header)
                duration_s = duration_ms / 1000.0
                n_samples = int(self.sample_rate * duration_s)
                t = np.linspace(0, duration_s, n_samples, False)
                tone = np.sin(2 * np.pi * frequency * t) * self.volume
                audio_data = (tone * 32767).astype(np.int16).tobytes()
                proc.communicate(input=audio_data, timeout=timeout_s)

            self.is_playing = False
            return True

        except subprocess.TimeoutExpired:
            print(f"[AudioCommander] Playback timeout")
            self.is_playing = False
            return False
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
        import re

        # Sanitize text to prevent command injection
        sanitized = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)
        if not sanitized:
            print("[AudioCommander] No valid text to speak after sanitization")
            return

        # Turn speaker on (don't fail if audio unavailable - espeak might still work)
        self.speaker_on(200)
        time.sleep(0.1)

        # Use espeak-ng (or espeak) - outputs to default audio via PipeWire
        try:
            # Try espeak-ng first, fall back to espeak
            try:
                subprocess.run(['espeak-ng', '--', sanitized], timeout=10,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                subprocess.run(['espeak', '--', sanitized], timeout=10,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[AudioCommander] espeak not found - install with: sudo apt install espeak-ng")
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
