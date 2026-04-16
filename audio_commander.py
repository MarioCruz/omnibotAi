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
import threading

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

        # Thread safety for speech processes
        self._proc_lock = threading.Lock()
        self._espeak_proc = None
        self._pw_proc = None
        self._stopping = False  # Flag to signal speech should stop

        # Try system player first (routes through PipeWire/PulseAudio for Bluetooth support)
        self._init_system_player()

        # Fall back to sounddevice if no system player available
        if not self.audio_available and SOUNDDEVICE_AVAILABLE:
            self._init_audio_device()

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
            sd.wait()  # Block until playback finishes (simpler and safer)
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
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass
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
        # Also kill any running speech and send speaker off
        self.stop_speaking()

    def stop_speaking(self):
        """Kill any running speech and send speaker_off to reset robot."""
        # Set stopping flag to signal speak() to abort
        self._stopping = True

        # Kill processes with lock protection
        with self._proc_lock:
            espeak = self._espeak_proc
            pw = self._pw_proc
            self._espeak_proc = None
            self._pw_proc = None

        # Kill outside lock to avoid deadlock
        if espeak:
            try:
                espeak.kill()
            except Exception:
                pass
        if pw:
            try:
                pw.kill()
            except Exception:
                pass

        # Kill only our tracked espeak processes, not system-wide
        try:
            subprocess.run(['pkill', '-f', 'espeak-ng'], capture_output=True, timeout=1)
        except Exception:
            pass

        # Small delay after killing processes
        time.sleep(0.2)

        # Send speaker_off tone, THEN reset stopping flag
        # (keeps _stopping=True until tone is sent so speak() can't start mid-tone)
        try:
            sox_cmd = ['sox', '-n', '-t', 'wav', '-', 'synth', '0.5', 'sine', str(self.FREQUENCIES['speaker_off'])]
            pw_cmd = ['pw-play', '-']
            sox_proc = subprocess.Popen(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            pw_proc = subprocess.Popen(pw_cmd, stdin=sox_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pw_proc.wait(timeout=3)
            print("[AudioCommander] Speech stopped, speaker off sent via sox")
        except Exception as e:
            # Fallback to regular method
            print(f"[AudioCommander] sox method failed ({e}), using fallback")
            self.speaker_off(300)
        finally:
            # Reset stopping flag AFTER speaker_off tone is done
            self._stopping = False

    def speak(self, text: str):
        """
        Speak text using the speak_pi.sh script (proven to work with Bluetooth).

        Args:
            text: Text to speak (sanitized to alphanumeric, spaces, and basic punctuation)
        """
        import re

        # Sanitize text to prevent command injection
        sanitized = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)
        if not sanitized:
            print("[AudioCommander] No valid text to speak after sanitization")
            return

        # Use speak_pi.sh script which is proven to work with Bluetooth
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speak_pi.sh')

        try:
            with self._proc_lock:
                if self._stopping:
                    return
                self._espeak_proc = subprocess.Popen(
                    ['bash', script_path, sanitized],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                proc = self._espeak_proc

            if proc:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("[AudioCommander] speak script timed out")
                    proc.kill()

            with self._proc_lock:
                self._espeak_proc = None

        except FileNotFoundError:
            print(f"[AudioCommander] speak_pi.sh not found at {script_path}")
        except Exception as e:
            print(f"[AudioCommander] speak error: {e}")

    # Valid pre-recorded phrase names
    VALID_PHRASES = {'hello', 'yes', 'no', 'thanks', 'omnibot'}

    def speak_phrase(self, phrase: str):
        """
        Play a pre-recorded phrase (faster than generating speech).

        Args:
            phrase: Phrase name (hello, yes, no, thanks, omnibot)
        """
        # Validate against whitelist to prevent injection
        if phrase not in self.VALID_PHRASES:
            print(f"[AudioCommander] Unknown phrase: {phrase}")
            return

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speak_phrase.sh')

        try:
            with self._proc_lock:
                if self._stopping:
                    return
                self._espeak_proc = subprocess.Popen(
                    ['bash', script_path, phrase],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                proc = self._espeak_proc

            if proc:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print("[AudioCommander] speak_phrase timed out")
                    proc.kill()

            with self._proc_lock:
                self._espeak_proc = None

        except FileNotFoundError:
            print(f"[AudioCommander] speak_phrase.sh not found at {script_path}")
        except Exception as e:
            print(f"[AudioCommander] speak_phrase error: {e}")


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
