#!/usr/bin/env python3
"""
Robot Command Executor
Executes movement commands via audio tones for Tomy Omnibot
"""

from dataclasses import dataclass
from typing import Optional
import time

try:
    from audio_commander import AudioCommander
except ImportError:
    AudioCommander = None


@dataclass
class CommandResult:
    """Result of a robot command execution"""
    success: bool
    message: str = ""


class RobotCommandExecutor:
    """
    Executes robot commands via audio frequency tones.

    The Tomy Omnibot uses audio signals for control:
    - Forward: 1614 Hz
    - Backward: 2013 Hz
    - Left: 2208 Hz
    - Right: 1811 Hz
    - Speaker On: 1422 Hz
    - Speaker Off: 4650 Hz
    """

    def __init__(self, volume: float = 0.5, robot_url: Optional[str] = None):
        """
        Initialize the robot executor.

        Args:
            volume: Audio volume for tones (0.0 to 1.0)
            robot_url: Unused for audio control (kept for compatibility)
        """
        # Clamp volume to valid range
        self.volume = max(0.0, min(1.0, volume))
        self.connected = False
        self.audio = None
        self._cancel_pattern = False  # Flag to cancel running patterns

        # Movement durations (ms)
        self.step_duration = 500
        self.turn_duration = 750  # Short turn for course correction (was 3000 = 90°)

        # Pattern definitions
        self.patterns = {
            'dance': ['left', 'right', 'left', 'right', 'forward', 'backward'],
            'circle': ['forward', 'left'] * 4,
            'square': ['forward', 'left', 'forward', 'left', 'forward', 'left', 'forward', 'left'],
            'triangle': ['forward', 'left', 'forward', 'left', 'forward', 'left'],
            'zigzag': ['forward', 'right', 'forward', 'left'] * 3,
            'spiral': ['forward', 'left', 'forward', 'forward', 'left', 'forward', 'forward', 'forward', 'left'],
            'search': ['forward', 'left', 'forward', 'right', 'right', 'forward', 'left', 'left'],
            'patrol': ['forward', 'forward', 'forward', 'left', 'left', 'forward', 'forward', 'forward', 'left', 'left'],
            'wave': ['left', 'right', 'left', 'right']
        }

    def connect(self) -> bool:
        """Initialize audio connection"""
        try:
            if AudioCommander is not None:
                self.audio = AudioCommander(volume=self.volume)
                self.connected = True
                print(f"[Robot] Audio commander initialized (volume: {self.volume})")
                return True
            else:
                print("[Robot] AudioCommander not available")
                self.connected = False
                return False
        except Exception as e:
            print(f"[Robot] Connection error: {e}")
            self.connected = False
            return False

    def stop(self):
        """Stop any current movement/audio"""
        if self.audio:
            self._cancel_pattern = True
            self.audio.stop()

    def disconnect(self):
        """Cleanup audio resources"""
        self.stop()
        self.connected = False
        print("[Robot] Disconnected")

    def execute(self, command: str) -> CommandResult:
        """
        Execute a robot command.

        Args:
            command: Command string (e.g., 'forward', 'left', 'dance', 'speakText("Hello")')

        Returns:
            CommandResult with success status
        """
        if not self.connected or not self.audio:
            return CommandResult(False, "Not connected")

        command = command.strip().lower()

        try:
            # Movement commands
            if command == 'forward':
                self.audio.forward(self.step_duration)
                return CommandResult(True, "Moving forward")

            elif command == 'backward':
                self.audio.backward(self.step_duration)
                return CommandResult(True, "Moving backward")

            elif command == 'left':
                self.audio.left(self.turn_duration)
                return CommandResult(True, "Turning left")

            elif command == 'right':
                self.audio.right(self.turn_duration)
                return CommandResult(True, "Turning right")

            elif command == 'stop':
                self._cancel_pattern = True  # Cancel any running pattern
                self.audio.stop()
                return CommandResult(True, "Stopped")

            # Pattern commands
            elif command in self.patterns:
                self._run_pattern(command)
                return CommandResult(True, f"Running pattern: {command}")

            # Speech commands
            elif command.startswith('speaktext('):
                # Extract text from speakText("...")
                start = command.find('"') + 1
                end = command.rfind('"')
                if start > 0 and end > start:
                    text = command[start:end]
                    self.audio.speak(text)
                    return CommandResult(True, f"Speaking: {text}")
                return CommandResult(False, "Invalid speakText format")

            # Pre-recorded phrase commands (faster)
            elif command.startswith('phrase('):
                # Extract phrase name from phrase("...")
                start = command.find('"') + 1
                end = command.rfind('"')
                if start > 0 and end > start:
                    phrase = command[start:end]
                    self.audio.speak_phrase(phrase)
                    return CommandResult(True, f"Phrase: {phrase}")
                return CommandResult(False, "Invalid phrase format")

            # Step commands with direction
            elif command.startswith('step('):
                direction = command[5:-1].strip('"\'')
                return self.execute(direction)

            # Run pattern commands
            elif command.startswith('runpattern('):
                pattern = command[11:-1].strip('"\'')
                if pattern in self.patterns:
                    self._run_pattern(pattern)
                    return CommandResult(True, f"Running pattern: {pattern}")
                return CommandResult(False, f"Unknown pattern: {pattern}")

            else:
                return CommandResult(False, f"Unknown command: {command}")

        except Exception as e:
            return CommandResult(False, f"Error: {e}")

    def execute_sequence(self, commands, delay: float = 0.3):
        """Execute a list of commands with a fixed delay between each."""
        results = []
        for cmd in commands:
            results.append(self.execute(cmd))
            time.sleep(max(0.0, delay))
        return results

    def _run_pattern(self, pattern_name: str):
        """Execute a movement pattern (can be cancelled with stop command)"""
        if pattern_name not in self.patterns:
            return

        self._cancel_pattern = False  # Reset cancel flag
        steps = self.patterns[pattern_name]
        for step in steps:
            if self._cancel_pattern:
                print(f"[Robot] Pattern '{pattern_name}' cancelled")
                break
            if step == 'forward':
                self.audio.forward(self.step_duration)
            elif step == 'backward':
                self.audio.backward(self.step_duration)
            elif step == 'left':
                self.audio.left(self.turn_duration // 4)  # Quarter turn
            elif step == 'right':
                self.audio.right(self.turn_duration // 4)
            time.sleep(0.1)  # Brief pause between steps


# For testing
if __name__ == '__main__':
    print("Testing robot executor...")

    robot = RobotCommandExecutor(volume=0.5)
    if robot.connect():
        print("Testing forward...")
        result = robot.execute('forward')
        print(f"Result: {result}")

        time.sleep(1)

        print("Testing stop...")
        result = robot.execute('stop')
        print(f"Result: {result}")

        robot.disconnect()
    else:
        print("Failed to connect")
