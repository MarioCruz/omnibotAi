#!/usr/bin/env python3
"""
Robot Command Executor Module
Sends commands to robot web interface
"""

import requests
import time
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotCommandExecutor:
    """Execute commands on the robot"""

    def __init__(self, robot_url: str = "http://localhost:5000", 
                 command_timeout: float = 2.0):
        """
        Initialize robot executor

        Args:
            robot_url: Base URL of robot control interface
            command_timeout: Timeout for command requests
        """
        self.robot_url = robot_url
        self.command_timeout = command_timeout
        self.last_command_time = 0
        self.min_interval = 0.2  # Min 200ms between commands
        self.command_history: List[Dict] = []

        logger.info(f"Robot executor initialized: {robot_url}")

    def execute_command(self, command: str) -> bool:
        """
        Execute single command on robot

        Args:
            command: Robot command string (e.g., 'step("forward")')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Rate limiting
            elapsed = time.time() - self.last_command_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            # Send command
            response = requests.post(
                f"{self.robot_url}/api/command",
                json={"command": command},
                timeout=self.command_timeout
            )

            success = response.status_code == 200

            # Record history
            self.command_history.append({
                'command': command,
                'timestamp': time.time(),
                'success': success,
                'status_code': response.status_code
            })

            self.last_command_time = time.time()

            if success:
                logger.debug(f"Command executed: {command}")
            else:
                logger.warning(f"Command failed: {command} (status: {response.status_code})")

            return success

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to robot at {self.robot_url}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Command timeout: {command}")
            return False
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False

    def execute_sequence(self, commands: List[str], delay: float = 0.5) -> int:
        """
        Execute multiple commands in sequence

        Args:
            commands: List of command strings
            delay: Delay between commands (seconds)

        Returns:
            Number of successful commands
        """
        successful = 0

        for i, cmd in enumerate(commands, 1):
            logger.info(f"Executing command {i}/{len(commands)}: {cmd}")

            if self.execute_command(cmd):
                successful += 1

            if i < len(commands):  # Don't sleep after last command
                time.sleep(delay)

        logger.info(f"Sequence complete: {successful}/{len(commands)} commands executed")
        return successful

    def execute_with_retry(self, command: str, max_retries: int = 3) -> bool:
        """
        Execute command with retry logic

        Args:
            command: Command to execute
            max_retries: Maximum retry attempts

        Returns:
            True if successful
        """
        for attempt in range(max_retries):
            if self.execute_command(command):
                return True

            if attempt < max_retries - 1:
                backoff = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{max_retries - 1} after {backoff}s")
                time.sleep(backoff)

        logger.error(f"Command failed after {max_retries} attempts: {command}")
        return False

    def emergency_stop(self) -> bool:
        """Execute emergency stop"""
        logger.warning("EMERGENCY STOP!")
        return self.execute_command('step("stop")')

    def get_history(self, last_n: int = 10) -> List[Dict]:
        """Get last N commands from history"""
        return self.command_history[-last_n:]

    def clear_history(self):
        """Clear command history"""
        self.command_history.clear()

if __name__ == "__main__":
    executor = RobotCommandExecutor()

    # Test commands
    test_commands = [
        'step("forward")',
        'step("left")',
        'runPattern("circle")',
        'speakText("Hello!")'
    ]

    executor.execute_sequence(test_commands, delay=1.0)

    print("\nCommand History:")
    for cmd_info in executor.get_history():
        status = "✓" if cmd_info['success'] else "✗"
        print(f"  {status} {cmd_info['command']}")
