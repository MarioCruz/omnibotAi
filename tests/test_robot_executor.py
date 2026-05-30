#!/usr/bin/env python3
"""
Unit tests for RobotCommandExecutor duration handling.

Covers the config clamps and the quarter-turn floor — a low turn_duration must
not produce a sub-100ms pattern tone that the robot's audio relay silently
drops. No audio is emitted (we never call connect()).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_executor import RobotCommandExecutor


class DurationClamps(unittest.TestCase):
    def test_durations_clamped_to_safe_floor(self):
        r = RobotCommandExecutor(step_duration=1, turn_duration=1, nudge_duration=1)
        self.assertEqual(r.step_duration, 100)
        self.assertEqual(r.turn_duration, 100)
        self.assertEqual(r.nudge_duration, 100)

    def test_durations_clamped_to_ceiling(self):
        r = RobotCommandExecutor(step_duration=99999, turn_duration=99999)
        self.assertEqual(r.step_duration, 2000)
        self.assertEqual(r.turn_duration, 3000)

    def test_quarter_turn_never_below_relay_minimum(self):
        # turn_duration=100 -> 100//4 == 25, which the relay would miss.
        r = RobotCommandExecutor(turn_duration=100)
        self.assertEqual(r._quarter_turn(), 100)

    def test_quarter_turn_normal_case(self):
        r = RobotCommandExecutor(turn_duration=750)
        self.assertEqual(r._quarter_turn(), 187)

    def test_volume_clamped(self):
        self.assertEqual(RobotCommandExecutor(volume=5.0).volume, 1.0)
        self.assertEqual(RobotCommandExecutor(volume=-1.0).volume, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
