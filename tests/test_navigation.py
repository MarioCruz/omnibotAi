#!/usr/bin/env python3
"""
Unit tests for the rule-based NavigationEngine.

These are pure-logic tests — no camera, no robot, no Pi hardware — so they run
on a laptop in milliseconds. They lock in the behavior that has actually bitten
us before (notably "Find the laptop" steering toward a person in frame).

Run:  python -m pytest tests/           (if pytest installed)
      python tests/test_navigation.py   (stdlib unittest, no deps)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navigation import NavigationEngine


def det(label, conf, x, width=40, height=40, y=220):
    """Build a detection dict in the shape NavigationEngine expects."""
    return {'label': label, 'confidence': conf,
            'bbox': {'x': x, 'y': y, 'width': width, 'height': height}}


class TargetsFromContext(unittest.TestCase):
    def test_empty_context_means_no_filter(self):
        self.assertEqual(NavigationEngine.targets_from_context(""), set())
        self.assertEqual(NavigationEngine.targets_from_context(None), set())

    def test_direct_coco_label(self):
        self.assertEqual(NavigationEngine.targets_from_context("find the laptop"), {'laptop'})

    def test_alias_maps_to_canonical(self):
        self.assertEqual(NavigationEngine.targets_from_context("go find my kitty"), {'cat'})
        self.assertEqual(NavigationEngine.targets_from_context("where is the puppy"), {'dog'})

    def test_multiple_targets(self):
        self.assertEqual(
            NavigationEngine.targets_from_context("find a person or a cat"),
            {'person', 'cat'},
        )

    def test_word_boundaries_avoid_false_hits(self):
        # "tv" must not match inside "gravity", "cat" not inside "category".
        self.assertNotIn('tv', NavigationEngine.targets_from_context("defy gravity"))
        self.assertNotIn('cat', NavigationEngine.targets_from_context("sort by category"))

    def test_ball_alias_resolves_to_sports_ball(self):
        self.assertEqual(NavigationEngine.targets_from_context("fetch the ball"), {'sports ball'})


class GenerateCommands(unittest.TestCase):
    def setUp(self):
        self.nav = NavigationEngine(frame_width=640, frame_height=480)

    def test_no_detections_returns_empty(self):
        self.assertEqual(self.nav.generate_commands([]), [])

    def test_centered_person_goes_forward(self):
        # center_x = 300 + 40/2 = 320 == frame center -> forward
        cmds = self.nav.generate_commands([det('person', 0.9, x=300)])
        self.assertEqual(cmds, ['step("forward")'])

    def test_far_left_full_turn(self):
        # center_x = 20, delta = -300 (> 0.35*640 far threshold) -> full left
        cmds = self.nav.generate_commands([det('person', 0.9, x=0)])
        self.assertIn('step("left")', cmds)

    def test_far_right_full_turn(self):
        cmds = self.nav.generate_commands([det('person', 0.9, x=600)])
        self.assertIn('step("right")', cmds)

    def test_slightly_left_nudges(self):
        # center_x = 200, delta = -120 (between near 96 and far 224) -> nudge
        cmds = self.nav.generate_commands([det('person', 0.9, x=180)])
        self.assertIn('nudge("left")', cmds)

    def test_object_filling_frame_stops(self):
        # centered + area > 60% of frame -> stop
        big = det('person', 0.9, x=70, width=500, height=400)
        cmds = self.nav.generate_commands([big])
        self.assertEqual(cmds, ['step("stop")'])

    # --- The regression that motivated context filtering ---
    def test_find_laptop_does_not_chase_person(self):
        # A high-confidence centered person and a low-confidence laptop off to
        # the left. "Find the laptop" must steer toward the laptop, never the
        # person — even though the person scores higher and is centered.
        dets = [
            det('person', 0.95, x=300),          # centered, would be "forward"
            det('laptop', 0.50, x=0, width=40),  # far left
        ]
        cmds = self.nav.generate_commands(dets, context="find the laptop")
        self.assertIn('step("left")', cmds)
        self.assertNotIn('step("forward")', cmds)
        self.assertEqual(self.nav.last_debug['target'].split()[0], 'laptop')

    def test_named_target_not_in_frame_holds_position(self):
        # Asked for a laptop, only a person is visible -> no commands (hold).
        cmds = self.nav.generate_commands([det('person', 0.9, x=300)],
                                          context="find the laptop")
        self.assertEqual(cmds, [])
        self.assertIn('not visible', self.nav.last_debug['target'])

    def test_command_list_capped_at_three(self):
        cmds = self.nav.generate_commands([det('person', 0.9, x=300)])
        self.assertLessEqual(len(cmds), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
