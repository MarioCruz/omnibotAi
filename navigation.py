#!/usr/bin/env python3
"""
Rule-based navigation engine for Tomy Omnibot.
Uses bounding box positions to generate movement commands — no LLM needed.
"""

import time
from typing import List, Dict


class NavigationEngine:
    """Generate robot movement commands from object detections using position math."""

    # Robot commands (same format as RobotCommandExecutor expects)
    COMMANDS = {
        'forward': 'step("forward")',
        'backward': 'step("backward")',
        'left': 'step("left")',
        'right': 'step("right")',
        'stop': 'step("stop")',
    }

    # What to do when an object is centered (label -> commands)
    APPROACH_RULES = {
        'person': ['forward'],
        'cat': ['forward', 'stop'],
        'dog': ['forward', 'stop'],
        'ball': ['forward', 'forward'],
        'sports ball': ['forward', 'forward'],
        'chair': ['left', 'forward'],
        'bottle': ['forward'],
        'cup': ['forward'],
    }

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.last_debug = {}

    def generate_commands(self, detections: List[Dict], context: str = "") -> List[str]:
        """
        Generate movement commands from detections.

        Args:
            detections: List of dicts with 'label', 'confidence', 'bbox'
            context: Task description (unused for now, reserved for future use)

        Returns:
            List of command strings for RobotCommandExecutor
        """
        if not detections:
            return []

        commands = []
        log = []

        # Pick the best target: highest confidence object that matches approach rules,
        # or highest confidence overall if nothing matches
        sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)

        target = None
        for det in sorted_dets:
            if det['label'].lower() in self.APPROACH_RULES:
                target = det
                break
        if target is None:
            target = sorted_dets[0]

        label = target['label'].lower()
        bbox = target['bbox']
        obj_center_x = bbox['x'] + bbox['width'] / 2
        frame_center_x = self.frame_width / 2

        # 15% dead zone in the center to avoid jittering
        threshold = self.frame_width * 0.15

        if obj_center_x < frame_center_x - threshold:
            # Target is left — turn to face it
            commands.append(self.COMMANDS['left'])
            log.append(f"{label} at x:{bbox['x']} -> LEFT of center -> turn left")

        elif obj_center_x > frame_center_x + threshold:
            # Target is right — turn to face it
            commands.append(self.COMMANDS['right'])
            log.append(f"{label} at x:{bbox['x']} -> RIGHT of center -> turn right")

        else:
            # Target is centered — approach or use specific rule
            area = bbox['width'] * bbox['height']
            if area > (self.frame_width * self.frame_height * 0.6):
                # Object fills >60% of frame — we're very close, stop
                commands.append(self.COMMANDS['stop'])
                log.append(f"{label} fills {area/(self.frame_width*self.frame_height):.0%} of frame -> STOP (close enough)")
            elif label in self.APPROACH_RULES:
                for cmd in self.APPROACH_RULES[label]:
                    commands.append(self.COMMANDS[cmd])
                log.append(f"{label} CENTERED -> {', '.join(self.APPROACH_RULES[label])}")
            else:
                commands.append(self.COMMANDS['forward'])
                log.append(f"{label} CENTERED -> forward")

        result = commands[:3]

        self.last_debug = {
            'target': f"{target['label']} ({target['confidence']:.0%})",
            'position': f"x:{bbox['x']} cx:{int(obj_center_x)} frame_cx:{int(frame_center_x)}",
            'response': ' | '.join(log),
            'parsed_commands': result,
            'mode': 'rules',
            'timestamp': time.strftime('%H:%M:%S')
        }

        return result
