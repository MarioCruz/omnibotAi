#!/usr/bin/env python3
"""
Rule-based navigation engine for Tomy Omnibot.
Uses bounding box positions to generate movement commands — no LLM needed.
"""

import re
import time
from typing import List, Dict

# Pull COCO labels from the detector so both modules stay in sync.
from object_detector import ObjectDetector as _Detector
COCO_LABELS = [l.lower() for l in _Detector.labels]

# Common aliases so kids / non-technical users don't have to know COCO names.
# Keep this tight — matching too aggressively means "find the mousepad"
# drives toward a mouse.
LABEL_ALIASES = {
    'human': 'person',
    'people': 'person',
    'kid': 'person',
    'child': 'person',
    'sofa': 'couch',
    'couch': 'couch',
    'kitty': 'cat',
    'puppy': 'dog',
    'phone': 'cell phone',
    'mobile': 'cell phone',
    'television': 'tv',
    'bottle': 'bottle',
    'ball': 'sports ball',
    'soda': 'bottle',
}


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

    @staticmethod
    def targets_from_context(context: str):
        """Extract COCO labels mentioned in a free-form task string.

        "Find the laptop"       -> {'laptop'}
        "Go find my kitty"      -> {'cat'}           (alias)
        "Find a person or cat"  -> {'person', 'cat'}
        ""                      -> set()             (no filter)

        Uses word boundaries so "tv" doesn't match "gravity" and
        "cat" doesn't match "catch".
        """
        if not context:
            return set()
        ctx = context.lower()
        hits = set()
        # Direct COCO label hits (longest first so "sports ball" wins over "ball")
        for label in sorted(COCO_LABELS, key=len, reverse=True):
            if re.search(rf'\b{re.escape(label)}\b', ctx):
                hits.add(label)
        # Aliases
        for alias, canonical in LABEL_ALIASES.items():
            if re.search(rf'\b{re.escape(alias)}\b', ctx):
                hits.add(canonical)
        return hits

    def generate_commands(self, detections: List[Dict], context: str = "") -> List[str]:
        """
        Generate movement commands from detections.

        Args:
            detections: List of dicts with 'label', 'confidence', 'bbox'
            context: Task description. If it names any COCO label (or an
                alias like "human"/"kitty"), detections are filtered to
                only those labels — so "Find the laptop" won't drive
                toward a person that happens to be in frame.

        Returns:
            List of command strings for RobotCommandExecutor
        """
        if not detections:
            return []

        commands = []
        log = []

        # If the task names one or more targets, only consider those labels.
        # If the target is simply not in frame right now, return no commands
        # so the robot holds position until it reappears. (A real search
        # pattern is future work.)
        targets = self.targets_from_context(context)
        filtered = detections
        if targets:
            filtered = [d for d in detections if d['label'].lower() in targets]
            if not filtered:
                visible = ', '.join(sorted({d['label'] for d in detections[:5]}))
                self.last_debug = {
                    'target': ', '.join(sorted(targets)) + ' (not visible)',
                    'position': '',
                    'response': f"target {sorted(targets)} not in frame — visible: {visible}",
                    'parsed_commands': [],
                    'mode': 'rules',
                    'timestamp': time.strftime('%H:%M:%S'),
                }
                return []

        # Pick the best target: highest confidence object that matches approach rules,
        # or highest confidence overall if nothing matches
        sorted_dets = sorted(filtered, key=lambda d: d['confidence'], reverse=True)

        target = None
        # When the user named a specific target, prefer that label over generic
        # APPROACH_RULES picks (even if a person has higher confidence).
        if targets:
            for det in sorted_dets:
                if det['label'].lower() in targets:
                    target = det
                    break
        if target is None:
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
