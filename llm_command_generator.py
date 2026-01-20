#!/usr/bin/env python3
"""
LLM Command Generator Module
Uses local Ollama or cloud LLMs to generate robot commands from detected objects
"""

import requests
import json
import re
import time
import os
from typing import List, Dict, Optional


class LLMCommandGenerator:
    """Generate robot commands using LLM based on detected objects"""

    def __init__(self,
                 model_name: str = "llama-3.1-8b-instant",
                 api_url: str = "http://localhost:11434",
                 use_cloud: bool = True,
                 cloud_api_key: str = None,
                 cloud_provider: str = "groq",
                 frame_width: int = 640,
                 frame_height: int = 480):

        self.model_name = model_name
        self.api_url = api_url
        self.use_cloud = use_cloud
        self.cloud_api_key = cloud_api_key or os.environ.get('GROQ_API_KEY')
        self.cloud_provider = cloud_provider
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Available robot commands mapped to function calls
        self.robot_commands = {
            'forward': 'step("forward")',
            'backward': 'step("backward")',
            'left': 'step("left")',
            'right': 'step("right")',
            'stop': 'step("stop")',
            'dance': 'runPattern("dance")',
            'circle': 'runPattern("circle")',
            'square': 'runPattern("square")',
            'triangle': 'runPattern("triangle")',
            'zigzag': 'runPattern("zigzag")',
            'spiral': 'runPattern("spiral")',
            'search': 'runPattern("search")',
            'patrol': 'runPattern("patrol")',
            'wave': 'runPattern("wave")',
            'greet': 'speakText("Hello!")',
        }

        # Simple rule-based responses as fallback (no speech - too slow)
        self.simple_rules = {
            'person': ['forward'],  # Approach person (don't speak - blocks processing)
            'cat': ['forward', 'stop'],
            'dog': ['forward', 'stop'],
            'ball': ['forward', 'forward'],
            'sports ball': ['forward', 'forward'],
            'chair': ['left', 'forward'],
            'bottle': ['forward'],
            'cup': ['forward'],
        }

        # Debug info - stores last prompt/response for UI visibility
        self.last_debug = {
            'prompt': '',
            'response': '',
            'parsed_commands': [],
            'mode': 'none',  # 'llm', 'rules', or 'none'
            'timestamp': ''
        }

        if self.use_cloud:
            print(f"[LLM] Initialized with cloud ({cloud_provider}): {model_name}")
        else:
            print(f"[LLM] Initialized with local Ollama: {model_name}")
        print(f"[LLM] Available commands: {list(self.robot_commands.keys())}")

    def generate_commands(self,
                         detected_objects: List[Dict],
                         context: str = "",
                         use_llm: bool = True) -> List[str]:
        """
        Generate robot commands based on detected objects

        Args:
            detected_objects: List of detection dicts with label, confidence, bbox
            context: Task context (e.g., "Find and approach balls")
            use_llm: If False, use simple rule-based logic

        Returns:
            List of robot command strings
        """
        if not detected_objects:
            return []

        if use_llm:
            try:
                return self._generate_with_llm(detected_objects, context)
            except Exception as e:
                print(f"[LLM] Error: {e}, falling back to rules")
                return self._generate_with_rules(detected_objects)
        else:
            return self._generate_with_rules(detected_objects)

    def _generate_with_llm(self, objects: List[Dict], context: str) -> List[str]:
        """Generate commands using LLM"""
        import time as _time
        object_summary = self._format_objects(objects)
        prompt = self._create_prompt(object_summary, context)

        if self.use_cloud:
            response = self._call_cloud_llm(prompt)
        else:
            response = self._call_ollama(prompt)

        commands = self._parse_response(response)

        # Store debug info
        self.last_debug = {
            'prompt': prompt,
            'response': response,
            'parsed_commands': commands,
            'mode': 'llm',
            'timestamp': _time.strftime('%H:%M:%S')
        }

        return commands

    def _generate_with_rules(self, objects: List[Dict]) -> List[str]:
        """Generate commands using simple rules (no LLM needed)"""
        import time as _time
        commands = []
        rule_log = []

        # Sort by confidence, process highest first
        sorted_objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)

        for obj in sorted_objects[:3]:  # Process top 3 objects
            label = obj['label'].lower()
            bbox = obj['bbox']

            # Check if object is to left, right, or center
            frame_center_x = self.frame_width / 2
            obj_center_x = bbox['x'] + bbox['width'] / 2

            if label in self.simple_rules:
                rule_commands = self.simple_rules[label]
                for cmd in rule_commands:
                    if cmd in self.robot_commands:
                        commands.append(self.robot_commands[cmd])
                        rule_log.append(f"Rule: {label} -> {cmd}")

            # Add directional adjustment based on position (threshold is 15% of frame width)
            threshold = self.frame_width * 0.15
            if obj_center_x < frame_center_x - threshold:
                commands.insert(0, self.robot_commands['left'])
                rule_log.insert(0, f"Position: {label} is LEFT of center -> turn left")
            elif obj_center_x > frame_center_x + threshold:
                commands.insert(0, self.robot_commands['right'])
                rule_log.insert(0, f"Position: {label} is RIGHT of center -> turn right")

        result = commands[:5]  # Limit to 5 commands

        # Store debug info for rules mode
        self.last_debug = {
            'prompt': f"Rule-based mode (no LLM)\nObjects: {[o['label'] for o in sorted_objects[:3]]}",
            'response': '\n'.join(rule_log) if rule_log else 'No matching rules',
            'parsed_commands': result,
            'mode': 'rules',
            'timestamp': _time.strftime('%H:%M:%S')
        }

        return result

    def _format_objects(self, objects: List[Dict]) -> str:
        """Format detected objects for LLM prompt"""
        if not objects:
            return "No objects detected."

        summary = "Detected objects:\n"
        for obj in objects:
            bbox = obj['bbox']
            position = self._describe_position(bbox)
            summary += f"  - {obj['label']} ({obj['confidence']:.0%}) at {position}\n"

        return summary

    def _describe_position(self, bbox: Dict) -> str:
        """Describe object position in natural language"""
        x, y = bbox['x'], bbox['y']
        w, h = bbox['width'], bbox['height']

        center_x = x + w / 2
        center_y = y + h / 2

        # Calculate position thresholds based on frame dimensions
        left_threshold = self.frame_width / 3
        right_threshold = self.frame_width * 2 / 3
        top_threshold = self.frame_height / 3
        bottom_threshold = self.frame_height * 2 / 3

        h_pos = "center"
        if center_x < left_threshold:
            h_pos = "left"
        elif center_x > right_threshold:
            h_pos = "right"

        v_pos = "middle"
        if center_y < top_threshold:
            v_pos = "top"
        elif center_y > bottom_threshold:
            v_pos = "bottom"

        # Estimate distance based on size
        area = w * h
        if area > 50000:
            dist = "very close"
        elif area > 20000:
            dist = "close"
        elif area > 5000:
            dist = "medium distance"
        else:
            dist = "far away"

        return f"{h_pos}-{v_pos}, {dist}"

    def _create_prompt(self, object_summary: str, context: str) -> str:
        """Create prompt for LLM"""
        available = ', '.join(self.robot_commands.keys())

        prompt = f"""You are a robot control AI. Generate movement commands based on what the camera sees.

Available commands: {available}

Current view:
{object_summary}

Task: {context if context else 'Explore and interact with objects'}

Rules:
- Generate 1-3 movement commands only (no speech commands)
- Move toward interesting objects
- Turn left/right to center objects in view
- Avoid obstacles by turning
- If nothing interesting, search or patrol

Respond with ONLY a JSON object:
{{"commands": ["cmd1", "cmd2"]}}"""

        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API"""
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 100,
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('response', '{}')
        except requests.exceptions.ConnectionError:
            print("[LLM] Ollama not running. Start with: ollama serve")
            raise
        except Exception as e:
            print(f"[LLM] Ollama error: {e}")
            raise

    def _call_cloud_llm(self, prompt: str) -> str:
        """Call cloud LLM API (Groq, OpenAI compatible)"""
        if not self.cloud_api_key:
            raise ValueError("Cloud API key not set. Set GROQ_API_KEY environment variable.")

        if self.cloud_provider == "groq":
            return self._call_groq(prompt)
        else:
            raise NotImplementedError(f"Cloud provider {self.cloud_provider} not supported.")

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API (OpenAI compatible)"""
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cloud_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a robot control AI. Respond only with JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 100,
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            print("[LLM] Groq timeout")
            raise
        except Exception as e:
            print(f"[LLM] Groq error: {e}")
            raise

    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response into validated commands"""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                raw_commands = data.get('commands', [])

                # Validate and convert commands
                validated = []
                for cmd in raw_commands:
                    cmd_lower = cmd.lower().strip()
                    if cmd_lower in self.robot_commands:
                        validated.append(self.robot_commands[cmd_lower])

                return validated
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"[LLM] Parse error: {e}")

        return []

    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'].split(':')[0] for m in models]
                if self.model_name in model_names:
                    print(f"[LLM] Ollama ready with {self.model_name}")
                    return True
                else:
                    print(f"[LLM] Model {self.model_name} not found. Available: {model_names}")
                    print(f"[LLM] Run: ollama pull {self.model_name}")
                    return False
        except requests.exceptions.ConnectionError:
            print("[LLM] Ollama not running. Start with: ollama serve")
            return False
        except Exception as e:
            print(f"[LLM] Status check error: {e}")
            return False


# For testing
if __name__ == '__main__':
    print("Testing LLM Command Generator...")

    llm = LLMCommandGenerator()

    # Check Ollama status
    ollama_ready = llm.check_ollama_status()

    # Test with sample detections
    test_objects = [
        {'label': 'person', 'confidence': 0.92, 'bbox': {'x': 200, 'y': 100, 'width': 150, 'height': 300}},
        {'label': 'chair', 'confidence': 0.78, 'bbox': {'x': 50, 'y': 200, 'width': 100, 'height': 150}},
    ]

    # Test rule-based generation (always works)
    print("\nRule-based commands:")
    commands = llm.generate_commands(test_objects, use_llm=False)
    for cmd in commands:
        print(f"  {cmd}")

    # Test LLM generation (requires Ollama)
    if ollama_ready:
        print("\nLLM-based commands:")
        commands = llm.generate_commands(test_objects, context="Greet any people you see", use_llm=True)
        for cmd in commands:
            print(f"  {cmd}")
    else:
        print("\nSkipping LLM test (Ollama not available)")

    print("\nTest complete")
