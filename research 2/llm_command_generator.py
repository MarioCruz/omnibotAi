#!/usr/bin/env python3
"""
LLM Command Generator Module
Converts detected objects into robot commands using Ollama
"""

import requests
import json
import re
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMCommandGenerator:
    """Generate robot commands using local LLM (Ollama)"""

    # Mapping from LLM output to actual robot commands
    ROBOT_COMMANDS = {
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
    }

    def __init__(self, model_name: str = "mistral", 
                 api_url: str = "http://localhost:11434"):
        """
        Initialize LLM command generator

        Args:
            model_name: Ollama model name (mistral, neural-chat, orca-mini, etc)
            api_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.api_url = api_url
        self.model_available = self._check_model_available()

        if self.model_available:
            logger.info(f"LLM initialized: {model_name} at {api_url}")
        else:
            logger.warning(f"LLM not available at {api_url}")

    def _check_model_available(self) -> bool:
        """Check if Ollama model is available"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate_commands(self, detected_objects: List[Dict],
                         context: str = "") -> Tuple[List[str], str]:
        """
        Generate robot commands based on detected objects

        Args:
            detected_objects: List of detected objects
            context: Task context/goal

        Returns:
            Tuple of (commands list, reasoning string)
        """
        # Format object information
        object_summary = self._format_objects(detected_objects)

        # Create LLM prompt
        prompt = self._create_prompt(object_summary, context)

        # Call LLM
        if not self.model_available:
            logger.warning("LLM not available, using fallback commands")
            return self._fallback_commands(detected_objects)

        response = self._call_llm(prompt)

        # Parse response
        commands, reasoning = self._parse_response(response)

        return commands, reasoning

    def _format_objects(self, objects: List[Dict]) -> str:
        """Format detected objects into readable text"""
        if not objects:
            return "No objects detected in the field."

        summary = "Detected objects:\n"
        for i, obj in enumerate(objects, 1):
            label = obj['label']
            conf = obj['confidence']
            x = obj['bbox']['x']
            y = obj['bbox']['y']
            summary += f"  {i}. {label} (confidence: {conf:.1%}) at position ({x:.0f}, {y:.0f})\n"

        return summary

    def _create_prompt(self, object_summary: str, context: str) -> str:
        """Create LLM prompt for command generation"""
        available_cmds = ', '.join(self.ROBOT_COMMANDS.keys())

        prompt = f"""You are a robot control AI. Analyze the detected objects and 
generate appropriate robot commands.

AVAILABLE COMMANDS: {available_cmds}

CURRENT SITUATION:
{object_summary}

TASK: {context if context else 'Explore and interact with detected objects'}

Generate a sequence of 2-4 robot commands that makes sense for this situation.
Be specific and decisive. Use ONLY the available commands.

RESPOND WITH THIS JSON FORMAT ONLY:
{{
    "reasoning": "Brief one-sentence explanation",
    "commands": ["command1", "command2"]
}}"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Low temp for consistency
                    "top_p": 0.9,
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('response', '{}')
        except requests.exceptions.Timeout:
            logger.error("LLM request timeout")
            return '{}'
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return '{}'

    def _parse_response(self, response: str) -> Tuple[List[str], str]:
        """Parse LLM response into commands"""
        reasoning = "AI response"

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                return [], reasoning

            json_str = json_match.group()
            data = json.loads(json_str)

            reasoning = data.get('reasoning', 'Generated commands')
            raw_commands = data.get('commands', [])

            # Validate and convert commands
            validated_commands = []
            for cmd in raw_commands:
                cmd_lower = cmd.lower().strip()
                if cmd_lower in self.ROBOT_COMMANDS:
                    validated_commands.append(
                        self.ROBOT_COMMANDS[cmd_lower]
                    )
                else:
                    logger.warning(f"Unknown command from LLM: {cmd}")

            return validated_commands, reasoning

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return [], reasoning
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return [], reasoning

    def _fallback_commands(self, objects: List[Dict]) -> Tuple[List[str], str]:
        """Fallback command generation when LLM not available"""
        commands = []

        if not objects:
            commands = ['step("forward")', 'step("forward")']
            reasoning = "No objects found, exploring..."
        else:
            # Approach first object
            first_obj = objects[0]
            label = first_obj['label']

            if label == 'person':
                commands = ['speakText("Hello!")', 'runPattern("dance")']
            elif label == 'ball':
                commands = ['step("forward")', 'step("forward")']
            else:
                commands = ['step("forward")']

            reasoning = f"Found {label}, proceeding..."

        return commands, reasoning

if __name__ == "__main__":
    gen = LLMCommandGenerator()

    test_objects = [
        {'label': 'ball', 'confidence': 0.8, 'bbox': {'x': 100, 'y': 150}},
        {'label': 'person', 'confidence': 0.9, 'bbox': {'x': 300, 'y': 100}},
    ]

    commands, reasoning = gen.generate_commands(
        test_objects, 
        context="Find and interact with objects"
    )

    print(f"Reasoning: {reasoning}")
    print(f"Commands: {commands}")
