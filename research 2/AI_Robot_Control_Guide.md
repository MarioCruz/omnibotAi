
# AI-Powered Robot Control System Using Raspberry Pi AI Camera
## Complete Implementation Guide

---

## PART 1: SYSTEM ARCHITECTURE OVERVIEW

### Components:
1. **Raspberry Pi 5** (or Pi 4B)
2. **Raspberry Pi AI Camera** (12MP with neural accelerator)
3. **Robot Platform** (with motor control)
4. **LLM Integration** (Local or Cloud-based)
5. **Web Interface** (Your provided index.html)

### Data Flow:
Raspberry Pi AI Camera 
    -> Object Detection (YOLO/MediaPipe)
    -> LLM Processing (Generate commands)
    -> Robot Command Formatter
    -> Web Socket/HTTP to Robot Interface
    -> Robot Executes Movement

---

## PART 2: HARDWARE SETUP

### 2.1 Raspberry Pi Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable

# Enable SPI for neural accelerator
# Navigate to: Interface Options -> SPI -> Enable

# Reboot
sudo reboot
```

### 2.2 Install Required Libraries

```bash
# Python 3.11+
sudo apt install python3-pip python3-dev -y

mkdir -p ~/omnibotai
cd ~/omnibotai


# Create the venv (you can name it anything, here we call it .venv)
python3 -m venv .venv

source .venv/bin/activate


pip install --upgrade pip setuptools wheel


# OmniAi Installation Summary (Raspberry Pi OS, Python 3.13)


# Setup
cd ~/omnibotai
python3 -m venv .venv
source .venv/bin/activate

# System deps
sudo apt update
sudo apt install -y libcap-dev libtiff-dev libopenjp2-7 libjpeg62-turbo python3-dev

# Pip upgrade
pip install --upgrade pip setuptools wheel

# Create requirements.txt
cat > requirements.txt <<'EOF'
Pillow==11.0.0
numpy==2.0.0
opencv-python==4.10.0.84
picamera2[libav]
yolov5==7.0.13
ollama==0.1.5
requests==2.31.0
flask==3.0.0
flask-cors==4.0.0
flask-socketio==5.3.5
python-socketio==5.10.0
websocket-client==1.7.0
EOF

# Install
pip install -r requirements.txt

# Verify
python -c "import yolov5, torch, flask, ollama, picamera2; print('✅ Success!')"



# AI Camera and vision libraries
pip install --upgrade pip
pip install Pillow numpy opencv-python
pip install picamera2[libav]

# Object detection frameworks
pip install yolov5
# OR for MediaPipe Lite (lower resource usage):
pip install mediapipe
# OR for TensorFlow Lite:
pip install tensorflow-lite-runtime

# LLM and AI integration
pip install ollama requests

# Web server and communication
pip install flask flask-cors flask-socketio python-socketio
pip install websocket-client requests
```

---

## PART 3: OBJECT DETECTION SETUP

### 3.1 Using MediaPipe (Recommended - Lightweight)

Create file: `object_detector.py`

```python
import cv2
import mediapipe as mp
from picamera2 import Picamera2
import numpy as np

class ObjectDetector:
    def __init__(self):
        self.mp_object_detection = mp.solutions.object_detector
        self.detector = self.mp_object_detection.ObjectDetector(
            model_asset_path='efficientdet_lite0.tflite',
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )

    def detect_objects(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create image tensor
        image = mp.tasks.vision.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect objects
        detection_result = self.detector.detect(image)

        # Parse results
        objects = []
        for detection in detection_result.detections:
            object_data = {
                'label': detection.categories[0].category_name,
                'confidence': detection.categories[0].score,
                'bbox': {
                    'x': detection.bounding_box.origin_x,
                    'y': detection.bounding_box.origin_y,
                    'width': detection.bounding_box.width,
                    'height': detection.bounding_box.height
                }
            }
            objects.append(object_data)

        return objects
```

### 3.2 Camera Capture Module

Create file: `camera_capture.py`

```python
from picamera2 import Picamera2
import numpy as np
import threading
import time

class CameraCapture:
    def __init__(self, resolution=(640, 480)):
        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": resolution}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_loop(self):
        while self.running:
            frame = self.picam2.capture_array()
            with self.frame_lock:
                self.current_frame = frame
            time.sleep(0.033)  # 30 FPS

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self):
        self.running = False
        self.capture_thread.join()
        self.picam2.stop()
```

---

## PART 4: LLM INTEGRATION FOR COMMAND GENERATION

### 4.1 Local LLM Setup (Using Ollama)

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a lightweight model
ollama pull mistral

# Start Ollama service
ollama serve
```

### 4.2 Command Generation Module

Create file: `llm_command_generator.py`

```python
import requests
import json
import re
import time
from typing import List, Dict

class LLMCommandGenerator:
    def __init__(self, model_name="mistral", api_url="http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url

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
        }

    def generate_commands(self, detected_objects: List[Dict], 
                         context: str = "") -> List[str]:
        object_summary = self._format_objects(detected_objects)
        prompt = self._create_prompt(object_summary, context)
        response = self._call_llm(prompt)
        commands = self._parse_response(response)
        return commands

    def _format_objects(self, objects: List[Dict]) -> str:
        if not objects:
            return "No objects detected in the field."

        summary = "Detected objects:\n"
        for obj in objects:
            summary += f"  - {obj['label']} "
            summary += f"(confidence: {obj['confidence']:.2%}) "
            summary += f"at position {obj['bbox']['x']:.0f},{obj['bbox']['y']:.0f}\n"

        return summary

    def _create_prompt(self, object_summary: str, context: str) -> str:
        available_commands = ', '.join(self.robot_commands.keys())

        prompt = f"""You are a robot control AI. Based on detected objects,
generate robot commands. Keep responses brief.

Available commands: {available_commands}

Situation: {object_summary}

Task: {context if context else 'Explore and interact'}

Respond in JSON format ONLY:
{{"commands": ["cmd1", "cmd2"]}}"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"LLM error: {e}")
            return '{"commands": []}'

    def _parse_response(self, response: str) -> List[str]:
        try:
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                commands = data.get('commands', [])

                validated_commands = []
                for cmd in commands:
                    cmd_lower = cmd.lower().strip()
                    if cmd_lower in self.robot_commands:
                        validated_commands.append(
                            self.robot_commands[cmd_lower]
                        )

                return validated_commands
        except:
            pass

        return []
```

---

## PART 5: ROBOT COMMAND TRANSMISSION

### 5.1 Available Robot Commands

Based on index.html, your robot supports:

**Movement:** 
  - step("forward"), step("backward"), step("left"), step("right")

**Patterns:**
  - runPattern("square"), runPattern("circle"), runPattern("triangle")
  - runPattern("zigzag"), runPattern("spiral"), runPattern("dance")

**Control:**
  - step("stop"), speakText("message")

### 5.2 Command Executor

Create file: `robot_executor.py`

```python
import requests
import time

class RobotCommandExecutor:
    def __init__(self, robot_url="http://localhost:5000"):
        self.robot_url = robot_url
        self.last_command_time = 0
        self.min_interval = 0.2  # Min 200ms between commands

    def execute_command(self, command: str) -> bool:
        try:
            # Rate limiting
            elapsed = time.time() - self.last_command_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            response = requests.post(
                f"{self.robot_url}/api/command",
                json={"command": command},
                timeout=2
            )

            self.last_command_time = time.time()
            return response.status_code == 200

        except Exception as e:
            print(f"Command error: {e}")
            return False

    def execute_sequence(self, commands: list, delay: float = 0.5):
        for cmd in commands:
            self.execute_command(cmd)
            time.sleep(delay)
```

---

## PART 6: MAIN INTEGRATION SCRIPT

Create file: `main.py`

```python
#!/usr/bin/env python3
import argparse
import time
import signal
import sys

from camera_capture import CameraCapture
from object_detector import ObjectDetector
from llm_command_generator import LLMCommandGenerator
from robot_executor import RobotCommandExecutor

class AIRobotSystem:
    def __init__(self, robot_url="http://localhost:5000"):
        print("[*] Initializing AI Robot System...")

        print("[*] Starting camera...")
        self.camera = CameraCapture()

        print("[*] Loading object detector...")
        self.detector = ObjectDetector()

        print("[*] Initializing LLM...")
        self.llm = LLMCommandGenerator()

        print("[*] Connecting to robot...")
        self.robot = RobotCommandExecutor(robot_url)

        self.running = False
        self.task_context = "Explore the field"

    def set_task(self, task: str):
        self.task_context = task
        print(f"[*] Task: {task}")

    def process_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return None

        # Detect objects
        objects = self.detector.detect_objects(frame)
        print(f"[*] Detected {len(objects)} objects")

        for obj in objects:
            print(f"    - {obj['label']} ({obj['confidence']:.1%})")

        # Generate commands
        commands = self.llm.generate_commands(objects, self.task_context)
        print(f"[*] Generated {len(commands)} commands")

        return objects, commands

    def run(self, interval: float = 2.0):
        self.running = True
        print("[*] Starting. Press Ctrl+C to stop.\n")

        try:
            iteration = 0
            while self.running:
                iteration += 1
                print(f"[ITERATION {iteration}] {time.strftime('%H:%M:%S')}")

                result = self.process_frame()
                if result is None:
                    time.sleep(interval)
                    continue

                objects, commands = result

                if commands:
                    print(f"[*] Executing {len(commands)} commands")
                    self.robot.execute_sequence(commands, delay=0.5)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[*] Shutting down...")
            self.stop()

    def stop(self):
        self.running = False
        self.camera.stop()
        print("[*] System stopped")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Robot Control")
    parser.add_argument('--task', type=str, 
                       default="Explore and interact with objects",
                       help='Robot task')
    parser.add_argument('--robot-url', type=str, 
                       default="http://localhost:5000",
                       help='Robot URL')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Processing interval (seconds)')

    args = parser.parse_args()

    system = AIRobotSystem(robot_url=args.robot_url)
    system.set_task(args.task)
    system.run(interval=args.interval)
```

---

## PART 7: DEPLOYMENT

### Startup Sequence

Terminal 1: Start Ollama
```bash
ollama serve
```

Terminal 2: Start robot web interface
```bash
python3 flask_app.py
```

Terminal 3: Start AI system
```bash
python3 main.py --task "Find and approach balls"
```

### Testing

- Verify camera captures frames
- Check object detection in logs
- Monitor LLM command generation
- Observe robot executing commands
- Test with different task contexts

---

## PART 8: ADVANCED FEATURES

### Custom Decision Logic

```python
def smart_response(objects):
    commands = []

    # Find closest object
    closest = min(objects, key=lambda x: x['bbox']['x'])

    if closest['label'] == 'person':
        commands.append('speakText("Hello!")')
        commands.append('runPattern("dance")')
    elif closest['label'] == 'ball':
        commands.extend(['step("forward")', 'step("forward")'])

    return commands
```

### Performance Tuning

```python
# Reduce resolution
CameraCapture(resolution=(320, 240))

# Use lightweight models
detector = LiteObjectDetector()

# Batch processing
process_batch(frames, batch_size=4)

# Threading
threading.Thread(target=detection_loop).start()
threading.Thread(target=execution_loop).start()
```

---

## Summary

This complete system enables:

1. Real-time vision capture with Raspberry Pi AI Camera
2. Intelligent object detection using MediaPipe/YOLO
3. Smart command generation using local LLM (Ollama)
4. Autonomous robot control and execution
5. Flexible task-based operation
6. Web-based monitoring and manual override

Your robot can now intelligently perceive, reason, and act!
