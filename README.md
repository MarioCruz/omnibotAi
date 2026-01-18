# OmniAI - Raspberry Pi AI Robot Control System

AI-powered robot control using Raspberry Pi 5, IMX500 AI Camera, local LLM (Ollama), and real-time object detection.

## Hardware Requirements

- **Raspberry Pi 5** (16GB RAM recommended)
- **Raspberry Pi AI Camera** (IMX500 with neural accelerator)
- **Robot Platform** with motor control (optional for dashboard testing)

## Fresh Pi OS Install

Run these commands on a fresh Raspberry Pi OS installation:

```bash
# 1. Update system
sudo apt update && sudo apt full-upgrade -y

# 2. Install IMX500 AI Camera support
sudo apt install imx500-all -y

# 3. Install system dependencies (including PortAudio for audio tones)
sudo apt install -y libcap-dev python3-dev python3-venv libportaudio2 portaudio19-dev

# 4. Reboot to apply changes
sudo reboot
```

After reboot, verify the camera works:
```bash
rpicam-hello -t 5s
```

## Quick Start

```bash
# 1. Clone/copy to Pi
cd ~/omniai

# 2. Create virtual environment (IMPORTANT: use --system-site-packages for Pi camera access)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 3. Install dependencies
pip install flask flask-cors flask-socketio requests ollama websocket-client python-socketio opencv-python

# 4. Install Ollama (for LLM commands)
curl https://ollama.ai/install.sh | sh
ollama pull mistral
# Ollama starts automatically as a service

# 5. Run the dashboard (uses IMX500 AI Camera for detection)
python dashboard.py --port 8080 --no-ssl --detector imx500
```

> **Note:** The `--system-site-packages` flag is required because `libcamera` and `picamera2` are system packages that can't be installed via pip. This lets the venv access them.

Open browser: `https://<pi-ip>:8080`

## Applications

### 1. Dashboard (Recommended)
Full web interface with live camera, detection overlays, and controls.

```bash
python dashboard.py --port 8080
```

Features:
- Live MJPEG stream with detection boxes
- Start/Pause/Stop AI system
- Set task context for LLM
- Manual robot controls
- Real-time statistics
- Activity log

### 2. Simple Stream Server
Basic HTTPS camera stream without AI processing.

```bash
python app.py
# or
./start.sh
```

### 3. Headless Robot Control
Command-line robot control without web interface.

```bash
python robot_control.py --task "Find and greet people"
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| `--task` | Task for LLM context | "Explore and interact" |
| `--robot-url` | Robot API endpoint | http://localhost:5000 |
| `--detector` | imx500, yolov5, mediapipe | imx500 |
| `--llm-model` | Ollama model name | mistral |
| `--no-llm` | Use rule-based commands | false |
| `--interval` | Seconds between cycles | 2.0 |

## Project Structure

```
omniai/
├── dashboard.py          # Web dashboard with live stream
├── app.py                # Simple HTTPS stream server
├── robot_control.py      # Headless control system
├── camera_capture.py     # Threaded Pi camera capture (atomic frame+metadata)
├── object_detector.py    # Multi-backend detection (IMX500, YOLOv5, MediaPipe)
├── llm_command_generator.py  # Ollama LLM integration (configurable resolution)
├── robot_executor.py     # Robot command executor (audio tones for Omnibot)
├── audio_commander.py    # Audio frequency generator for Tomy Omnibot
├── requirements.txt      # Python dependencies
├── generate_certs.sh     # SSL certificate generator
├── start.sh              # Quick start script
└── research/             # Reference documentation
```

## Module Configuration

### LLM Command Generator

The `LLMCommandGenerator` adapts to different camera resolutions:

```python
from llm_command_generator import LLMCommandGenerator

# Default 640x480
llm = LLMCommandGenerator(model_name='mistral')

# Custom resolution (e.g., 1920x1080)
llm = LLMCommandGenerator(
    model_name='mistral',
    frame_width=1920,
    frame_height=1080
)
```

The frame dimensions are used to:
- Calculate center point for left/right object positioning
- Set directional thresholds (15% of frame width) for turn commands
- Describe object positions in natural language (left/center/right, top/middle/bottom)

### Audio Commander

The `AudioCommander` generates audio tones for Tomy Omnibot control:

```python
from audio_commander import AudioCommander

commander = AudioCommander(volume=0.5)
commander.forward(500)     # Play 1614 Hz for 500ms
commander.speak("Hello!")  # Text-to-speech via espeak
```

**Security**: Text input to `speak()` is sanitized (alphanumeric + basic punctuation only) to prevent command injection.

## Endpoints

| URL | Description |
|-----|-------------|
| `https://<ip>:8080/` | Dashboard UI |
| `https://<ip>:8080/stream` | MJPEG stream with detections |
| `https://<ip>:8080/api/start` | Start AI system (POST) |
| `https://<ip>:8080/api/stop` | Stop AI system (POST) |
| `https://<ip>:8080/api/task` | Set task context (POST) |
| `https://<ip>:8080/api/command` | Send manual command (POST) |
| `https://<ip>:8080/api/status` | Get system status (GET) |

## Robot Commands

**Movement:**
- `step("forward")`, `step("backward")`, `step("left")`, `step("right")`, `step("stop")`

**Patterns:**
- `runPattern("dance")`, `runPattern("circle")`, `runPattern("square")`
- `runPattern("triangle")`, `runPattern("zigzag")`, `runPattern("spiral")`
- `runPattern("search")`, `runPattern("patrol")`

**Other:**
- `speakText("Hello!")`

## Tomy Omnibot Audio Control

This system controls the Tomy Omnibot using audio frequency tones.

### Omnibot Frequencies

| Command | Frequency |
|---------|-----------|
| Forward | 1614 Hz |
| Backward | 2013 Hz |
| Left | 2208 Hz |
| Right | 1811 Hz |
| Speaker On | 1422 Hz |
| Speaker Off | 4650 Hz |

### To Run

First, install the audio library:
```bash
pip install sounddevice
```

Then run the dashboard:
```bash
python dashboard.py --port 8080 --no-ssl --volume 0.5
```

The dashboard now has:
- Movement buttons that send audio tones
- Pattern buttons (dance, circle, square, spiral)
- Speech section with text input and quick phrases
- Stop button to halt movement

Make sure your Pi's audio output is connected to the Omnibot (via audio jack or Bluetooth).

You can test audio with:
```bash
speaker-test -t sine -f 1614 -l 1
```

## LLM Models for Pi 5

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `mistral` | 4.1GB | Good | Best |
| `phi` | 1.6GB | Faster | Good |
| `tinyllama` | 637MB | Fastest | Basic |

```bash
# Pull alternative model
ollama pull phi

# Use with dashboard
python dashboard.py --llm-model phi
```

## Troubleshooting

### Camera not working
```bash
# Test camera
rpicam-hello -t 5s

# Check camera is detected
vcgencmd get_camera
```

### Ollama not responding
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart service
ollama serve
```

### SSL certificate warning
Self-signed certificates trigger browser warnings. Click "Advanced" → "Proceed" to accept.

### Detection not working
```bash
# Test detector standalone
python object_detector.py
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Pi Camera     │────▶│  Object Detector │────▶│  LLM Command    │
│   (IMX500)      │     │  (YOLOv5/MP)     │     │  Generator      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│   Web Dashboard │◀────│  Flask + Socket  │◀─────────────┤
│   (Browser)     │     │  Server          │              │
└─────────────────┘     └──────────────────┘              │
                                                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   Robot          │◀────│  Command        │
                        │   Platform       │     │  Executor       │
                        └──────────────────┘     └─────────────────┘
```

## License

MIT
