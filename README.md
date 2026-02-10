# OmniAI - Raspberry Pi AI Robot Control System

AI-powered robot control using Raspberry Pi 5, IMX500 AI Camera with **YOLOv8 hardware-accelerated detection**, cloud LLM (Groq), and real-time object detection.

## Features

- **Hardware-Accelerated AI**: YOLOv8 runs directly on the IMX500 camera chip (~17ms inference, ~30fps)
- **Real-time Object Detection**: 80 COCO classes including people, animals, vehicles, household items
- **Cloud LLM Integration**: Groq-powered command generation (Llama 3.1 8B) - fast and free tier available
- **Local LLM Fallback**: Optional Ollama support (Mistral, Phi, TinyLlama)
- **Tomy Omnibot Control**: Audio frequency-based robot control
- **Web Dashboard**: Live MJPEG stream with detection overlays and controls
- **Thread-Safe Camera**: Robust multi-threaded capture with proper resource management

## Hardware Requirements

- **Raspberry Pi 5** (8GB+ RAM, 16GB recommended)
- **Raspberry Pi AI Camera** (IMX500 with neural accelerator)
- **Tomy Omnibot** (optional - for robot control via audio tones)

## Quick Start

### 1. Fresh Pi OS Install

```bash
# Update system
sudo apt update && sudo apt full-upgrade -y

# Install IMX500 AI Camera support
sudo apt install imx500-all -y

# Install system dependencies
sudo apt install -y libcap-dev python3-dev python3-venv libportaudio2 portaudio19-dev

# Reboot
sudo reboot
```

### 2. Install YOLOv8 Model

The YOLOv8 model isn't included in the default package - download it:

```bash
cd /usr/share/imx500-models
sudo wget https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_yolov8n_pp.rpk
```

### 3. Setup Project

```bash
# Clone/copy to Pi
cd ~/omniai

# Create virtual environment (IMPORTANT: --system-site-packages for picamera2 access)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install flask flask-cors flask-socketio requests ollama websocket-client python-socketio opencv-python sounddevice

# Set up Groq API key (recommended - fast cloud LLM)
echo "GROQ_API_KEY=your_api_key_here" > .env
# Get free API key at https://console.groq.com

# OR install Ollama for local LLM (optional)
curl https://ollama.ai/install.sh | sh
ollama pull mistral
```

### 4. Run

```bash
# Test detection standalone (recommended first)
python test_detection.py
# Open https://omniai.local:8080

# Full dashboard with LLM and robot control
python dashboard.py --port 8080
```

## IMX500 AI Camera

The IMX500 is a **smart camera** with an on-chip neural network accelerator. Key points:

- **Model Upload**: Neural network firmware is uploaded directly to the camera chip
- **Hardware Inference**: Detection runs on dedicated AI silicon, NOT on Pi's CPU
- **Performance**: ~17ms inference time, ~30fps real-time detection
- **YOLOv8**: Best accuracy for object detection (640x640 input, 80 COCO classes)

### How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                     IMX500 AI Camera                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │   Image     │───▶│  Neural Net │───▶│  Detection      │   │
│  │   Sensor    │    │  Processor  │    │  Results        │   │
│  └─────────────┘    └─────────────┘    └────────┬────────┘   │
└─────────────────────────────────────────────────┼────────────┘
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │  picamera2  │───▶│  Dashboard  │───▶│  Web Browser    │   │
│  │  (metadata) │    │  (Flask)    │    │  (MJPEG+WebSocket)│  │
│  └─────────────┘    └─────────────┘    └─────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
omniai/
├── dashboard.py              # Web dashboard with live stream + robot control
├── camera_capture.py         # Thread-safe camera capture with IMX500 support
├── object_detector.py        # Multi-backend detection (IMX500 YOLOv8 default)
├── llm_command_generator.py  # Cloud (Groq) + local (Ollama) LLM integration
├── robot_executor.py         # Robot command executor (audio tones + speech)
├── audio_commander.py        # Audio frequency generator + speech (thread-safe)
├── eye_display.py            # ST7735S TFT animated eye display
├── speak_pi.sh               # Text-to-speech script (Pi - espeak + pw-play)
├── speak_phrase.sh           # Pre-recorded phrase player (Pi)
├── speak.sh                  # Text-to-speech script (macOS - for testing)
├── audio_phrases/            # Pre-recorded WAV files for fast speech
│   ├── hello.wav, yes.wav, no.wav, thanks.wav, omnibot.wav
├── util/                     # Test scripts and utilities
│   ├── test_eye_display.py   # Eye display test
│   ├── test_detection.py     # Camera/detection test
│   ├── generate_certs.sh     # SSL certificate generator
│   └── start.sh              # Quick start script
├── .env                      # API keys (GROQ_API_KEY)
├── CLAUDE.md                 # Technical reference for Claude Code
└── README.md                 # This file
```

## Detection Models

| Model | File | Input | Speed | Accuracy |
|-------|------|-------|-------|----------|
| **YOLOv8 nano** | `imx500_network_yolov8n_pp.rpk` | 640x640 | Good | **Best** |
| MobileNet SSD | `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | 320x320 | **Fastest** | Good |
| NanoDet Plus | `imx500_network_nanodet_plus_416x416_pp.rpk` | 416x416 | Good | Good |
| EfficientDet | `imx500_network_efficientdet_lite0_pp.rpk` | Various | Medium | Good |

## Tomy Omnibot Audio Control

The robot is controlled via audio frequency tones:

| Command | Frequency |
|---------|-----------|
| Forward | 1614 Hz |
| Backward | 2013 Hz |
| Left | 2208 Hz |
| Right | 1811 Hz |
| Speaker On | 1422 Hz |
| Speaker Off | 4650 Hz |

Connect the Pi's audio output to the Omnibot's cassette input (via audio jack or Bluetooth).

## Speech Features

The robot can speak using text-to-speech or pre-recorded phrases.

### Pre-recorded Phrases (Fast)
Pre-generated WAV files for instant playback:
| Phrase | Description |
|--------|-------------|
| `hello` | "Hello" greeting |
| `yes` | "Yes" affirmative |
| `no` | "No" negative |
| `thanks` | "Thank you" |
| `omnibot` | "Hello, I am Omnibot" intro |

### Text-to-Speech (Flexible)
Any text can be spoken via espeak-ng (slower than pre-recorded).

### Speech Sequence
1. **Speaker On tone** (1422 Hz) - Enables robot's speaker relay
2. **Audio playback** - Speech or WAV file via Bluetooth
3. **Speaker Off tone** (4650 Hz) - Disables relay, stops audio

### Bluetooth Audio Setup
The Pi sends audio via Bluetooth to the robot's speaker:
```bash
# Pair Bluetooth speaker/robot
bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
```

## Eye Display (ST7735S TFT)

Animated robot eye on a 1.8" ST7735S TFT display (128x160 RGB) for personality.

### Wiring

| ST7735S Pin | Raspberry Pi |
|-------------|--------------|
| VCC | 3.3V |
| GND | GND |
| SCL | GPIO11 (SCLK) |
| SDA | GPIO10 (MOSI) |
| RES | GPIO25 |
| DC | GPIO24 |
| CS | GPIO8 (CE0) |
| BLK | 3.3V |

### Setup
```bash
# Enable SPI
sudo raspi-config  # Interface Options -> SPI -> Enable

# Install libraries
pip install st7735 gpiodevice

# Test display
python util/test_eye_display.py
```

### Display Configuration
- **Mode**: Portrait (128x160)
- **Rotation**: 0
- **Offsets**: left=2, top=1

### Expressions
| Expression | Description |
|------------|-------------|
| `normal` | Default relaxed eye |
| `happy` | Dilated pupil, curved smile |
| `surprised` | Wide eye, small pupil |
| `sleepy` | Half-closed eyelids |
| `angry` | Angled eyebrow |
| `look_left/right/up/down` | Pupil tracking |

### Testing Expressions

The eye reacts automatically to robot activity:

| Trigger | Expression | How to Test |
|---------|------------|-------------|
| Person detected | Happy | Stand in front of the camera |
| Cat/dog detected | Surprised | Show it a cat or dog (or a picture) |
| Left command | Look left | Press left button on dashboard |
| Right command | Look right | Press right button on dashboard |
| Forward command | Look up | Press forward button |
| Backward command | Look down | Press backward button |
| Speech command | Blink + Happy | Press "Hello" or other speech buttons |
| 30s inactivity | Sleepy | Wait with nothing happening |
| Random | Blink | Automatic every 3-7 seconds |

### Integration
```python
from eye_display import EyeDisplay, eye_happy, eye_surprised

eye = EyeDisplay()
eye.start()

eye.set_expression(EyeDisplay.EXPR_HAPPY)
eye.look_at(0.5, 0)  # Look right
eye.blink()
```

## Dashboard Features

### Main Dashboard (`/`)
- **Live Camera Stream**: MJPEG with detection bounding boxes
- **Object Detection Panel**: Real-time list of detected objects
- **Detection History**: Log of all detections with timestamps
- **Manual Robot Controls**: Movement buttons, patterns, speech
- **Speech Buttons**: Hello, Yes, No, Thanks (pre-recorded phrases)
- **Speaker Off Button**: 🔇 Kills speech and resets robot speaker
- **Statistics**: FPS, inference time, detection count
- **LLM Debug Panel**: Shows prompts and AI responses
- **Bluetooth Status**: Connection indicator

### Kids Dashboard (`/kids`)
- **Simplified Interface**: Large, colorful buttons for children
- **Mission Buttons**: Find Shoes, Find Person, Explore, Dance
- **Big Direction Controls**: Emoji arrows (⬆️ ⬇️ ⬅️ ➡️)
- **Say Hello**: Plays "Hello, I am Omnibot" greeting
- **Quiet Button**: 🔇 Speaker off for kids

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard UI |
| `/kids` | GET | Kid-friendly dashboard |
| `/stream` | GET | MJPEG video stream |
| `/api/detections` | GET | Current detections JSON |
| `/api/history` | GET | Detection history |
| `/api/start` | POST | Start AI system |
| `/api/stop` | POST | Stop AI system |
| `/api/command` | POST | Send robot command |
| `/api/task` | POST | Set LLM task context |
| `/api/bluetooth` | GET | Bluetooth connection status |

### Robot Commands via `/api/command`
```json
// Movement
{"command": "forward"}
{"command": "backward"}
{"command": "left"}
{"command": "right"}
{"command": "stop"}

// Patterns
{"command": "dance"}
{"command": "circle"}

// Speech (text-to-speech)
{"command": "speakText(\"Hello world\")"}

// Speech (pre-recorded - faster)
{"command": "phrase(\"hello\")"}

// Speaker control
{"command": "speaker_off"}
```

## LLM Options

### Cloud LLM (Recommended): Groq

Groq provides fast, free-tier access to Llama 3.1 8B - much faster than running locally on Pi.

```bash
# Set up API key
echo "GROQ_API_KEY=your_key_here" > .env
# Get free key at https://console.groq.com
```

| Model | Provider | Latency | Quality |
|-------|----------|---------|---------|
| `llama-3.1-8b-instant` | Groq | **~100ms** | **Best** |
| `mistral` | Ollama (local) | ~2-5s | Good |
| `phi` | Ollama (local) | ~1-3s | Good |
| `tinyllama` | Ollama (local) | ~500ms | Basic |

### Local LLM (Optional): Ollama

For offline operation or privacy:

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh
ollama pull mistral

# Use local LLM (set use_cloud=False in code)
python dashboard.py --llm-model mistral
```

## Troubleshooting

### YOLOv8 model not found
```bash
# Download it manually
cd /usr/share/imx500-models
sudo wget https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_yolov8n_pp.rpk
```

### Camera not working
```bash
rpicam-hello -t 5s
vcgencmd get_camera
```

### Detection not showing objects
1. Run `python test_detection.py` first to verify hardware works
2. Check confidence threshold (default 0.3 = 30%)
3. Ensure good lighting - AI cameras need decent light

### Slow performance
- YOLOv8 runs at ~30fps on IMX500 hardware
- If slower, check if running other processes
- Try MobileNet SSD for faster (but less accurate) detection

## Development

Primary development happens on local machine, deploy to Pi:

### Deploy via Git (recommended)
```bash
# From local machine: commit and push
git add . && git commit -m "your changes" && git push

# On the Pi: pull latest
ssh admin@omniai.local "cd /home/admin/omniai && git pull"
```

### Deploy via rsync (alternative)
```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' --exclude='*.pyc' \
    /path/to/omnibotAi/ admin@omniai.local:/home/admin/omniai/
```

### Starting the Dashboard
```bash
# On the Pi (works from any directory)
~/omniai/util/start.sh

# With options
~/omniai/util/start.sh --volume 0.7 --port 8080
```

## Future: Custom Model Training

Currently using pre-trained YOLOv8 with 80 COCO classes. The IMX500 ecosystem supports training custom models to recognize specific objects (e.g., "Mario's shoes" instead of generic "shoes").

See: [Streamline dataset creation for the Raspberry Pi AI Camera](https://www.raspberrypi.com/news/streamline-dataset-creation-for-the-raspberry-pi-ai-camera/)

### Available Tools

| Tool | Description | Link |
|------|-------------|------|
| **Brain Builder for AITRIOS** | No-code custom model training, needs only ~50 images | [Sony AITRIOS](https://info.aitrios.sony-semicon.com/developers-blog/build-custom-ai-models-for-raspberry-pi-ai-camera-zero-coding-required) |
| **Roboflow** | Dataset annotation/labeling, exports to YOLOv8 format | [roboflow.com](https://roboflow.com) |
| **imx500_zoo** | Command-line training with NanoDet | [Hackster Tutorial](https://www.hackster.io/541341/how-to-create-a-custom-object-detection-ai-model-pt-1-f7203e) |
| **Edge-MDT** | Model conversion and quantization toolkit | [Sony Developer Docs](https://www.aitrios.sony-semicon.com/edge-ai-devices/raspberry-pi-ai-camera) |

### Custom Training Pipeline

```
1. Capture images → 2. Annotate (Roboflow) → 3. Train (Brain Builder/imx500_zoo)
                                                         ↓
4. Deploy RPK ← 5. Package (imx500-package) ← 6. Quantize (Edge-MDT)
```

### Model Types Beyond Detection

| Type | Use Case |
|------|----------|
| **Classifier** | "Is this a cat or dog?" - single class output |
| **Detector** | "Where are all the people?" - bounding boxes |
| **Anomaly Recognizer** | "Is something wrong?" - defect detection |
| **Segmentation** | Pixel-level object boundaries |

### Model Caching

The IMX500 has **16MB flash** for caching multiple models. With 2-3MB models, you can store 4-5 models and switch between them without re-uploading.

### Potential Use Cases

- Train on specific household items for better "find my X" missions
- Recognize family members by face
- Detect specific pets
- Custom gesture recognition

## Resources

- [IMX500 Documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [IMX500 Model Zoo](https://github.com/raspberrypi/imx500-models)
- [Picamera2 Examples](https://github.com/raspberrypi/picamera2/tree/main/examples/imx500)
- [Ollama](https://ollama.ai/)
- [Brain Builder for AITRIOS](https://info.aitrios.sony-semicon.com/developers-blog/build-custom-ai-models-for-raspberry-pi-ai-camera-zero-coding-required)
- [Custom Model Tutorial](https://www.hackster.io/541341/how-to-create-a-custom-object-detection-ai-model-pt-1-f7203e)

## License

MIT
