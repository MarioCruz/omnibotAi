# OmniAI Project Reference

Technical notes and commands for Claude Code context.

## Tomy Omnibot Audio Control

The robot is controlled via audio frequency tones sent through speaker/Bluetooth.

### Command Frequencies (Hz)
| Command | Frequency |
|---------|-----------|
| Forward | 1614 Hz |
| Backward | 2013 Hz |
| Left | 2208 Hz |
| Right | 1811 Hz |
| Speaker On | 1422 Hz |
| Speaker Off | 4650 Hz |

### Default Durations
- Step (forward/backward): 500ms
- Turn (90 degrees): 3000ms

## Working Camera Command (IMX500)

```bash
rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080 --framerate 30
```

## Raspberry Pi 5 Setup

### Fresh Install
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install imx500-all -y
sudo apt install -y libcap-dev python3-dev python3-venv
sudo reboot
```

### Python Environment (IMPORTANT: use --system-site-packages)
```bash
cd ~/omniai
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install flask flask-cors flask-socketio requests ollama websocket-client python-socketio opencv-python sounddevice
```

### Enable Camera & SPI
```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
# Interface Options -> SPI -> Enable (for IMX500 neural accelerator)
sudo reboot
```

## Ollama Setup (Local LLM)

```bash
# Install
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull mistral

# Test
ollama run mistral "Say hello"

# Check status
curl http://localhost:11434/api/tags
```

### Recommended Models for Pi 5 (16GB)
- `mistral` - 4.1GB, best quality
- `phi` - 1.6GB, faster
- `tinyllama` - 637MB, fastest

## Available Robot Commands

### Movement (audio tones)
```python
forward    # 1614 Hz
backward   # 2013 Hz
left       # 2208 Hz
right      # 1811 Hz
stop       # Stops tone
```

### Patterns (sequences)
```python
dance, circle, square, triangle, zigzag, spiral, search, patrol
```

### Speech
```python
speakText("Hello!")  # Uses espeak + speaker on/off tones
```

## IMX500 AI Camera - Picamera2 Integration

### Picamera2 Python Usage
```python
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection

# Load model
imx500 = IMX500('/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk')

# Configure camera (standard config, IMX500 handles model separately)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

# IMPORTANT: Capture frame and metadata atomically using capture_request()
job = picam2.capture_request()
frame = job.make_array("main")
metadata = job.get_metadata()
job.release()  # Always release the request

# Get inference results from metadata
np_outputs = imx500.get_outputs(metadata)
input_w, input_h = imx500.get_input_size()
```

### Important API Notes
- **DO NOT use `imx500.configure(picam2, config)`** - this method doesn't exist
- Configure camera normally with `picam2.configure(config)`
- Use `capture_request()` to get frame and metadata atomically (avoids race condition)
- Always call `job.release()` after extracting data

### Available IMX500 Models (install: `sudo apt install imx500-models`)

**Object Detection:**
| Model | Path |
|-------|------|
| SSD MobileNetV2 (default) | `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` |
| YOLOv8n | `imx500_network_yolov8n_pp.rpk` |
| YOLO11n | `imx500_network_yolo11n_pp.rpk` |
| EfficientDet Lite-0 | `imx500_network_efficientdet_lite0_pp.rpk` |
| NanoDet Plus | `imx500_network_nanodet_plus_416x416_pp.rpk` |

**Classification:**
| Model | Path |
|-------|------|
| EfficientNet-B0 | `imx500_network_efficientnet_bo.rpk` |
| MobileNetV2 | `imx500_network_mobilenet_v2.rpk` |
| ResNet-18 | `imx500_network_resnet18.rpk` |

**Pose Estimation:**
| Model | Path |
|-------|------|
| HigherHRNet | `imx500_network_higherhrnet_coco.rpk` |

**Segmentation:**
| Model | Path |
|-------|------|
| DeepLabv3Plus | `imx500_network_deeplabv3plus.rpk` |

### Demo Commands (from picamera2 examples)
```bash
# Object detection
python imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk

# YOLOv8
python imx500_object_detection_demo.py --model /usr/share/imx500-models/imx500_network_yolov8n_pp.rpk --bbox-normalization --bbox-order xy

# Classification
python imx500_classification_demo.py --model /usr/share/imx500-models/imx500_network_efficientnet_bo.rpk

# Pose estimation
python imx500_pose_estimation_higherhrnet_demo.py --model /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk

# Segmentation
python imx500_segmentation_demo.py --model /usr/share/imx500-models/imx500_network_deeplabv3plus.rpk
```

### IMX500 Resources
- Official Docs: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Model Zoo: https://github.com/raspberrypi/imx500-models
- Picamera2 Examples: https://github.com/raspberrypi/picamera2/tree/main/examples/imx500

### COCO Classes (MobileNet SSD)
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, 
traffic light, fire hydrant, stop sign, parking meter, bench, bird, 
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich,
orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
potted plant, bed, dining table, toilet, tv, laptop, mouse, remote,
keyboard, cell phone, microwave, oven, toaster, sink, refrigerator,
book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Useful Debug Commands

```bash
# Test camera
rpicam-hello -t 5s
vcgencmd get_camera

# Check Pi temperature
vcgencmd measure_temp

# Network info
hostname -I

# Test Ollama
curl http://localhost:11434/api/tags

# Test audio
speaker-test -t sine -f 1000 -l 1

# Test modules
python audio_commander.py
python robot_executor.py
```

## Port Reference
- `8080` - Dashboard (dashboard.py)
- `11434` - Ollama API

## Module API Reference

### LLMCommandGenerator
```python
from llm_command_generator import LLMCommandGenerator

# Supports configurable frame dimensions (default 640x480)
llm = LLMCommandGenerator(
    model_name='mistral',
    frame_width=640,   # Used for left/right positioning
    frame_height=480   # Used for top/bottom positioning
)

# Frame dimensions affect:
# - Center calculation for directional commands
# - Threshold = 15% of frame_width for turn triggers
# - Position descriptions (left/center/right at 1/3 intervals)
```

### RobotCommandExecutor
```python
from robot_executor import RobotCommandExecutor

robot = RobotCommandExecutor(volume=0.5)  # Volume clamped to 0.0-1.0
robot.connect()

# Patterns can be cancelled with stop command
robot.execute('dance')   # Starts pattern
robot.execute('stop')    # Cancels mid-pattern
```

### Dashboard Shutdown
The dashboard handles graceful shutdown on SIGINT/SIGTERM:
- Sets `system_state['shutdown'] = True`
- Stops processing loop
- Disconnects robot
- Stops camera capture

### AudioCommander
```python
from audio_commander import AudioCommander

commander = AudioCommander(volume=0.5)

# Movement tones (returns True if successful)
commander.forward(500)   # 1614 Hz for 500ms
commander.backward(500)  # 2013 Hz
commander.left(500)      # 2208 Hz
commander.right(500)     # 1811 Hz

# Speech (text is sanitized to prevent command injection)
commander.speak("Hello!")  # Only alphanumeric, spaces, basic punctuation allowed

# Check state
commander.is_playing  # True if audio currently playing
commander.stop()      # Stop any playing audio
```

### Security Notes
- `speak()` sanitizes input with regex: `[^a-zA-Z0-9\s.,!?'-]` removed
- `espeak` called with `--` to prevent option injection
- `sd.wait()` has timeout (duration + 1 second) to prevent hangs
- API endpoints validate `request.json` (handles None case)
- Camera metadata copied by value (not reference) for thread safety

## Development Workflow

**Primary development happens on local GitHub repo** to avoid Pi SD card corruption issues.

### Push to Pi (deploy changes)
When the user says "push" or "deploy", copy files FROM local TO Pi:

```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pem' --exclude='.DS_Store' --exclude='._*' --exclude='models/' --exclude='.git/' --exclude='*.log' /Users/mariocruz/Documents/GitHub/omnibotAi/ admin@omniai.local:/home/admin/omniai/
```

### Sync from Pi (backup)
When the user says "sync" or "synch", run the backup sync script to copy files from Pi to local GitHub repo:

```bash
/Users/mariocruz/Documents/GitHub/omnibotAi/sync_from_pi.sh
```

This syncs from `admin@omniai.local:/home/admin/omniai/` to `/Users/mariocruz/Documents/GitHub/omnibotAi/` and auto-commits changes.

## Audio Setup

Make sure audio output is configured:
```bash
# Check audio devices
aplay -l

# Set default output (HDMI or headphone jack)
sudo raspi-config
# System Options -> Audio -> Choose output
```

For Bluetooth speaker:
```bash
# Pair via GUI or:
bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
```


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Jan 18, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #936 | 9:24 AM | ✅ | Dashboard startup banner updated with Tomy Omnibot branding and audio configuration info | ~361 |
| #935 | 9:23 AM | 🔄 | Dashboard refactored for audio-based robot control with volume parameter | ~354 |
| #923 | 9:11 AM | 🔵 | Complete AI robot control system with five integrated components discovered | ~708 |
| #905 | 8:36 AM | 🔵 | Initial project state discovered - empty directory with camera command reference | ~310 |
</claude-mem-context>