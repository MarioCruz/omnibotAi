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
- Turn (course correction): 750ms

## Raspberry Pi 5 Setup

### Fresh Install
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install imx500-all -y
sudo apt install -y libcap-dev python3-dev python3-venv libportaudio2 portaudio19-dev
sudo reboot
```

### Python Environment (IMPORTANT: use --system-site-packages)
```bash
cd ~/omniai
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install flask flask-cors flask-socketio requests ollama websocket-client python-socketio opencv-python sounddevice st7735 gpiodevice
```

### Enable Camera & SPI
```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
# Interface Options -> SPI -> Enable (for IMX500 and ST7735S display)
sudo reboot
```

## IMX500 AI Camera - Official picamera2 API

The IMX500 is a smart camera with an on-chip neural network accelerator. Uses YOLO11 nano (upgraded from YOLOv8). The model firmware is uploaded directly to the camera chip, and inference runs on dedicated hardware - NOT on the Pi's CPU.

### Key Concepts

1. **Model Upload**: When you create an `IMX500(model_path)` instance, the neural network firmware is uploaded to the camera's AI chip
2. **Hardware Inference**: Detection runs at ~30fps on-chip with ~17ms inference time
3. **NetworkIntrinsics**: Model metadata (task type, inference rate, bbox format) embedded in the .rpk file
4. **Coordinate Conversion**: Use `imx500.convert_inference_coords()` for accurate bounding box mapping

### Official Python Usage (YOLO11)

```python
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# Initialize IMX500 with model (uploads firmware to camera chip)
model_path = '/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk'
imx500 = IMX500(model_path)

# Get model intrinsics (inference rate, bbox format, etc.)
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"
intrinsics.labels = COCO_LABELS  # Set your labels
intrinsics.update_with_defaults()

# Configure camera using IMX500's camera number
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)},
    controls={"FrameRate": intrinsics.inference_rate},
    buffer_count=12  # Larger buffer for smooth inference
)
picam2.configure(config)

# Show model loading progress bar
imx500.show_network_fw_progress_bar()

picam2.start()

# Detection loop
while True:
    # Get metadata containing inference results
    metadata = picam2.capture_metadata()

    # Get raw outputs with batch dimension
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        continue

    # Parse outputs (boxes, scores, classes)
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

    # Handle bbox format based on intrinsics
    if intrinsics.bbox_normalization:
        input_w, input_h = imx500.get_input_size()
        boxes = boxes / input_h

    if intrinsics.bbox_order == "xy":  # YOLO uses xy order
        boxes = boxes[:, [1, 0, 3, 2]]

    # Convert to screen coordinates
    for box, score, cls in zip(boxes, scores, classes):
        if score > 0.3:
            x, y, w, h = imx500.convert_inference_coords(box, metadata, picam2)
            print(f"Detected {LABELS[int(cls)]} at ({x},{y}) conf={score:.0%}")

    # Get frame for display
    frame = picam2.capture_array("main")
```

### Key API Methods

| Method | Description |
|--------|-------------|
| `IMX500(model_path)` | Create instance, uploads firmware to camera |
| `imx500.camera_num` | Camera index to use with Picamera2() |
| `imx500.network_intrinsics` | Get NetworkIntrinsics (bbox_order, inference_rate, etc.) |
| `imx500.show_network_fw_progress_bar()` | Show firmware upload progress |
| `imx500.get_outputs(metadata, add_batch=True)` | Get inference outputs from metadata |
| `imx500.get_input_size()` | Get model input dimensions (e.g., 640x640) |
| `imx500.convert_inference_coords(box, metadata, picam2)` | Convert bbox to screen coords |
| `imx500.get_kpi_info(metadata)` | Get inference timing (DNN runtime in ms) |

### NetworkIntrinsics Properties

| Property | Description | Example |
|----------|-------------|---------|
| `task` | Model task type | "object detection" |
| `inference_rate` | Optimal FPS | 30 |
| `bbox_normalization` | Boxes need /input_size | True |
| `bbox_order` | Coordinate order | "xy" (YOLO) or "yx" (SSD) |
| `postprocess` | Special postprocessing | "nanodet" |
| `labels` | Class label list | [...] |

### Available IMX500 Models

**Object Detection (YOLO-family only — the bbox parser assumes xy ordering):**
| Model | File | Notes |
|-------|------|-------|
| **YOLO11 nano** | `imx500_network_yolo11n_pp.rpk` | Best accuracy, 640x640 input |
| YOLOv8 nano | `imx500_network_yolov8n_pp.rpk` | Previous default, 640x640 input |

SSD MobileNet / NanoDet / EfficientDet (yx-ordered) are NOT supported.
`_detect_imx500()` hardcodes xy ordering and would mis-project their boxes.

**YOLO11 is included in the imx500-models package (no manual download needed).**

**Classification:**
- `imx500_network_efficientnet_bo.rpk`
- `imx500_network_mobilenet_v2.rpk`
- `imx500_network_resnet18.rpk`

**Pose Estimation:**
- `imx500_network_higherhrnet_coco.rpk`

**Segmentation:**
- `imx500_network_deeplabv3plus.rpk`

### Running Official picamera2 Demo

```bash
# Clone picamera2 examples
git clone https://github.com/raspberrypi/picamera2.git
cd picamera2/examples/imx500

# Run YOLO11 detection
python imx500_object_detection_demo.py \
    --model /usr/share/imx500-models/imx500_network_yolo11n_pp.rpk \
    --ignore-dash-labels -r
```

### COCO Classes (80 classes)
```
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
```

## Navigation (Rule-Based)

Navigation uses rule-based position math instead of LLM. The LLM approach was tested
(Groq Llama 3.1 8B and 3.3 70B) but both always returned "right" regardless of actual
object position and added 15s latency per decision. Rule-based is instant and correct.

### How It Works
1. Pick highest-confidence target from detections
2. If target is LEFT of center → turn left
3. If target is RIGHT of center → turn right
4. If target is CENTERED → move forward
5. If target fills >60% of frame → stop (close enough)

### NavigationEngine
```python
from navigation import NavigationEngine

nav = NavigationEngine(frame_width=640, frame_height=480)
commands = nav.generate_commands(detections, context="Find the person")
# Returns: ['step("forward")'] or ['step("left")'] etc.
```

### Task Logging
Navigation decisions are logged to `logs/task.log`:
```
NAV target=person (73%) pos=x:197 cx:345 frame_cx:320 commands=['step("forward")'] | person CENTERED -> forward
NAV target=person (78%) pos=x:0 cx:140 frame_cx:320 commands=['step("left")'] | person LEFT -> turn left
NAV target=person (73%) pos=x:140 cx:346 frame_cx:320 commands=['step("stop")'] | person fills 64% -> STOP
```

## LLM Setup (Optional - for future "describe scene" feature)

Groq API key can be set for potential future features (scene description, voice interaction):

```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
# Get free API key at https://console.groq.com
```

The `llm_command_generator.py` module is kept in the repo but is not used for navigation.

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

### Speech Commands
```python
speakText("Hello!")   # Text-to-speech via espeak (slower, any text)
phrase("hello")       # Pre-recorded phrase (faster, limited options)
speaker_off           # Kill speech and reset robot speaker relay
```

### Pre-recorded Phrases
Located in `audio_phrases/` directory. Much faster than text-to-speech:
| Phrase | File | Use Case |
|--------|------|----------|
| `hello` | hello.wav | Greeting |
| `yes` | yes.wav | Affirmative response |
| `no` | no.wav | Negative response |
| `thanks` | thanks.wav | Gratitude |
| `omnibot` | omnibot.wav | "Hello, I am Omnibot" intro |

### Speech Scripts (Raspberry Pi)
- `speak_pi.sh` - Text-to-speech with speaker on/off tones (espeak-ng + sox + pw-play)
- `speak_phrase.sh` - Play pre-recorded WAV with speaker on/off tones

## Bluetooth Audio Setup (PipeWire)

The robot receives audio via Bluetooth. PipeWire routes audio to the paired speaker.

### Pairing Bluetooth Speaker
```bash
bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
> exit
```

### Verify Audio Routing
```bash
# Check PipeWire sinks
pactl list sinks short

# Check Bluetooth device
pactl list sinks | grep -A5 bluez

# Test audio output
sox -n -t wav - synth 1 sine 1000 | pw-play -
```

### Audio Path for Speech
```
Text → espeak-ng --stdout → pw-play → PipeWire → Bluetooth → Robot Speaker
```

### Audio Path for Tones
```
sox (generate sine) → pw-play → PipeWire → Bluetooth → Robot Speaker
```

## Useful Debug Commands

```bash
# Test camera
rpicam-hello -t 5s
vcgencmd get_camera

# Test IMX500 with YOLO11
python test_detection.py  # Opens https://omniai.local:8080

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
- `8080` - Dashboard (dashboard.py) / Test detection (test_detection.py)

## Dashboard Features

### Main Dashboard (`/`)
- Live MJPEG stream with detection bounding boxes
- Manual robot controls (forward, back, left, right, stop)
- Pattern buttons (dance, circle, square, etc.)
- Speech buttons (Hello, Yes, No, Thanks) - uses pre-recorded phrases
- Speaker Off button (🔇) - kills speech and resets robot
- Detection history panel
- Navigation log (real-time target/action/reason per decision cycle)
- Statistics (iterations, FPS, detections, commands)
- Bluetooth status indicator

### Kids Dashboard (`/kids`)
- Simplified, colorful interface for children
- Large mission buttons (Find Shoes, Find Person, Explore, etc.)
- Big directional controls with emoji arrows
- "Say Hello" button - plays pre-recorded omnibot greeting
- "Quiet" button (🔇) - speaker off for kids

### Dashboard API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/kids` | GET | Kid-friendly dashboard |
| `/stream` | GET | MJPEG video stream |
| `/api/command` | POST | Send robot command |
| `/api/start` | POST | Start AI system |
| `/api/stop` | POST | Stop AI system |
| `/api/bluetooth` | GET | Bluetooth connection status |

## Module API Reference

### ObjectDetector (IMX500 YOLO11)
```python
from object_detector import ObjectDetector

# Uses YOLO11 by default for best accuracy
detector = ObjectDetector(backend='imx500')

# Or pin a specific YOLO variant (YOLO family only — see supported models above)
detector = ObjectDetector(
    backend='imx500',
    model_path='/usr/share/imx500-models/imx500_network_yolov8n_pp.rpk',
    confidence_threshold=0.3
)

# Pass picam2 for accurate coordinate conversion
detector.set_picam2(camera.picam2)

# Detection loop
detector.set_metadata(metadata)  # From picam2.capture_metadata()
detections = detector.detect(frame)

# Access IMX500 instance and intrinsics
imx500 = detector.get_imx500()
intrinsics = detector.get_intrinsics()
```

### CameraCapture (IMX500 Support)
```python
from camera_capture import CameraCapture

# With IMX500 hardware acceleration
imx500 = detector.get_imx500()
intrinsics = detector.get_intrinsics()
camera = CameraCapture(
    resolution=(640, 480),
    framerate=30,
    imx500=imx500,       # Enables hardware acceleration
    intrinsics=intrinsics # Uses model's inference rate
)

# Get frame and metadata atomically
frame, metadata = camera.get_frame_and_metadata()
```

### NavigationEngine (replaces LLM for movement)
```python
from navigation import NavigationEngine

nav = NavigationEngine(frame_width=640, frame_height=480)
commands = nav.generate_commands(detections, context="Find the person")
# Returns: ['step("forward")'], ['step("left")'], ['step("stop")'], etc.
# nav.last_debug has target, position, response for logging
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

### AudioCommander
```python
from audio_commander import AudioCommander

commander = AudioCommander(volume=0.5)

# Movement tones (generated sine waves via sox + pw-play)
commander.forward(500)   # 1614 Hz for 500ms
commander.backward(500)  # 2013 Hz
commander.left(500)      # 2208 Hz
commander.right(500)     # 1811 Hz
commander.speaker_on()   # 1422 Hz - enable robot speaker relay
commander.speaker_off()  # 4650 Hz - disable robot speaker relay

# Text-to-speech (sanitized, calls speak_pi.sh)
commander.speak("Hello!")  # Slower, supports any text

# Pre-recorded phrases (faster, calls speak_phrase.sh)
commander.speak_phrase("hello")   # hello.wav
commander.speak_phrase("yes")     # yes.wav
commander.speak_phrase("no")      # no.wav
commander.speak_phrase("thanks")  # thanks.wav
commander.speak_phrase("omnibot") # omnibot.wav

# Stop speech and reset robot state
commander.stop()           # Stops tones
commander.stop_speaking()  # Kills speech processes + sends speaker_off
```

### AudioCommander Thread Safety
Speech processes are managed with thread-safe locks:
```python
# Thread-safe process tracking
self._proc_lock = threading.Lock()  # Protects process references
self._espeak_proc = None            # Current speech process
self._stopping = False              # Flag to abort speech

# stop_speaking() safely:
# 1. Sets _stopping flag
# 2. Acquires lock, copies refs, clears refs
# 3. Kills processes outside lock (avoids deadlock)
# 4. Sends speaker_off tone via sox | pw-play
```

### EyeDisplay (ST7735S TFT)
Animated robot eye on 1.8" ST7735S TFT (128x160 RGB) connected via SPI.

```python
from eye_display import EyeDisplay

eye = EyeDisplay(dc_pin=24, rst_pin=25, cs_pin=0)
eye.start()

# Set expressions
eye.set_expression(EyeDisplay.EXPR_HAPPY)
eye.set_expression(EyeDisplay.EXPR_SURPRISED)
eye.set_expression(EyeDisplay.EXPR_SLEEPY)
eye.set_expression(EyeDisplay.EXPR_ANGRY)
eye.set_expression(EyeDisplay.EXPR_LOOKING_LEFT)
eye.set_expression(EyeDisplay.EXPR_LOOKING_RIGHT)

# Pupil tracking (-1.0 to 1.0 range)
eye.look_at(0.5, 0)   # Look right
eye.look_at(-0.5, 0)  # Look left
eye.look_at(0, -0.5)  # Look up

# Manual blink
eye.blink()

# Stop animation
eye.stop()
```

### ST7735S Wiring
| ST7735S Pin | Raspberry Pi GPIO |
|-------------|-------------------|
| VCC | 3.3V |
| GND | GND |
| SCL (SCLK) | GPIO11 (SPI0 SCLK) |
| SDA (MOSI) | GPIO10 (SPI0 MOSI) |
| RES (RST) | GPIO25 |
| DC | GPIO24 |
| CS | GPIO8 (SPI0 CE0) |
| BLK | 3.3V (backlight always on) |

### Eye Expressions
| Constant | Description |
|----------|-------------|
| `EXPR_NORMAL` | Default relaxed eye |
| `EXPR_HAPPY` | Dilated pupil, curved line below |
| `EXPR_SURPRISED` | Wide eye, small pupil |
| `EXPR_SLEEPY` | Half-closed eyelids |
| `EXPR_ANGRY` | Angled eyebrow overlay |
| `EXPR_LOOKING_LEFT` | Pupil offset left |
| `EXPR_LOOKING_RIGHT` | Pupil offset right |
| `EXPR_LOOKING_UP` | Pupil offset up |
| `EXPR_LOOKING_DOWN` | Pupil offset down |
| `EXPR_BLINK` | Fully closed |

### Display Configuration
```python
# ST7735S 1.8" TFT working settings:
width=128, height=160   # Portrait mode
rotation=0
offset_left=2, offset_top=1
invert=False
```

### Testing Eye Display
```bash
# Install libraries
pip install st7735 gpiodevice

# Run test (cycles through all expressions)
python util/test_eye_display.py
```

## Development Workflow

**Primary development happens on local GitHub repo** to avoid Pi SD card corruption issues.

### Deploy via Git (recommended)
```bash
# Push from local
git push origin main

# Pull on Pi
ssh admin@omniai.local "cd /home/admin/omniai && git pull"
```

The Pi has `gh` CLI authenticated and tracks `origin/main` via HTTPS.

### Deploy via rsync (alternative)
```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pem' --exclude='.DS_Store' --exclude='._*' --exclude='models/' --exclude='.git/' --exclude='*.log' /Users/mariocruz/Documents/GitHub/omnibotAi/ admin@omniai.local:/home/admin/omniai/
```

### Sync from Pi (backup)
```bash
/Users/mariocruz/Documents/GitHub/omnibotAi/sync_from_pi.sh
```

### Starting the Dashboard
```bash
# On the Pi (works from any directory)
~/omniai/util/start.sh

# With options
~/omniai/util/start.sh --volume 0.7 --port 8080
```

`start.sh` runs `util/smoke_test.py` before launching the dashboard. If
imports, camera, or audio fail, the script exits non-zero and the dashboard
is not started — preventing systemd from flapping a broken build.

### Running as a systemd service
```bash
# On the Pi, one-time install:
~/omniai/util/install_service.sh

# Thereafter:
sudo systemctl status omniai
sudo systemctl restart omniai
journalctl -u omniai -f            # live logs
journalctl -u omniai --since '10 min ago'
```

The unit (`util/omniai.service`) sets `Restart=on-failure` with a 5-second
backoff and a 5-crash-in-2-minutes rate limit. stdout/stderr go to journald
(rotation handled automatically). The dashboard itself calls `os._exit(1)`
when the camera is stale for 60+ seconds so systemd restarts cleanly.

### Health checks
`GET /healthz` returns a JSON snapshot:
```json
{
  "status": "ok",
  "reasons": [],
  "subsystems": {
    "camera": {"age_seconds": 0.12, "fps": 29, "stale": false},
    "robot": {"connected": true},
    "eye": {"alive": true},
    "process": {"uptime_seconds": 1234.5, "running": true, ...},
    "detection": {"last_ago_seconds": 0.5}
  }
}
```
Returns `503` when any subsystem is degraded. `/health` is an alias.

### Logs
`logs/task.log` rotates at 5MB with 3 backups (RotatingFileHandler).
stdout prints under systemd go to journald, which has its own rotation.

## Security Notes
- `speak()` sanitizes input with regex: `[^a-zA-Z0-9\s.,!?'-]` removed
- `espeak` called with `--` to prevent option injection
- API endpoints validate `request.json` (handles None case)
- Camera metadata copied by value (not reference) for thread safety

## Thread Safety Notes

### CameraCapture Thread Safety
The camera runs in a background thread. Critical patterns:

```python
# CORRECT - Returns a copy to prevent race conditions
def get_frame(self):
    with self.frame_lock:
        if self.current_frame is None:
            return None
        return self.current_frame.copy()  # MUST copy!

# WRONG - Returns reference, caller could see partial data
def get_frame(self):
    with self.frame_lock:
        return self.current_frame  # Race condition!
```

### Resource Management
- `capture_request()` MUST be followed by `release()` in a finally block
- `stop()` handles stuck threads by force-stopping the camera first
- Input validation prevents invalid resolution/framerate values

### CameraCapture Validation
```python
# Resolution must be (width, height) tuple of positive integers
# Framerate must be positive number, clamped to 1-120 fps
camera = CameraCapture(
    resolution=(640, 480),  # Validated: tuple of 2 positive ints
    framerate=30            # Validated: positive, clamped 1-120
)
```

## IMX500 Resources
- Official Docs: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Model Zoo: https://github.com/raspberrypi/imx500-models
- Picamera2 Examples: https://github.com/raspberrypi/picamera2/tree/main/examples/imx500
