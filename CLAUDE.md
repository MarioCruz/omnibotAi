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
pip install flask flask-cors flask-socketio requests ollama websocket-client python-socketio opencv-python sounddevice
```

### Enable Camera & SPI
```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
# Interface Options -> SPI -> Enable (for IMX500 neural accelerator)
sudo reboot
```

## IMX500 AI Camera - Official picamera2 API

The IMX500 is a smart camera with an on-chip neural network accelerator. The model firmware is uploaded directly to the camera chip, and inference runs on dedicated hardware - NOT on the Pi's CPU.

### Key Concepts

1. **Model Upload**: When you create an `IMX500(model_path)` instance, the neural network firmware is uploaded to the camera's AI chip
2. **Hardware Inference**: Detection runs at ~30fps on-chip with ~17ms inference time
3. **NetworkIntrinsics**: Model metadata (task type, inference rate, bbox format) embedded in the .rpk file
4. **Coordinate Conversion**: Use `imx500.convert_inference_coords()` for accurate bounding box mapping

### Official Python Usage (YOLOv8)

```python
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# Initialize IMX500 with model (uploads firmware to camera chip)
model_path = '/usr/share/imx500-models/imx500_network_yolov8n_pp.rpk'
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

    if intrinsics.bbox_order == "xy":  # YOLOv8 uses xy order
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

**Object Detection (recommended: YOLOv8):**
| Model | File | Notes |
|-------|------|-------|
| **YOLOv8 nano** | `imx500_network_yolov8n_pp.rpk` | Best accuracy, 640x640 input |
| SSD MobileNetV2 | `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | Fastest, 320x320 input |
| NanoDet Plus | `imx500_network_nanodet_plus_416x416_pp.rpk` | Balanced, needs postprocess |
| EfficientDet Lite-0 | `imx500_network_efficientdet_lite0_pp.rpk` | Good accuracy |

**Installing YOLOv8 model (not in default apt package):**
```bash
cd /usr/share/imx500-models
sudo wget https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_yolov8n_pp.rpk
```

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

# Run YOLOv8 detection
python imx500_object_detection_demo.py \
    --model /usr/share/imx500-models/imx500_network_yolov8n_pp.rpk \
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

## Useful Debug Commands

```bash
# Test camera
rpicam-hello -t 5s
vcgencmd get_camera

# Test IMX500 with YOLOv8
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
- `11434` - Ollama API

## Module API Reference

### ObjectDetector (IMX500 YOLOv8)
```python
from object_detector import ObjectDetector

# Uses YOLOv8 by default for best accuracy
detector = ObjectDetector(backend='imx500')

# Or specify a different model
detector = ObjectDetector(
    backend='imx500',
    model_path='/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk',
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

### LLMCommandGenerator
```python
from llm_command_generator import LLMCommandGenerator

# Supports configurable frame dimensions (default 640x480)
llm = LLMCommandGenerator(
    model_name='mistral',
    frame_width=640,   # Used for left/right positioning
    frame_height=480   # Used for top/bottom positioning
)
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

# Movement tones
commander.forward(500)   # 1614 Hz for 500ms
commander.backward(500)  # 2013 Hz
commander.left(500)      # 2208 Hz
commander.right(500)     # 1811 Hz

# Speech (sanitized)
commander.speak("Hello!")
```

## Development Workflow

**Primary development happens on local GitHub repo** to avoid Pi SD card corruption issues.

### Push to Pi (deploy changes)
```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pem' --exclude='.DS_Store' --exclude='._*' --exclude='models/' --exclude='.git/' --exclude='*.log' /Users/mariocruz/Documents/GitHub/omnibotAi/ admin@omniai.local:/home/admin/omniai/
```

### Sync from Pi (backup)
```bash
/Users/mariocruz/Documents/GitHub/omnibotAi/sync_from_pi.sh
```

## Security Notes
- `speak()` sanitizes input with regex: `[^a-zA-Z0-9\s.,!?'-]` removed
- `espeak` called with `--` to prevent option injection
- API endpoints validate `request.json` (handles None case)
- Camera metadata copied by value (not reference) for thread safety

## IMX500 Resources
- Official Docs: https://www.raspberrypi.com/documentation/accessories/ai-camera.html
- Model Zoo: https://github.com/raspberrypi/imx500-models
- Picamera2 Examples: https://github.com/raspberrypi/picamera2/tree/main/examples/imx500
