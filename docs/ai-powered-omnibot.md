# Bringing a 1980s Tomy Omnibot to Life with Modern AI

*How hardware-accelerated vision and cloud LLMs turn a vintage robot into an autonomous companion*

---

## The Vision: Retro Meets Modern AI

The Tomy Omnibot was a marvel of 1980s consumer robotics - a charming wheeled robot that could be programmed via cassette tapes to deliver drinks and perform simple tasks. Four decades later, we've given it a brain transplant: a Raspberry Pi 5 with an AI-accelerated camera and cloud LLM integration.

The result? A robot that can see, think, and act autonomously while maintaining its vintage charm.

---

## The AI Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    PERCEPTION LAYER                          │
│  ┌─────────────┐                                            │
│  │  IMX500 AI  │  Hardware-accelerated YOLOv8               │
│  │   Camera    │  ~17ms inference, 30 FPS                   │
│  │             │  80 object classes (COCO dataset)          │
│  └──────┬──────┘                                            │
└─────────┼───────────────────────────────────────────────────┘
          │ Detections: [{label, confidence, bbox, position}]
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    REASONING LAYER                           │
│  ┌─────────────┐                                            │
│  │  Groq LLM   │  Llama 3.1 8B via cloud API               │
│  │  (Cloud)    │  ~100ms latency                            │
│  │             │  Context: task + detections + history      │
│  └──────┬──────┘                                            │
└─────────┼───────────────────────────────────────────────────┘
          │ Commands: [forward, left, speak("Hello!"), ...]
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ACTION LAYER                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Audio     │    │   Speech    │    │    Eye      │     │
│  │   Tones     │    │   (TTS)     │    │  Display    │     │
│  │  (Movement) │    │             │    │ (Emotion)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Perception - The IMX500 AI Camera

The magic starts with the **Raspberry Pi AI Camera** - an IMX500 sensor with a built-in neural network accelerator. Unlike traditional computer vision that runs on the CPU, this camera has dedicated silicon for AI inference.

### How It Works

1. **Model Upload**: When the system starts, YOLOv8 neural network weights are uploaded directly to the camera chip
2. **Hardware Inference**: Every frame is processed by the on-chip AI accelerator - not the Pi's CPU
3. **Metadata Output**: Detection results arrive as metadata alongside the image data

```python
# The camera returns both image and AI inference results
frame, metadata = camera.get_frame_and_metadata()

# Extract detections from hardware inference
detections = imx500.get_outputs(metadata, add_batch=True)
# Returns: boxes, scores, classes for all detected objects
```

### Performance

| Metric | Value |
|--------|-------|
| Inference Time | ~17ms per frame |
| Frame Rate | 30 FPS continuous |
| Object Classes | 80 (COCO dataset) |
| Model | YOLOv8 nano (640x640) |

The camera can detect:
- **People** (person, face tracking)
- **Animals** (cat, dog, bird, etc.)
- **Vehicles** (car, bicycle, motorcycle)
- **Household items** (cup, bottle, chair, couch, TV)
- **And 70+ more classes**

---

## Layer 2: Reasoning - The LLM Brain

Raw detections aren't useful on their own. The robot needs to understand *what to do* with what it sees. This is where the **Large Language Model** comes in.

### The LLM's Role

The LLM acts as a "mission controller" - it receives:
1. **Current Task**: What the robot is trying to accomplish ("Find and approach people")
2. **Detections**: What objects are visible and where they are
3. **History**: Recent actions taken

And outputs:
- **Movement commands**: forward, backward, left, right, stop
- **Speech commands**: What to say when encountering objects
- **Pattern commands**: dance, circle, search patterns

### The Prompt Structure

```
SYSTEM: You are controlling a Tomy Omnibot robot. Generate movement
commands based on what the camera sees.

Frame dimensions: 640x480
Available commands: forward, backward, left, right, stop,
                   speakText("..."), phrase("hello")

TASK: Find and greet people

CURRENT DETECTIONS:
- person (87% confidence) at x=450, position=right
- chair (72% confidence) at x=200, position=center-left

Generate 1-3 commands to accomplish the task.
```

### LLM Response Processing

```python
# LLM generates natural language response
response = "I see a person on the right side. I'll turn right to face them
           and say hello."

# Parser extracts actionable commands
commands = ["right", "forward", "phrase('hello')"]

# Robot executes sequentially
for cmd in commands:
    robot.execute(cmd)
```

### Cloud vs Local LLM

| Provider | Model | Latency | Use Case |
|----------|-------|---------|----------|
| **Groq** (cloud) | Llama 3.1 8B | ~100ms | Primary - fast, free tier |
| **Ollama** (local) | Mistral 7B | ~2-5s | Offline fallback |

We use Groq's cloud API by default because it's 20-50x faster than running inference locally on the Pi.

---

## Layer 3: Action - Bringing Commands to Life

The Tomy Omnibot was designed to receive commands via audio tones on its cassette input. We exploit this by generating precise sine waves through the Pi's audio output.

### Movement Control

| Command | Frequency | Duration |
|---------|-----------|----------|
| Forward | 1614 Hz | 500ms (one step) |
| Backward | 2013 Hz | 500ms |
| Left | 2208 Hz | 3000ms (90° turn) |
| Right | 1811 Hz | 3000ms |

```python
# Generate a movement tone
def forward(duration_ms):
    frequency = 1614  # Hz
    generate_sine_wave(frequency, duration_ms)
    send_to_bluetooth_speaker()
```

### Speech System

The robot can speak using two methods:

1. **Pre-recorded Phrases** (fast, ~100ms)
   - "Hello", "Yes", "No", "Thank you"
   - Stored as WAV files, instant playback

2. **Text-to-Speech** (flexible, ~500ms)
   - Any text via espeak-ng
   - Used for dynamic responses

### Personality: The Eye Display

A 1.8" ST7735S TFT display shows an animated "eye" that reacts to what the robot sees and does:

| Event | Eye Reaction |
|-------|--------------|
| Person detected | Happy (dilated pupil, smile) |
| Cat/dog detected | Surprised (wide eye) |
| Turning left | Looks left |
| Turning right | Looks right |
| Speaking | Blinks |
| 30s idle | Sleepy (half-closed) |

This gives the robot emotional expressiveness that makes interactions feel more natural.

---

## The Autonomous Loop

When running autonomously, the system operates in a continuous loop:

```
┌──────────────────────────────────────────────────────────────┐
│  1. CAPTURE                                                   │
│     Camera captures frame + runs YOLOv8 on-chip              │
│     Output: [{label: "person", confidence: 0.87, x: 450}]    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  2. REASON                                                    │
│     LLM receives detections + current mission                │
│     Output: ["right", "forward", "phrase('hello')"]          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  3. ACT                                                       │
│     Robot executes commands via audio tones                  │
│     Eye display shows emotional reaction                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  4. REPEAT                                                    │
│     Loop runs every 500ms                                    │
│     ~2 decisions per second with full AI pipeline            │
└──────────────────────────────────────────────────────────────┘
```

---

## Mission Examples

### "Find and Approach People"

```
Detection: person at x=450 (right side of frame)
LLM reasoning: "Person is on the right. Turn right to center them."
Commands: ["right"]

Detection: person at x=320 (center of frame)
LLM reasoning: "Person is centered. Move forward to approach."
Commands: ["forward", "phrase('hello')"]
```

### "Find My Shoes"

```
Detection: [no shoes visible]
LLM reasoning: "No shoes detected. Search by rotating."
Commands: ["left"]

Detection: shoes at x=280 (center-left)
LLM reasoning: "Found shoes! Move toward them."
Commands: ["forward", "speakText('Found your shoes!')"]
```

### "Patrol Mode"

```
Detection: [nothing of interest]
LLM reasoning: "Clear area. Continue patrol pattern."
Commands: ["forward", "forward", "left"]

Detection: person at x=100 (left side)
LLM reasoning: "Person detected during patrol. Alert!"
Commands: ["left", "phrase('hello')"]
```

---

## Why This Architecture Works

### 1. Hardware Acceleration Where It Matters
The IMX500's on-chip inference means the Pi's CPU is free for other tasks. Vision doesn't bottleneck the system.

### 2. Cloud LLM for Complex Reasoning
Running a 7B parameter model locally takes 2-5 seconds per inference. Groq's cloud API returns in ~100ms. This makes real-time autonomous operation practical.

### 3. Simple, Robust Action Layer
Audio tones are deterministic and reliable. No motors to calibrate, no encoders to read. If the frequency is right, the robot moves.

### 4. Graceful Degradation
- No internet? Falls back to local Ollama
- Display not connected? System runs without it
- Camera fails? Manual control still works via web dashboard

---

## The Result

A 40-year-old toy robot that can:
- **See** with hardware-accelerated AI vision
- **Think** using a cloud LLM brain
- **Act** through its original audio-controlled motors
- **Express** emotion through an animated eye display
- **Speak** with pre-recorded and synthesized speech

All while maintaining its vintage charm and that distinctive 1980s robot aesthetic.

---

## Technical Specifications

| Component | Specification |
|-----------|---------------|
| Compute | Raspberry Pi 5 (8GB RAM) |
| Vision | IMX500 AI Camera + YOLOv8 |
| LLM | Groq Llama 3.1 8B (cloud) |
| Display | ST7735S 1.8" TFT (128x160) |
| Audio | Bluetooth to robot speaker |
| Interface | Web dashboard (Flask + WebSocket) |

---

*The future of robotics isn't always about building new hardware - sometimes it's about breathing new life into the classics.*
