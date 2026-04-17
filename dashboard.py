#!/usr/bin/env python3
"""
AI Robot Dashboard
Live camera stream with detection overlays, LLM commands, and robot controls
"""

import io
import json
import logging
import re
import subprocess
import threading
import time
import os
import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template_string, jsonify, request

# Task logging — captures every decision cycle when a mission is active
task_logger = logging.getLogger('task')
task_logger.setLevel(logging.DEBUG)
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_log_dir, exist_ok=True)
_task_handler = logging.FileHandler(os.path.join(_log_dir, 'task.log'))
_task_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
task_logger.addHandler(_task_handler)
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
                    print(f"[Env] Loaded {key.strip()}")

load_env()

# Import our modules
from camera_capture import CameraCapture
from object_detector import ObjectDetector
from navigation import NavigationEngine
from robot_executor import RobotCommandExecutor
from eye_display import EyeDisplay


def load_config():
    """Load robot configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}

robot_config = load_config()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
system_state = {
    'running': False,
    'paused': False,
    'shutdown': False,  # Flag for graceful shutdown
    'task': None,  # No task by default - must select a mission to enable autonomous mode
    'detections': [],
    'last_commands': [],
    'stats': {
        'iterations': 0,
        'total_detections': 0,
        'total_commands': 0,
        'fps': 0
    }
}

# Components
camera = None
detector = None
nav = None
robot = None
eye_display = None
frame_lock = threading.Lock()
current_frame = None
annotated_frame = None

# Eye display activity tracking — accessed from process_loop and Flask request threads
activity_lock = threading.Lock()
last_activity_time = time.time()
IDLE_TIMEOUT = 30  # Seconds before going sleepy

# Guards the "robot is currently running a command sequence" flag. Without it,
# two paths could both observe is_executing=False and double-dispatch commands.
executing_lock = threading.Lock()
is_executing = False

# Cached MJPEG bytes (encode once, share with all clients)
mjpeg_lock = threading.Lock()
cached_mjpeg_bytes = None


def mark_activity():
    """Record that the robot is doing something — used for eye idle timeout."""
    global last_activity_time
    with activity_lock:
        last_activity_time = time.time()


def get_idle_time():
    """Seconds since last activity, thread-safe."""
    with activity_lock:
        return time.time() - last_activity_time


def init_system(detector_backend='imx500', volume=0.5,
                eye_display_type='st7735', eye_dc_pin=24, eye_rst_pin=25, eye_cs_pin=0, eye_spi_port=0,
                eye_brightness=15, eye_rotation=0, eye_offset_x=0, eye_offset_y=0):
    """Initialize all system components"""
    global camera, detector, robot

    print("[Dashboard] Initializing system...")

    # Detector first (for IMX500, we need it before camera to get the imx500 instance)
    print(f"[Dashboard] Loading detector ({detector_backend})...")
    try:
        detector = ObjectDetector(backend=detector_backend)
    except Exception as e:
        print(f"[Dashboard] Detector error: {e}")
        detector = None

    # Camera - pass IMX500 instance and intrinsics for hardware-accelerated detection
    print("[Dashboard] Starting camera...")
    imx500_instance = detector.get_imx500() if detector and detector_backend == 'imx500' else None
    intrinsics = detector.get_intrinsics() if detector and detector_backend == 'imx500' else None
    camera_resolution = (640, 480)
    camera = CameraCapture(resolution=camera_resolution, framerate=30, imx500=imx500_instance, intrinsics=intrinsics)

    # Pass picam2 instance to detector for accurate coordinate conversion
    if detector and detector_backend == 'imx500' and hasattr(camera, 'picam2'):
        detector.set_picam2(camera.picam2)

    # Navigation engine - rule-based, no LLM needed
    global nav
    print("[Dashboard] Initializing navigation engine...")
    nav = NavigationEngine(
        frame_width=camera_resolution[0],
        frame_height=camera_resolution[1]
    )

    # Robot - now uses audio tones for Tomy Omnibot
    print(f"[Dashboard] Initializing audio commander (volume: {volume})...")
    robot = RobotCommandExecutor(volume=volume)
    robot.connect()

    # Eye display - configurable via config.json: st7735, ssd1351, or none
    global eye_display
    if eye_display_type == 'none':
        print("[Dashboard] Eye display disabled")
        eye_display = None
    else:
        print(f"[Dashboard] Initializing eye display ({eye_display_type})...")
        try:
            eye_display = EyeDisplay(
                display_type=eye_display_type,
                dc_pin=eye_dc_pin,
                rst_pin=eye_rst_pin,
                cs_pin=eye_cs_pin,
                spi_port=eye_spi_port,
                brightness=eye_brightness,
                rotation=eye_rotation,
                offset_x=eye_offset_x,
                offset_y=eye_offset_y,
            )
            eye_display.start()
            print(f"[Dashboard] Eye display started ({eye_display_type})")
        except Exception as e:
            print(f"[Dashboard] Eye display not available: {e}")
            eye_display = None

    print("[Dashboard] System ready!")


def process_loop():
    """Main processing loop - runs in background thread"""
    global current_frame, annotated_frame, system_state

    last_stale_warn = 0.0
    while not system_state['shutdown']:
        if not system_state['running'] or system_state['paused']:
            time.sleep(0.1)
            continue

        try:
            # Get frame and metadata (metadata needed for IMX500 hardware inference)
            frame, metadata = camera.get_frame_and_metadata()
            if frame is None:
                time.sleep(0.1)
                continue

            # Detect a dead/stuck capture thread — warn at most once per 10s
            age = camera.frame_age() if hasattr(camera, 'frame_age') else None
            if age is not None and age > 5.0 and time.time() - last_stale_warn > 10.0:
                print(f"[Dashboard] Warning: camera frame is {age:.1f}s old — capture thread may be stuck")
                last_stale_warn = time.time()

            with frame_lock:
                current_frame = frame  # Already a copy from get_frame_and_metadata()

            # Detect objects
            detections = []
            if detector:
                # Pass metadata to detector for IMX500 hardware-accelerated inference
                if hasattr(detector, 'set_metadata'):
                    detector.set_metadata(metadata)
                detections = detector.detect(frame)
                # Count detections from this frame (full count, not limited)
                system_state['stats']['total_detections'] += len(detections)
                # Limit stored detections to prevent unbounded memory growth
                system_state['detections'] = detections[:100]
                # Keep last non-empty detections for describe endpoint
                if detections:
                    system_state['last_detections'] = detections[:10]

            # Eye display reactions to detections
            if eye_display and detections:
                mark_activity()
                # Check what we detected
                labels = [d['label'].lower() for d in detections]
                if 'person' in labels:
                    # Happy when seeing a person
                    eye_display.set_expression(EyeDisplay.EXPR_HAPPY)
                elif 'cat' in labels or 'dog' in labels:
                    # Surprised for animals
                    eye_display.set_expression(EyeDisplay.EXPR_SURPRISED)
                else:
                    # Normal for other objects
                    eye_display.set_expression(EyeDisplay.EXPR_NORMAL)

            # Draw detections on frame
            annotated = draw_detections(frame, detections)
            with frame_lock:
                annotated_frame = annotated

            # Generate and execute commands ONLY if a mission/task has been set AND we have detections
            commands = []
            if system_state['task'] and nav and detections:
                # Log detections for this cycle
                det_summary = ', '.join(
                    f"{d['label']}({d['confidence']:.0%} x:{d['bbox']['x']} y:{d['bbox']['y']} w:{d['bbox']['width']} h:{d['bbox']['height']})"
                    for d in detections[:5]
                )
                task_logger.debug(f"DETECT [{system_state['task']}] {det_summary}")

                # Skip if robot is still executing previous commands.
                # Reserve the slot under a lock to prevent two iterations from
                # both observing the flag false and double-dispatching.
                global is_executing
                claimed = False
                with executing_lock:
                    if is_executing:
                        task_logger.debug("SKIP busy executing previous commands")
                    else:
                        is_executing = True
                        claimed = True

                if claimed:
                    commands = nav.generate_commands(detections, context=system_state['task'])

                    debug = nav.last_debug
                    task_logger.info(
                        f"NAV target={debug.get('target','')} pos={debug.get('position','')} "
                        f"commands={commands} | {debug.get('response','')}"
                    )

                    # Execute commands in background to avoid blocking detection pipeline
                    if commands and robot and robot.connected:
                        cmds_to_run = commands[:3]
                        task_logger.info(f"EXEC {cmds_to_run}")
                        def _run_commands(cmds):
                            global is_executing
                            try:
                                for cmd in cmds:
                                    if not system_state['running']:
                                        break
                                    robot.execute(cmd)
                            finally:
                                with executing_lock:
                                    is_executing = False
                                task_logger.debug("EXEC done")
                        threading.Thread(target=_run_commands, args=(cmds_to_run,), daemon=True).start()
                    else:
                        # Nothing to dispatch — release the slot we reserved
                        with executing_lock:
                            is_executing = False

            # Update stats
            system_state['last_commands'] = commands[:20]
            system_state['stats']['total_commands'] += len(commands)
            system_state['stats']['iterations'] += 1
            system_state['stats']['fps'] = camera.get_fps()

            # Emit update via WebSocket
            nav_debug = nav.last_debug if nav and hasattr(nav, 'last_debug') else {}
            socketio.emit('update', {
                'detections': detections,
                'commands': commands,
                'stats': system_state['stats'],
                'llm_debug': nav_debug  # Keep key name for dashboard JS compatibility
            })

            # Eye display idle behavior - go sleepy after inactivity
            if eye_display and not detections:
                if get_idle_time() > IDLE_TIMEOUT:
                    eye_display.set_expression(EyeDisplay.EXPR_SLEEPY)

            time.sleep(0.5)  # Process every 0.5 seconds

        except Exception as e:
            print(f"[Dashboard] Process error: {e}")
            time.sleep(1)


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    annotated = frame.copy()

    for det in detections:
        bbox = det['bbox']
        x = int(bbox['x'])
        y = int(bbox['y'])
        w = int(bbox['width'])
        h = int(bbox['height'])

        # Skip invalid bounding boxes
        if w <= 0 or h <= 0:
            continue

        # Draw box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label (flip below box if too close to top edge)
        label = f"{det['label']} {det['confidence']:.0%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        if y - label_h - 10 < 0:
            # Draw below the top of the box
            cv2.rectangle(annotated, (x, y), (x + label_w + 10, y + label_h + 10), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x + 5, y + label_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.rectangle(annotated, (x, y - label_h - 10), (x + label_w + 10, y), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw status overlay
    task_text = system_state['task'][:30] if system_state['task'] else 'None'
    status = f"Detections: {len(detections)} | Task: {task_text}"
    cv2.putText(annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated


def mjpeg_encoder_loop():
    """Background thread that encodes JPEG once for all clients."""
    global cached_mjpeg_bytes

    while not system_state['shutdown']:
        with frame_lock:
            frame = annotated_frame if annotated_frame is not None else current_frame

        if frame is not None:
            try:
                # Encode as JPEG once (picamera2 RGB888 is already BGR in memory)
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    with mjpeg_lock:
                        cached_mjpeg_bytes = frame_bytes
            except Exception as e:
                print(f"[MJPEG] Encode error: {e}")

        time.sleep(0.033)  # ~30 FPS


def generate_mjpeg():
    """Generate MJPEG stream with detection overlays (uses cached bytes)."""
    while not system_state['shutdown']:
        with mjpeg_lock:
            frame_bytes = cached_mjpeg_bytes

        if frame_bytes is not None:
            yield frame_bytes

        time.sleep(0.033)  # ~30 FPS


# HTML Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Robot Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #00d4ff;
            font-size: 28px;
            margin-bottom: 5px;
        }
        .header .status {
            font-size: 14px;
            color: #888;
        }
        .header .status.running { color: #00ff88; }
        .header .status.stopped { color: #ff4444; }

        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .panel h2 {
            color: #00d4ff;
            font-size: 16px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .stream-container {
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }
        .stream-container img {
            width: 100%;
            height: auto;
            display: block;
        }

        .controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .controls button {
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .controls button:hover { transform: scale(1.02); }
        .btn-start { background: #00ff88; color: #000; }
        .btn-stop { background: #ff4444; color: #fff; }
        .btn-pause { background: #ffaa00; color: #000; }
        .btn-cmd { background: #0066ff; color: #fff; }

        .task-input {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .task-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 14px;
        }
        .task-input button {
            padding: 10px 20px;
            background: #00d4ff;
            color: #000;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }

        .detections {
            max-height: 200px;
            overflow-y: auto;
        }
        .detection-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(0,255,136,0.1);
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid #00ff88;
        }
        .detection-item .label { font-weight: 600; }
        .detection-item .confidence { color: #00ff88; }

        .commands {
            max-height: 200px;
            overflow-y: auto;
        }
        .command-item {
            padding: 8px 12px;
            background: rgba(0,102,255,0.1);
            border-radius: 6px;
            margin-bottom: 8px;
            font-family: monospace;
            font-size: 13px;
            border-left: 3px solid #0066ff;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .stat-item {
            background: rgba(0,0,0,0.2);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-item .value {
            font-size: 24px;
            font-weight: 700;
            color: #00d4ff;
        }
        .stat-item .label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }

        .log {
            max-height: 150px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 6px;
        }
        .log-entry { margin-bottom: 5px; color: #888; }
        .log-entry.info { color: #00d4ff; }
        .log-entry.success { color: #00ff88; }
        .log-entry.error { color: #ff4444; }

        .nav-log {
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .nav-entry {
            padding: 6px 10px;
            margin-bottom: 4px;
            border-radius: 4px;
            background: rgba(0,0,0,0.3);
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .nav-entry .nav-time { color: #666; font-size: 11px; min-width: 70px; }
        .nav-entry .nav-target { color: #00d4ff; font-weight: 600; min-width: 80px; }
        .nav-entry .nav-action { font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
        .nav-action.forward { background: #00ff88; color: #000; }
        .nav-action.left { background: #ff922b; color: #000; }
        .nav-action.right { background: #ff922b; color: #000; }
        .nav-action.stop { background: #ff4444; color: #fff; }
        .nav-entry .nav-reason { color: #aaa; font-size: 11px; }

        @media (max-width: 900px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Robot Dashboard</h1>
        <div class="status" id="systemStatus">Initializing...</div>
    </div>

    <div class="container">
        <div class="left-col">
            <div class="panel">
                <h2>Camera Feed</h2>
                <div class="stream-container">
                    <img src="/stream" alt="Camera Stream" id="stream" onerror="setTimeout(() => this.src='/stream?' + Date.now(), 1000)">
                </div>
                <div class="controls">
                    <button class="btn-start" onclick="startSystem()">Start</button>
                    <button class="btn-pause" onclick="pauseSystem()">Pause</button>
                    <button class="btn-stop" onclick="stopSystem()">Stop</button>
                </div>
                <div class="task-input">
                    <input type="text" id="taskInput" placeholder="Enter task (e.g., Find the person)" value="Find the person">
                    <button onclick="setTask()">Set Task</button>
                    <button onclick="endTask()" style="background:#ff922b;">End Task</button>
                    <button onclick="describeScene()" style="background:#aa44ff; color:#fff;">Describe</button>
                    <label style="display:flex; align-items:center; gap:5px; font-size:12px; color:#888; cursor:pointer;">
                        <input type="checkbox" id="speakLocal" checked> Speak here
                    </label>
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Manual Controls (Audio Tones)</h2>
                <div class="controls">
                    <button class="btn-cmd" onclick="sendCommand('left')">&#9664; Left</button>
                    <button class="btn-cmd" onclick="sendCommand('forward')">&#9650; Forward</button>
                    <button class="btn-cmd" onclick="sendCommand('right')">&#9654; Right</button>
                    <button class="btn-cmd" onclick="sendCommand('dance')">Dance</button>
                    <button class="btn-cmd" onclick="sendCommand('backward')">&#9660; Back</button>
                    <button class="btn-cmd" onclick="sendCommand('circle')">Circle</button>
                    <button class="btn-cmd" onclick="sendCommand('square')">Square</button>
                    <button class="btn-cmd" onclick="sendCommand('stop')" style="background:#ff4444;">Stop</button>
                    <button class="btn-cmd" onclick="sendCommand('spiral')">Spiral</button>
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Robot Speech</h2>
                <div class="task-input">
                    <input type="text" id="speechInput" placeholder="Type message for robot to speak..." value="Hello, I am Tomy Omnibot">
                    <button onclick="speakText()">Speak</button>
                </div>
                <div class="controls" style="grid-template-columns: repeat(5, 1fr); margin-top: 10px;">
                    <button class="btn-cmd" style="background:#ff922b;" onclick="playPhrase('hello')">Hello</button>
                    <button class="btn-cmd" style="background:#ff922b;" onclick="playPhrase('yes')">Yes</button>
                    <button class="btn-cmd" style="background:#ff922b;" onclick="playPhrase('no')">No</button>
                    <button class="btn-cmd" style="background:#ff922b;" onclick="playPhrase('thanks')">Thanks</button>
                    <button class="btn-cmd" style="background:#ff4444;" onclick="sendCommand('speaker_off')">🔇 Speaker Off</button>
                </div>
            </div>
        </div>

        <div class="right-col">
            <div class="panel">
                <h2>Statistics</h2>
                <div class="stats">
                    <div class="stat-item">
                        <div class="value" id="statIterations">0</div>
                        <div class="label">Iterations</div>
                    </div>
                    <div class="stat-item">
                        <div class="value" id="statFps">0</div>
                        <div class="label">FPS</div>
                    </div>
                    <div class="stat-item">
                        <div class="value" id="statDetections">0</div>
                        <div class="label">Detections</div>
                    </div>
                    <div class="stat-item">
                        <div class="value" id="statCommands">0</div>
                        <div class="label">Commands</div>
                    </div>
                    <div class="stat-item">
                        <div class="value" id="statBt" style="color: #666;">--</div>
                        <div class="label">Bluetooth</div>
                    </div>
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Detection History</h2>
                <div class="log" id="detectionsList" style="max-height: 200px; overflow-y: auto;">
                    <div style="color: #666; text-align: center; padding: 20px;">No detections yet</div>
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Navigation Log</h2>
                <div class="nav-log" id="navLog">
                    <div style="color: #666; text-align: center; padding: 20px;">Set a task to start navigation</div>
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Activity Log</h2>
                <div class="log" id="activityLog"></div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const MAX_HISTORY = 20;
        const MAX_NAV_LOG = 30;
        const detectionsList = document.getElementById('detectionsList');
        const navLogEl = document.getElementById('navLog');
        const activityLog = document.getElementById('activityLog');
        let navLogStarted = false;

        socket.on('connect', () => {
            log('Connected to server', 'success');
            updateStatus('Connected', false);
        });

        socket.on('disconnect', () => {
            log('Disconnected from server', 'error');
            updateStatus('Disconnected', false);
        });

        socket.on('update', (data) => {
            // Update stat numbers directly (no innerHTML rebuild)
            document.getElementById('statIterations').textContent = data.stats.iterations;
            document.getElementById('statFps').textContent = data.stats.fps;
            document.getElementById('statDetections').textContent = data.stats.total_detections;
            document.getElementById('statCommands').textContent = data.stats.total_commands;

            // Detection history — only add new entry if there are detections
            if (data.detections.length > 0) {
                const time = new Date().toLocaleTimeString();
                const summary = data.detections.map(d =>
                    d.label + ' (' + (d.confidence * 100 | 0) + '%)'
                ).join(', ');
                const entry = document.createElement('div');
                entry.className = 'log-entry info';
                entry.innerHTML = '<span style="color:#888">[' + time + ']</span> <span style="color:#00ff88">' + data.detections.length + 'x</span> ' + summary;
                detectionsList.prepend(entry);
                while (detectionsList.children.length > MAX_HISTORY) detectionsList.lastChild.remove();
            }

            // Navigation log — show real-time decisions
            const nav = data.llm_debug;
            if (nav && nav.target) {
                if (!navLogStarted) {
                    navLogEl.innerHTML = '';
                    navLogStarted = true;
                }
                const time = nav.timestamp || new Date().toLocaleTimeString();
                // Figure out action type for color coding
                const cmds = nav.parsed_commands || [];
                let actionText = 'idle';
                let actionClass = '';
                if (cmds.length > 0) {
                    const cmd = cmds[0];
                    if (cmd.includes('forward')) { actionText = 'FORWARD'; actionClass = 'forward'; }
                    else if (cmd.includes('left')) { actionText = 'LEFT'; actionClass = 'left'; }
                    else if (cmd.includes('right')) { actionText = 'RIGHT'; actionClass = 'right'; }
                    else if (cmd.includes('stop')) { actionText = 'STOP'; actionClass = 'stop'; }
                }
                const entry = document.createElement('div');
                entry.className = 'nav-entry';
                entry.innerHTML =
                    '<span class="nav-time">' + time + '</span>' +
                    '<span class="nav-target">' + (nav.target || '') + '</span>' +
                    '<span class="nav-action ' + actionClass + '">' + actionText + '</span>' +
                    '<span class="nav-reason">' + (nav.response || '') + '</span>';
                navLogEl.prepend(entry);
                while (navLogEl.children.length > MAX_NAV_LOG) navLogEl.lastChild.remove();
            }
        });

        function updateStatus(text, running) {
            const el = document.getElementById('systemStatus');
            el.textContent = text;
            el.className = 'status ' + (running ? 'running' : 'stopped');
        }

        function log(msg, type) {
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + (type || '');
            entry.textContent = '[' + time + '] ' + msg;
            activityLog.prepend(entry);
            while (activityLog.children.length > 30) activityLog.lastChild.remove();
        }

        function startSystem() {
            fetch('/api/start', { method: 'POST' }).then(() => {
                log('System started', 'success');
                updateStatus('Running', true);
            });
        }

        function stopSystem() {
            fetch('/api/stop', { method: 'POST' }).then(() => {
                log('System stopped', 'error');
                updateStatus('Stopped', false);
                navLogStarted = false;
            });
        }

        function pauseSystem() {
            fetch('/api/pause', { method: 'POST' }).then(() => {
                log('System paused', 'info');
                updateStatus('Paused', false);
            });
        }

        function setTask() {
            const task = document.getElementById('taskInput').value;
            fetch('/api/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: task })
            }).then(() => log('Task set: ' + task, 'info'));
        }

        function endTask() {
            fetch('/api/task/end', { method: 'POST' }).then(() => {
                log('Task ended', 'info');
            });
        }

        function describeScene() {
            const speakLocal = document.getElementById('speakLocal').checked;
            log('Asking robot to describe what it sees...', 'info');
            fetch('/api/describe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speak_robot: !speakLocal })
            })
                .then(r => r.json())
                .then(data => {
                    if (data.description) {
                        log('Robot says: ' + data.description, 'success');
                        if (speakLocal && window.speechSynthesis) {
                            window.speechSynthesis.cancel();
                            const utter = new SpeechSynthesisUtterance(data.description);
                            utter.rate = 1.0;
                            utter.pitch = 0.9;
                            window.speechSynthesis.speak(utter);
                        }
                    } else if (data.error) {
                        log('Describe error: ' + data.error, 'error');
                    }
                })
                .catch(() => log('Describe request failed', 'error'));
        }

        function sendCommand(cmd) {
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmd })
            }).then(() => log('Command: ' + cmd, 'info'));
        }

        function speakText() {
            const text = document.getElementById('speechInput').value;
            if (text) sendCommand('speakText("' + text.replace(/"/g, '') + '")');
        }

        function playPhrase(name) {
            sendCommand('phrase("' + name + '")');
        }

        // Bluetooth check — every 15s is plenty
        function checkBt() {
            fetch('/api/bluetooth').then(r => r.json()).then(data => {
                const el = document.getElementById('statBt');
                el.textContent = data.connected ? (data.devices[0] || 'ON') : 'OFF';
                el.style.color = data.connected ? '#4488ff' : '#ff4444';
            }).catch(() => {});
        }
        checkBt();
        setInterval(checkBt, 15000);
    </script>
</body>
</html>
"""

# Kid-Friendly Dashboard HTML - Retro 80s Tomy Robot Style
KIDS_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMNIBOT COMMAND CENTER</title>
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --neon-pink: #ff00ff;
            --neon-cyan: #00ffff;
            --neon-yellow: #ffff00;
            --neon-orange: #ff8800;
            --neon-green: #00ff00;
            --dark-purple: #1a0a2e;
            --mid-purple: #2d1b4e;
        }

        body {
            font-family: 'Press Start 2P', monospace;
            background: var(--dark-purple);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }

        /* Synthwave grid background */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                linear-gradient(180deg, var(--dark-purple) 0%, var(--mid-purple) 50%, #ff006650 100%),
                repeating-linear-gradient(90deg, transparent, transparent 50px, #ff00ff15 50px, #ff00ff15 51px),
                repeating-linear-gradient(0deg, transparent, transparent 50px, #00ffff15 50px, #00ffff15 51px);
            z-index: -2;
        }

        /* Animated grid floor */
        .grid-floor {
            position: fixed;
            bottom: 0;
            left: -50%;
            width: 200%;
            height: 40%;
            background:
                repeating-linear-gradient(90deg, var(--neon-pink) 0px, transparent 1px, transparent 60px),
                repeating-linear-gradient(0deg, var(--neon-pink) 0px, transparent 1px, transparent 40px);
            transform: perspective(500px) rotateX(60deg);
            transform-origin: center top;
            opacity: 0.3;
            animation: gridMove 2s linear infinite;
            z-index: -1;
        }
        @keyframes gridMove {
            0% { background-position: 0 0; }
            100% { background-position: 0 40px; }
        }

        /* CRT scanline effect */
        .scanlines {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 0, 0, 0.1) 2px,
                rgba(0, 0, 0, 0.1) 4px
            );
            pointer-events: none;
            z-index: 1000;
        }

        .main-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 15px;
            position: relative;
            z-index: 1;
        }

        /* Title with glow */
        .title {
            text-align: center;
            margin-bottom: 20px;
        }
        .title h1 {
            font-size: 18px;
            color: var(--neon-cyan);
            text-shadow:
                0 0 10px var(--neon-cyan),
                0 0 20px var(--neon-cyan),
                0 0 40px var(--neon-cyan);
            letter-spacing: 4px;
            animation: flicker 3s infinite;
        }
        .title .subtitle {
            font-size: 10px;
            color: var(--neon-pink);
            margin-top: 8px;
            text-shadow: 0 0 10px var(--neon-pink);
        }
        @keyframes flicker {
            0%, 100% { opacity: 1; }
            92% { opacity: 1; }
            93% { opacity: 0.8; }
            94% { opacity: 1; }
        }

        /* Video screen with CRT frame */
        .video-frame {
            background: linear-gradient(145deg, #444 0%, #222 50%, #111 100%);
            border-radius: 15px;
            padding: 15px;
            box-shadow:
                inset 0 2px 4px rgba(255,255,255,0.2),
                inset 0 -2px 4px rgba(0,0,0,0.5),
                0 10px 40px rgba(0,0,0,0.8),
                0 0 30px rgba(255,0,255,0.3);
            position: relative;
        }
        .video-frame::before {
            content: 'VISUAL FEED';
            position: absolute;
            top: -8px;
            left: 20px;
            background: var(--dark-purple);
            padding: 0 10px;
            font-size: 8px;
            color: var(--neon-green);
            text-shadow: 0 0 5px var(--neon-green);
        }
        .video-box {
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 3px solid #333;
            position: relative;
        }
        .video-box img {
            width: 100%;
            display: block;
            filter: contrast(1.1) saturate(1.2);
        }
        /* Screen reflection */
        .video-box::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, transparent 100%);
            pointer-events: none;
        }

        /* LED indicator panel */
        .status-panel {
            background: linear-gradient(180deg, #333 0%, #1a1a1a 100%);
            border-radius: 10px;
            padding: 12px 20px;
            margin: 15px 0;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            border: 2px solid #444;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
        }
        .led-cluster {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .led {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #330000;
            border: 2px solid #222;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.5);
        }
        .led.power { background: #003300; }
        .led.power.on {
            background: var(--neon-green);
            box-shadow: 0 0 10px var(--neon-green), 0 0 20px var(--neon-green);
        }
        .led.status { background: #333300; }
        .led.status.on {
            background: var(--neon-yellow);
            box-shadow: 0 0 10px var(--neon-yellow);
            animation: blink 1s infinite;
        }
        .led.bluetooth { background: #000033; }
        .led.bluetooth.on {
            background: #4488ff;
            box-shadow: 0 0 10px #4488ff, 0 0 20px #4488ff;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .status-text {
            font-size: 10px;
            color: var(--neon-green);
            text-shadow: 0 0 5px var(--neon-green);
        }
        .status-text.off { color: #666; text-shadow: none; }

        /* Section labels */
        .section-label {
            font-size: 10px;
            color: var(--neon-yellow);
            text-shadow: 0 0 10px var(--neon-yellow);
            margin: 25px 0 12px;
            text-align: center;
            letter-spacing: 2px;
        }

        /* Arcade buttons - Mission select */
        .missions {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .arcade-btn {
            padding: 15px 10px;
            border: none;
            border-radius: 8px;
            font-family: 'Press Start 2P', monospace;
            font-size: 9px;
            cursor: pointer;
            transition: all 0.1s;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            position: relative;
            text-transform: uppercase;

            /* 3D button effect */
            background: linear-gradient(180deg, #666 0%, #444 50%, #333 100%);
            border-top: 3px solid #888;
            border-left: 3px solid #777;
            border-right: 3px solid #222;
            border-bottom: 5px solid #111;
            color: #fff;
            text-shadow: 2px 2px 0 #000;
        }
        .arcade-btn:hover {
            transform: translateY(-2px);
            border-bottom-width: 7px;
        }
        .arcade-btn:active {
            transform: translateY(2px);
            border-bottom-width: 2px;
        }
        .arcade-btn .icon {
            font-size: 28px;
            filter: drop-shadow(2px 2px 0 #000);
        }

        /* Button color variants with glow strips */
        .arcade-btn.pink {
            background: linear-gradient(180deg, #ff66ff 0%, #cc00cc 50%, #990099 100%);
            border-top-color: #ff99ff;
            border-left-color: #ff66ff;
        }
        .arcade-btn.pink::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 10%;
            right: 10%;
            height: 3px;
            background: var(--neon-pink);
            box-shadow: 0 0 10px var(--neon-pink);
            border-radius: 2px;
        }

        .arcade-btn.cyan {
            background: linear-gradient(180deg, #66ffff 0%, #00cccc 50%, #009999 100%);
            border-top-color: #99ffff;
            border-left-color: #66ffff;
            color: #003333;
            text-shadow: 1px 1px 0 rgba(255,255,255,0.3);
        }
        .arcade-btn.cyan::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 10%;
            right: 10%;
            height: 3px;
            background: var(--neon-cyan);
            box-shadow: 0 0 10px var(--neon-cyan);
            border-radius: 2px;
        }

        .arcade-btn.yellow {
            background: linear-gradient(180deg, #ffff66 0%, #cccc00 50%, #999900 100%);
            border-top-color: #ffff99;
            border-left-color: #ffff66;
            color: #333300;
            text-shadow: 1px 1px 0 rgba(255,255,255,0.3);
        }
        .arcade-btn.yellow::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 10%;
            right: 10%;
            height: 3px;
            background: var(--neon-yellow);
            box-shadow: 0 0 10px var(--neon-yellow);
            border-radius: 2px;
        }

        .arcade-btn.orange {
            background: linear-gradient(180deg, #ffaa66 0%, #ff8800 50%, #cc6600 100%);
            border-top-color: #ffcc99;
            border-left-color: #ffaa66;
        }
        .arcade-btn.orange::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 10%;
            right: 10%;
            height: 3px;
            background: var(--neon-orange);
            box-shadow: 0 0 10px var(--neon-orange);
            border-radius: 2px;
        }

        .arcade-btn.green {
            background: linear-gradient(180deg, #66ff66 0%, #00cc00 50%, #009900 100%);
            border-top-color: #99ff99;
            border-left-color: #66ff66;
            color: #003300;
            text-shadow: 1px 1px 0 rgba(255,255,255,0.3);
        }
        .arcade-btn.green::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 10%;
            right: 10%;
            height: 3px;
            background: var(--neon-green);
            box-shadow: 0 0 10px var(--neon-green);
            border-radius: 2px;
        }
        .arcade-btn.red {
            background: linear-gradient(180deg, #ff6666 0%, #cc0000 50%, #990000 100%);
            border-top-color: #ff9999;
            border-left-color: #ff6666;
            color: #330000;
            text-shadow: 1px 1px 0 rgba(255,255,255,0.3);
        }
        .arcade-btn.red::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 10%;
            right: 10%;
            height: 3px;
            background: #ff4444;
            box-shadow: 0 0 10px #ff4444;
            border-radius: 2px;
        }

        /* D-Pad controller */
        .dpad-container {
            display: flex;
            justify-content: center;
            margin: 10px 0 20px;
        }
        .dpad {
            display: grid;
            grid-template-columns: repeat(3, 60px);
            grid-template-rows: repeat(3, 60px);
            gap: 5px;
        }
        .dpad-btn {
            border: none;
            border-radius: 8px;
            font-size: 24px;
            cursor: pointer;
            background: linear-gradient(180deg, #555 0%, #333 50%, #222 100%);
            border-top: 2px solid #666;
            border-left: 2px solid #555;
            border-right: 2px solid #111;
            border-bottom: 4px solid #000;
            color: var(--neon-cyan);
            text-shadow: 0 0 10px var(--neon-cyan);
            transition: all 0.05s;
        }
        .dpad-btn:hover {
            background: linear-gradient(180deg, #666 0%, #444 50%, #333 100%);
        }
        .dpad-btn:active {
            transform: translateY(2px);
            border-bottom-width: 2px;
            background: linear-gradient(180deg, #444 0%, #333 50%, #222 100%);
        }
        .dpad-btn.stop {
            background: linear-gradient(180deg, #ff4444 0%, #cc0000 50%, #990000 100%);
            border-top-color: #ff6666;
            border-left-color: #ff4444;
            color: #fff;
            text-shadow: 0 0 5px #fff;
            font-size: 16px;
        }
        .dpad-btn.empty {
            visibility: hidden;
        }

        /* Power button */
        .power-section {
            margin: 15px 0;
        }
        .power-btn {
            display: block;
            width: 100%;
            padding: 18px;
            border: none;
            border-radius: 10px;
            font-family: 'Press Start 2P', monospace;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.1s;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .power-btn.start {
            background: linear-gradient(180deg, #66ff66 0%, #00cc00 50%, #009900 100%);
            border-top: 3px solid #99ff99;
            border-left: 3px solid #66ff66;
            border-right: 3px solid #006600;
            border-bottom: 6px solid #003300;
            color: #003300;
            text-shadow: 1px 1px 0 rgba(255,255,255,0.3);
            box-shadow: 0 0 20px rgba(0,255,0,0.3);
        }
        .power-btn.stop {
            background: linear-gradient(180deg, #ff6666 0%, #cc0000 50%, #990000 100%);
            border-top: 3px solid #ff9999;
            border-left: 3px solid #ff6666;
            border-right: 3px solid #660000;
            border-bottom: 6px solid #330000;
            color: #fff;
            text-shadow: 2px 2px 0 #000;
            box-shadow: 0 0 20px rgba(255,0,0,0.3);
        }
        .power-btn:hover {
            transform: translateY(-2px);
        }
        .power-btn:active {
            transform: translateY(3px);
            border-bottom-width: 2px;
        }

        /* Mission display panel */
        .mission-display {
            background: #111;
            border: 3px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            position: relative;
        }
        .mission-display::before {
            content: 'CURRENT MISSION';
            position: absolute;
            top: -8px;
            left: 15px;
            background: var(--dark-purple);
            padding: 0 8px;
            font-size: 7px;
            color: var(--neon-orange);
            text-shadow: 0 0 5px var(--neon-orange);
        }
        .mission-display .mission-text {
            font-size: 10px;
            color: var(--neon-green);
            text-shadow: 0 0 10px var(--neon-green);
            text-align: center;
            min-height: 20px;
        }
        .mission-display .mission-text.none {
            color: #444;
            text-shadow: none;
        }

        /* Robot Brain panel */
        .robot-brain {
            background: #111;
            border: 3px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            position: relative;
            max-height: 250px;
            overflow-y: auto;
        }
        .robot-brain::before {
            content: 'ROBOT BRAIN';
            position: absolute;
            top: -8px;
            left: 15px;
            background: var(--dark-purple);
            padding: 0 8px;
            font-size: 7px;
            color: var(--neon-cyan);
            text-shadow: 0 0 5px var(--neon-cyan);
        }
        .brain-entry {
            padding: 5px 8px;
            margin-bottom: 3px;
            border-radius: 4px;
            background: rgba(0,0,0,0.5);
            font-size: 9px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .brain-entry .brain-icon { font-size: 16px; min-width: 20px; text-align: center; }
        .brain-entry .brain-target { color: var(--neon-cyan); min-width: 60px; }
        .brain-entry .brain-action {
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            min-width: 55px;
            text-align: center;
        }
        .brain-action.forward { background: var(--neon-green); color: #000; }
        .brain-action.left { background: var(--neon-orange); color: #000; }
        .brain-action.right { background: var(--neon-orange); color: #000; }
        .brain-action.stop { background: #ff4444; color: #fff; }
        .brain-entry .brain-detail { color: #888; }

        /* Decorative corner brackets */
        .corner-decor {
            position: fixed;
            width: 40px;
            height: 40px;
            border: 3px solid var(--neon-pink);
            opacity: 0.5;
        }
        .corner-decor.tl { top: 10px; left: 10px; border-right: none; border-bottom: none; }
        .corner-decor.tr { top: 10px; right: 10px; border-left: none; border-bottom: none; }
        .corner-decor.bl { bottom: 10px; left: 10px; border-right: none; border-top: none; }
        .corner-decor.br { bottom: 10px; right: 10px; border-left: none; border-top: none; }

        @media (max-width: 600px) {
            .title h1 { font-size: 12px; letter-spacing: 2px; }
            .title .subtitle { font-size: 8px; }
            .missions { grid-template-columns: 1fr; }
            .arcade-btn { font-size: 8px; padding: 12px 8px; }
            .arcade-btn .icon { font-size: 24px; }
            .dpad { grid-template-columns: repeat(3, 50px); grid-template-rows: repeat(3, 50px); }
            .section-label { font-size: 8px; }
            .power-btn { font-size: 10px; }
        }
    </style>
</head>
<body>
    <div class="scanlines"></div>
    <div class="grid-floor"></div>
    <div class="corner-decor tl"></div>
    <div class="corner-decor tr"></div>
    <div class="corner-decor bl"></div>
    <div class="corner-decor br"></div>

    <div class="main-container">
        <div class="title">
            <h1>OMNIBOT COMMAND CENTER</h1>
            <div class="subtitle">// TOMY ROBOTICS DIVISION //</div>
        </div>

        <div class="video-frame">
            <div class="video-box">
                <img src="/stream" alt="Robot Camera" id="cameraStream" onerror="setTimeout(() => this.src='/stream?' + Date.now(), 1000)">
            </div>
        </div>

        <div class="status-panel">
            <div class="led-cluster">
                <div class="led power" id="powerLed"></div>
                <span class="status-text off" id="statusText">STANDBY</span>
            </div>
            <div class="led-cluster">
                <div class="led status" id="statusLed"></div>
                <span class="status-text off" id="modeText">IDLE</span>
            </div>
            <div class="led-cluster">
                <div class="led bluetooth" id="btLed"></div>
                <span class="status-text off" id="btText">BT OFF</span>
            </div>
            <div class="led-cluster">
                <label style="display:flex; align-items:center; gap:5px; cursor:pointer;">
                    <input type="checkbox" id="speakLocal" checked>
                    <span class="status-text" style="color:var(--neon-yellow);">SPEAK HERE</span>
                </label>
            </div>
        </div>

        <div class="power-section">
            <button class="power-btn start" id="powerBtn" onclick="togglePower()">
                [ ACTIVATE ROBOT ]
            </button>
        </div>

        <div class="section-label">// SELECT MISSION //</div>
        <div class="missions">
            <button class="arcade-btn pink" onclick="startMission('FIND SHOE', 'Find and go to the shoe')">
                <span class="icon">👟</span>
                <span>Find Shoe</span>
            </button>
            <button class="arcade-btn cyan" onclick="startMission('FIND HUMAN', 'Find and greet any person you see')">
                <span class="icon">👤</span>
                <span>Find Human</span>
            </button>
            <button class="arcade-btn yellow" onclick="startMission('FIND BALL', 'Find and approach the sports ball')">
                <span class="icon">⚽</span>
                <span>Find Ball</span>
            </button>
            <button class="arcade-btn orange" onclick="startMission('EXPLORE', 'Explore and describe what you see')">
                <span class="icon">🔍</span>
                <span>Explore</span>
            </button>
            <button class="arcade-btn green" onclick="doDance()">
                <span class="icon">🕺</span>
                <span>Dance</span>
            </button>
            <button class="arcade-btn cyan" onclick="sayHello()">
                <span class="icon">📢</span>
                <span>Speak</span>
            </button>
            <button class="arcade-btn cyan" onclick="describeScene()">
                <span class="icon">👁</span>
                <span>What See?</span>
            </button>
            <button class="arcade-btn orange" onclick="endMission()">
                <span class="icon">✋</span>
                <span>End Mission</span>
            </button>
            <button class="arcade-btn red" onclick="speakerOff()">
                <span class="icon">🔇</span>
                <span>Quiet</span>
            </button>
        </div>

        <div class="section-label">// MANUAL CONTROL //</div>
        <div class="dpad-container">
            <div class="dpad">
                <div class="dpad-btn empty"></div>
                <button class="dpad-btn" onclick="drive('forward')">▲</button>
                <div class="dpad-btn empty"></div>
                <button class="dpad-btn" onclick="drive('left')">◄</button>
                <button class="dpad-btn stop" onclick="drive('stop')">STOP</button>
                <button class="dpad-btn" onclick="drive('right')">►</button>
                <div class="dpad-btn empty"></div>
                <button class="dpad-btn" onclick="drive('backward')">▼</button>
                <div class="dpad-btn empty"></div>
            </div>
        </div>

        <div class="mission-display">
            <div class="mission-text none" id="missionText">AWAITING ORDERS...</div>
        </div>

        <div class="robot-brain" id="robotBrain">
            <div style="color: #444; text-align: center; padding: 10px; font-size: 9px;">
                ACTIVATE ROBOT AND SELECT A MISSION
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        let isRunning = false;
        const socket = io();
        const brainEl = document.getElementById('robotBrain');
        let brainStarted = false;

        socket.on('update', (data) => {
            // Show what robot sees
            if (data.detections && data.detections.length > 0) {
                const det = data.detections[0];
                const label = det.label;
                const conf = (det.confidence * 100) | 0;
                const icon = label === 'person' ? '👤' : label === 'cat' ? '🐱' : label === 'dog' ? '🐶' : '📦';

                // Show nav decision if available
                const nav = data.llm_debug;
                if (nav && nav.target) {
                    if (!brainStarted) { brainEl.innerHTML = ''; brainStarted = true; }
                    const cmds = nav.parsed_commands || [];
                    let actionText = 'LOOK';
                    let actionClass = '';
                    if (cmds.length > 0) {
                        const cmd = cmds[0];
                        if (cmd.includes('forward')) { actionText = 'GO!'; actionClass = 'forward'; }
                        else if (cmd.includes('left')) { actionText = 'LEFT'; actionClass = 'left'; }
                        else if (cmd.includes('right')) { actionText = 'RIGHT'; actionClass = 'right'; }
                        else if (cmd.includes('stop')) { actionText = 'STOP'; actionClass = 'stop'; }
                    }
                    const entry = document.createElement('div');
                    entry.className = 'brain-entry';
                    entry.innerHTML =
                        '<span class="brain-icon">' + icon + '</span>' +
                        '<span class="brain-target">' + label + ' ' + conf + '%</span>' +
                        '<span class="brain-action ' + actionClass + '">' + actionText + '</span>' +
                        '<span class="brain-detail">' + (nav.response || '').substring(0, 40) + '</span>';
                    brainEl.prepend(entry);
                    while (brainEl.children.length > 20) brainEl.lastChild.remove();
                }
            }
        });

        function togglePower() {
            const btn = document.getElementById('powerBtn');
            const powerLed = document.getElementById('powerLed');
            const statusLed = document.getElementById('statusLed');
            const statusText = document.getElementById('statusText');
            const modeText = document.getElementById('modeText');

            if (!isRunning) {
                fetch('/api/start', { method: 'POST' });
                btn.textContent = '[ DEACTIVATE ROBOT ]';
                btn.className = 'power-btn stop';
                powerLed.className = 'led power on';
                statusLed.className = 'led status on';
                statusText.textContent = 'ONLINE';
                statusText.className = 'status-text';
                modeText.textContent = 'ACTIVE';
                modeText.className = 'status-text';
                isRunning = true;
            } else {
                fetch('/api/stop', { method: 'POST' });
                btn.textContent = '[ ACTIVATE ROBOT ]';
                btn.className = 'power-btn start';
                powerLed.className = 'led power';
                statusLed.className = 'led status';
                statusText.textContent = 'STANDBY';
                statusText.className = 'status-text off';
                modeText.textContent = 'IDLE';
                modeText.className = 'status-text off';
                isRunning = false;
            }
        }

        function startMission(displayName, taskText) {
            const missionEl = document.getElementById('missionText');
            missionEl.textContent = displayName;
            missionEl.className = 'mission-text';
            fetch('/api/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: taskText })
            });

            // Auto-start if not running
            if (!isRunning) {
                togglePower();
            }
        }

        function drive(direction) {
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: direction })
            });
        }

        function doDance() {
            const missionEl = document.getElementById('missionText');
            missionEl.textContent = 'DANCE MODE';
            missionEl.className = 'mission-text';
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: 'dance' })
            });
        }

        function sayHello() {
            const missionEl = document.getElementById('missionText');
            missionEl.textContent = 'VOICE OUTPUT';
            missionEl.className = 'mission-text';
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: 'phrase("omnibot")' })
            });
        }

        function speakerOff() {
            const missionEl = document.getElementById('missionText');
            missionEl.textContent = 'SPEAKER OFF';
            missionEl.className = 'mission-text';
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: 'speaker_off' })
            });
        }

        function endMission() {
            const missionEl = document.getElementById('missionText');
            missionEl.textContent = 'AWAITING ORDERS...';
            missionEl.className = 'mission-text none';
            fetch('/api/task/end', { method: 'POST' });
        }

        function describeScene() {
            const missionEl = document.getElementById('missionText');
            const speakLocal = document.getElementById('speakLocal').checked;
            missionEl.textContent = 'THINKING...';
            missionEl.className = 'mission-text';
            fetch('/api/describe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speak_robot: !speakLocal })
            })
                .then(r => r.json())
                .then(data => {
                    const text = data.description || data.error || 'NO RESPONSE';
                    missionEl.textContent = text;
                    if (speakLocal && window.speechSynthesis) {
                        window.speechSynthesis.cancel();
                        const utter = new SpeechSynthesisUtterance(text);
                        utter.rate = 1.0;
                        utter.pitch = 0.9;
                        window.speechSynthesis.speak(utter);
                    }
                    setTimeout(() => {
                        missionEl.textContent = 'AWAITING ORDERS...';
                        missionEl.className = 'mission-text none';
                    }, 10000);
                })
                .catch(() => { missionEl.textContent = 'ERROR'; });
        }

        function checkBluetooth() {
            fetch('/api/bluetooth')
                .then(r => r.json())
                .then(data => {
                    const led = document.getElementById('btLed');
                    const text = document.getElementById('btText');
                    if (data.connected) {
                        led.className = 'led bluetooth on';
                        text.textContent = data.devices[0] || 'BT ON';
                        text.className = 'status-text';
                    } else {
                        led.className = 'led bluetooth';
                        text.textContent = 'BT OFF';
                        text.className = 'status-text off';
                    }
                })
                .catch(() => {});
        }

        checkBluetooth();
        setInterval(checkBluetooth, 15000);
    </script>
</body>
</html>
"""


# Routes
@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/kids')
def kids_dashboard():
    """Kid-friendly simplified dashboard"""
    return render_template_string(KIDS_DASHBOARD_HTML)


@app.route('/stream')
def stream():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def api_start():
    system_state['running'] = True
    system_state['paused'] = False
    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    system_state['running'] = False
    if system_state['task']:
        task_logger.info(f"=== TASK STOPPED: {system_state['task']} ===")
    system_state['task'] = None  # Clear task to stop autonomous commands
    if robot:
        robot.stop()
    return jsonify({'status': 'stopped'})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    system_state['paused'] = not system_state['paused']
    return jsonify({'status': 'paused' if system_state['paused'] else 'resumed'})


@app.route('/api/task', methods=['POST'])
def api_task():
    data = request.json or {}
    raw = data.get('task', 'Explore')
    if not isinstance(raw, str):
        return jsonify({'status': 'error', 'message': 'task must be a string'}), 400
    # Strip control chars, cap length — task strings end up in log lines.
    cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', raw)[:200].strip()
    if not cleaned:
        return jsonify({'status': 'error', 'message': 'task is empty after sanitization'}), 400
    system_state['task'] = cleaned
    task_logger.info(f"=== TASK SET: {cleaned} ===")
    return jsonify({'status': 'ok', 'task': cleaned})


@app.route('/api/task/end', methods=['POST'])
def api_task_end():
    """End the current task without stopping the system (detection continues)."""
    if system_state['task']:
        task_logger.info(f"=== TASK ENDED: {system_state['task']} ===")
    system_state['task'] = None
    return jsonify({'status': 'ok', 'task': None})


@app.route('/api/describe', methods=['POST'])
def api_describe():
    """Use LLM to describe what the robot currently sees, optionally speak through robot."""
    data = request.json or {}
    speak_robot = data.get('speak_robot', True)
    detections = system_state.get('last_detections', []) or system_state.get('detections', [])
    if not detections:
        return jsonify({'error': 'No objects detected right now'})

    # Build a simple description from detections
    objects = [f"{d['label']} ({d['confidence']:.0%})" for d in detections[:5]]
    det_text = ', '.join(objects)

    # Try Groq LLM for a natural description
    api_key = os.environ.get('GROQ_API_KEY')
    description = None
    if api_key:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are a robot. Say what you see in under 10 words. No punctuation except periods. No dashes or special characters. Simple words only."},
                        {"role": "user", "content": f"What do you see: {det_text}"}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 30,
                },
                timeout=15
            )
            resp.raise_for_status()
            payload = resp.json()
            choices = payload.get('choices') or []
            if choices:
                content = (choices[0].get('message') or {}).get('content')
                if content:
                    description = content.strip()
            if not description:
                print(f"[Describe] Groq returned no content: {payload}")
        except Exception as e:
            print(f"[Describe] Groq error: {e}")

    # Fallback to simple description if no LLM
    if not description:
        count = len(detections)
        labels = list(set(d['label'] for d in detections[:5]))
        description = f"I can see {count} things: {', '.join(labels)}"

    # Clean up for speech (strip special chars, limit length)
    speak_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', description)[:100]

    # Bind robot.audio to a local before use — robot could be torn down between
    # the check and the dereference in another thread.
    audio = robot.audio if (robot and robot.connected) else None
    if speak_robot and audio and speak_text:
        threading.Thread(
            target=audio.speak,
            args=(speak_text,),
            daemon=True
        ).start()

    task_logger.info(f"DESCRIBE: {description}")
    return jsonify({'status': 'ok', 'description': description})


@app.route('/api/command', methods=['POST'])
def api_command():
    data = request.json or {}
    cmd = data.get('command', '')
    if not cmd:
        return jsonify({'status': 'error', 'message': 'No command provided'})

    # Eye display reactions to commands
    if eye_display:
        mark_activity()
        cmd_lower = cmd.lower()
        if cmd_lower == 'left':
            eye_display.set_expression(EyeDisplay.EXPR_LOOKING_LEFT)
        elif cmd_lower == 'right':
            eye_display.set_expression(EyeDisplay.EXPR_LOOKING_RIGHT)
        elif cmd_lower == 'forward':
            eye_display.set_expression(EyeDisplay.EXPR_LOOKING_UP)
        elif cmd_lower == 'backward':
            eye_display.set_expression(EyeDisplay.EXPR_LOOKING_DOWN)
        elif 'speak' in cmd_lower or 'phrase' in cmd_lower:
            eye_display.blink()
            eye_display.set_expression(EyeDisplay.EXPR_HAPPY)
        elif cmd_lower == 'dance':
            eye_display.set_expression(EyeDisplay.EXPR_HAPPY)
        elif cmd_lower == 'stop' or cmd_lower == 'speaker_off':
            eye_display.set_expression(EyeDisplay.EXPR_NORMAL)

    if robot and robot.connected:
        # Special handling for speaker_off - kills speech and sends speaker off tone
        if cmd == 'speaker_off':
            if robot.audio:
                robot.audio.stop_speaking()
                return jsonify({'status': 'ok', 'result': True, 'message': 'Speaker off sent'})
            return jsonify({'status': 'error', 'message': 'Audio not available'})
        result = robot.execute(cmd)
        return jsonify({'status': 'ok', 'result': result.success})
    return jsonify({'status': 'error', 'message': 'Robot not connected'})


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(system_state)


@app.route('/api/bluetooth', methods=['GET'])
def api_bluetooth():
    """Check Bluetooth connection status."""
    try:
        result = subprocess.run(
            ['bluetoothctl', 'devices', 'Connected'],
            capture_output=True, text=True, timeout=3
        )
        devices = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        connected = len(devices) > 0
        # Parse device names (format: "Device XX:XX:XX:XX:XX:XX Name")
        names = []
        for dev in devices:
            parts = dev.split(' ', 2)
            if len(parts) >= 3:
                names.append(parts[2])
        return jsonify({'connected': connected, 'devices': names})
    except Exception:
        return jsonify({'connected': False, 'devices': []})


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


def shutdown_system():
    """Gracefully shutdown all components"""
    global system_state, camera, detector, robot, eye_display, process_thread
    print("\n[Dashboard] Shutting down...")
    system_state['shutdown'] = True
    system_state['running'] = False

    if eye_display:
        eye_display.stop()
    if robot:
        robot.disconnect()
    if detector:
        detector.stop()
    if camera:
        camera.stop()

    print("[Dashboard] Shutdown complete")


if __name__ == '__main__':
    import argparse
    import ssl
    import os
    import signal

    parser = argparse.ArgumentParser(description='AI Robot Dashboard')
    parser.add_argument('--detector', default='imx500', choices=['imx500', 'yolov5', 'mediapipe'])
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--no-ssl', action='store_true', help='Disable HTTPS')
    parser.add_argument('--volume', type=float, default=robot_config.get('volume', 0.5),
                        help='Audio volume (0.0-1.0)')
    parser.add_argument('--eye-display', default=robot_config.get('eye_display', 'st7735'),
                        choices=['st7735', 'ssd1351', 'none'],
                        help='Eye display type (default: from config.json)')
    args = parser.parse_args()

    # Initialize system with config.json defaults merged with CLI overrides
    init_system(
        detector_backend=args.detector,
        volume=args.volume,
        eye_display_type=args.eye_display,
        eye_dc_pin=robot_config.get('eye_dc_pin', 24),
        eye_rst_pin=robot_config.get('eye_rst_pin', 25),
        eye_cs_pin=robot_config.get('eye_cs_pin', 0),
        eye_spi_port=robot_config.get('eye_spi_port', 0),
        eye_brightness=robot_config.get('eye_brightness', 15),
        eye_rotation=robot_config.get('eye_rotation', 0),
        eye_offset_x=robot_config.get('eye_offset_x', 0),
        eye_offset_y=robot_config.get('eye_offset_y', 0),
    )

    # Start processing thread
    process_thread = threading.Thread(target=process_loop, daemon=True)
    process_thread.start()

    # Start MJPEG encoder thread (encodes once, shared by all clients)
    encoder_thread = threading.Thread(target=mjpeg_encoder_loop, daemon=True)
    encoder_thread.start()

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        shutdown_system()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # SSL setup
    ssl_context = None
    if not args.no_ssl and os.path.exists('cert.pem') and os.path.exists('key.pem'):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain('cert.pem', 'key.pem')
        protocol = 'https'
    else:
        protocol = 'http'

    print("\n" + "=" * 50)
    print("  AI Robot Dashboard (Tomy Omnibot)")
    print("=" * 50)
    print(f"  Open: {protocol}://<your-pi-ip>:{args.port}")
    print(f"  Audio output: Pi audio jack/Bluetooth")
    print(f"  Volume: {args.volume}")
    print("=" * 50 + "\n")

    socketio.run(app, host='0.0.0.0', port=args.port, ssl_context=ssl_context, allow_unsafe_werkzeug=True)
