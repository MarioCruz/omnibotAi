#!/usr/bin/env python3
"""
AI Robot Dashboard
Live camera stream with detection overlays, LLM commands, and robot controls
"""

import io
import json
import threading
import time
import os
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
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
from llm_command_generator import LLMCommandGenerator
from robot_executor import RobotCommandExecutor

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
system_state = {
    'running': False,
    'paused': False,
    'shutdown': False,  # Flag for graceful shutdown
    'task': None,  # No task by default - must select a mission to enable autonomous mode
    'use_llm': True,
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
llm = None
robot = None
frame_lock = threading.Lock()
current_frame = None
annotated_frame = None

# Cached MJPEG bytes (encode once, share with all clients)
mjpeg_lock = threading.Lock()
cached_mjpeg_bytes = None


def init_system(detector_backend='imx500', llm_model='mistral', volume=0.5):
    """Initialize all system components"""
    global camera, detector, llm, robot

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

    # LLM - pass camera resolution for accurate object positioning
    print(f"[Dashboard] Initializing LLM ({llm_model})...")
    llm = LLMCommandGenerator(
        model_name=llm_model,
        frame_width=camera_resolution[0],
        frame_height=camera_resolution[1]
    )

    # Robot - now uses audio tones for Tomy Omnibot
    print(f"[Dashboard] Initializing audio commander (volume: {volume})...")
    robot = RobotCommandExecutor(volume=volume)
    robot.connect()

    print("[Dashboard] System ready!")


def process_loop():
    """Main processing loop - runs in background thread"""
    global current_frame, annotated_frame, system_state

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

            with frame_lock:
                current_frame = frame.copy()

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

            # Draw detections on frame
            annotated = draw_detections(frame, detections)
            with frame_lock:
                annotated_frame = annotated

            # Generate and execute commands ONLY if a mission/task has been set
            commands = []
            if system_state['task'] and llm:
                # Generate commands based on task
                if system_state['use_llm']:
                    commands = llm.generate_commands(
                        detections,
                        context=system_state['task'],
                        use_llm=True
                    )
                else:
                    commands = llm.generate_commands(detections, use_llm=False)

                # Execute commands
                if commands and robot and robot.connected:
                    for cmd in commands[:3]:  # Limit to 3 commands per cycle
                        robot.execute(cmd)

            # Update stats
            system_state['last_commands'] = commands[:20]
            system_state['stats']['total_commands'] += len(commands)
            system_state['stats']['iterations'] += 1
            system_state['stats']['fps'] = camera.get_fps()

            # Emit update via WebSocket (include LLM debug info)
            llm_debug = llm.last_debug if llm and hasattr(llm, 'last_debug') else {}
            socketio.emit('update', {
                'detections': detections,
                'commands': commands,
                'stats': system_state['stats'],
                'llm_debug': llm_debug
            })

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

        # Draw box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label background
        label = f"{det['label']} {det['confidence']:.0%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, y - label_h - 10), (x + label_w + 10, y), (0, 255, 0), -1)

        # Draw label text
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
                # Encode as JPEG once
                _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                with mjpeg_lock:
                    cached_mjpeg_bytes = frame_bytes
            except Exception:
                pass

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

        .llm-debug {
            background: rgba(0,0,0,0.3);
            border-radius: 6px;
            padding: 10px;
            font-family: monospace;
            font-size: 11px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
        }
        .llm-debug .section {
            margin-bottom: 12px;
        }
        .llm-debug .section-title {
            color: #ff922b;
            font-weight: bold;
            margin-bottom: 4px;
            display: flex;
            justify-content: space-between;
        }
        .llm-debug .section-content {
            color: #aaa;
            padding-left: 8px;
            border-left: 2px solid rgba(255,146,43,0.3);
        }
        .llm-debug .mode-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        }
        .llm-debug .mode-llm { background: #00d4ff; color: #000; }
        .llm-debug .mode-rules { background: #ff922b; color: #000; }

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
                    <img src="/stream" alt="Camera Stream" id="stream">
                </div>
                <div class="controls">
                    <button class="btn-start" onclick="startSystem()">Start</button>
                    <button class="btn-pause" onclick="pauseSystem()">Pause</button>
                    <button class="btn-stop" onclick="stopSystem()">Stop</button>
                </div>
                <div class="task-input">
                    <input type="text" id="taskInput" placeholder="Enter task (e.g., Find and greet people)" value="Explore and interact with objects">
                    <button onclick="setTask()">Set Task</button>
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
                <div class="controls" style="grid-template-columns: repeat(4, 1fr); margin-top: 10px;">
                    <button class="btn-cmd" style="background:#ff922b;" onclick="sendCommand('speakText(\"Hello\")')">Hello</button>
                    <button class="btn-cmd" style="background:#ff922b;" onclick="sendCommand('speakText(\"Yes\")')">Yes</button>
                    <button class="btn-cmd" style="background:#ff922b;" onclick="sendCommand('speakText(\"No\")')">No</button>
                    <button class="btn-cmd" style="background:#ff922b;" onclick="sendCommand('speakText(\"Thank you\")')">Thanks</button>
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
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Detection History</h2>
                <div class="log" id="detectionsList" style="max-height: 200px; overflow-y: auto;">
                    <div style="color: #666; text-align: center; padding: 20px;">No detections yet</div>
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Generated Commands</h2>
                <div class="commands" id="commandsList">
                    <div style="color: #666; text-align: center; padding: 20px;">No commands yet</div>
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>Activity Log</h2>
                <div class="log" id="activityLog"></div>
            </div>
        </div>
    </div>

    <!-- LLM Debug Panel - Full Width -->
    <div style="max-width: 1400px; margin: 20px auto; padding: 0 20px;">
        <div class="panel">
            <h2>LLM Debug <span id="llmModeBadge" class="mode-badge" style="margin-left: 10px;"></span></h2>
            <div class="llm-debug" id="llmDebug">
                <div style="color: #666; text-align: center; padding: 20px;">Waiting for LLM activity...</div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let detectionHistory = [];
        const MAX_DETECTION_HISTORY = 50;

        function logDetection(detections) {
            if (detections.length === 0) return;

            const timestamp = new Date().toLocaleTimeString();
            const summary = detections.map(d => `${d.label} (${(d.confidence * 100).toFixed(0)}%)`).join(', ');

            detectionHistory.unshift({ timestamp, summary, count: detections.length });
            if (detectionHistory.length > MAX_DETECTION_HISTORY) {
                detectionHistory.pop();
            }

            const detList = document.getElementById('detectionsList');
            detList.innerHTML = detectionHistory.map(h => `
                <div class="log-entry info">
                    <span style="color: #888;">[${h.timestamp}]</span>
                    <span style="color: #00ff88;">${h.count}x</span> ${h.summary}
                </div>
            `).join('');
        }

        socket.on('connect', () => {
            log('Connected to server', 'success');
            updateStatus('Connected', false);
        });

        socket.on('disconnect', () => {
            log('Disconnected from server', 'error');
            updateStatus('Disconnected', false);
        });

        socket.on('update', (data) => {
            // Update stats
            document.getElementById('statIterations').textContent = data.stats.iterations;
            document.getElementById('statFps').textContent = data.stats.fps;
            document.getElementById('statDetections').textContent = data.stats.total_detections;
            document.getElementById('statCommands').textContent = data.stats.total_commands;

            // Log detections to history
            logDetection(data.detections);

            // Update commands
            const cmdList = document.getElementById('commandsList');
            if (data.commands.length > 0) {
                cmdList.innerHTML = data.commands.map(c => `
                    <div class="command-item">${c}</div>
                `).join('');
                log(`Executed: ${data.commands.join(', ')}`, 'info');
            }

            // Update LLM debug panel
            if (data.llm_debug && data.llm_debug.prompt) {
                const llmDebug = document.getElementById('llmDebug');
                const modeBadge = document.getElementById('llmModeBadge');

                // Update mode badge
                const mode = data.llm_debug.mode || 'none';
                modeBadge.textContent = mode.toUpperCase();
                modeBadge.className = 'mode-badge mode-' + mode;

                // Escape HTML to prevent XSS
                const escapeHtml = (str) => {
                    const div = document.createElement('div');
                    div.textContent = str;
                    return div.innerHTML;
                };

                llmDebug.innerHTML = `
                    <div class="section">
                        <div class="section-title">
                            <span>PROMPT</span>
                            <span style="color: #666;">${data.llm_debug.timestamp || ''}</span>
                        </div>
                        <div class="section-content">${escapeHtml(data.llm_debug.prompt || '')}</div>
                    </div>
                    <div class="section">
                        <div class="section-title">RESPONSE</div>
                        <div class="section-content">${escapeHtml(data.llm_debug.response || '')}</div>
                    </div>
                    <div class="section">
                        <div class="section-title">PARSED COMMANDS</div>
                        <div class="section-content" style="color: #00ff88;">${
                            (data.llm_debug.parsed_commands || []).join('\\n') || 'None'
                        }</div>
                    </div>
                `;
            }
        });

        function updateStatus(text, running) {
            const el = document.getElementById('systemStatus');
            el.textContent = text;
            el.className = 'status ' + (running ? 'running' : 'stopped');
        }

        function log(message, type = '') {
            const logEl = document.getElementById('activityLog');
            const time = new Date().toLocaleTimeString();
            logEl.innerHTML = `<div class="log-entry ${type}">[${time}] ${message}</div>` + logEl.innerHTML;
            if (logEl.children.length > 50) {
                logEl.removeChild(logEl.lastChild);
            }
        }

        function startSystem() {
            fetch('/api/start', { method: 'POST' })
                .then(r => r.json())
                .then(d => {
                    log('System started', 'success');
                    updateStatus('Running', true);
                });
        }

        function stopSystem() {
            fetch('/api/stop', { method: 'POST' })
                .then(r => r.json())
                .then(d => {
                    log('System stopped', 'error');
                    updateStatus('Stopped', false);
                });
        }

        function pauseSystem() {
            fetch('/api/pause', { method: 'POST' })
                .then(r => r.json())
                .then(d => {
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
            }).then(r => r.json())
              .then(d => log(`Task set: ${task}`, 'info'));
        }

        function sendCommand(cmd) {
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmd })
            }).then(r => r.json())
              .then(d => log(`Audio: ${cmd}`, 'info'));
        }

        function speakText() {
            const text = document.getElementById('speechInput').value;
            if (text) {
                sendCommand(`speakText("${text}")`);
            }
        }
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
                <img src="/stream" alt="Robot Camera">
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
    </div>

    <script>
        let isRunning = false;

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
                body: JSON.stringify({ command: 'speakText("Hello! I am Omnibot!")' })
            });
        }
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
    system_state['task'] = data.get('task', 'Explore')
    return jsonify({'status': 'ok', 'task': system_state['task']})


@app.route('/api/command', methods=['POST'])
def api_command():
    data = request.json or {}
    cmd = data.get('command', '')
    if not cmd:
        return jsonify({'status': 'error', 'message': 'No command provided'})
    if robot and robot.connected:
        result = robot.execute(cmd)
        return jsonify({'status': 'ok', 'result': result.success})
    return jsonify({'status': 'error', 'message': 'Robot not connected'})


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(system_state)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


def shutdown_system():
    """Gracefully shutdown all components"""
    global system_state, camera, robot, process_thread
    print("\n[Dashboard] Shutting down...")
    system_state['shutdown'] = True
    system_state['running'] = False

    if robot:
        robot.disconnect()
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
    parser.add_argument('--llm-model', default='mistral', help='Ollama model')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--no-ssl', action='store_true', help='Disable HTTPS')
    parser.add_argument('--volume', type=float, default=0.5, help='Audio volume (0.0-1.0)')
    args = parser.parse_args()

    # Initialize system
    init_system(
        detector_backend=args.detector,
        llm_model=args.llm_model,
        volume=args.volume
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
