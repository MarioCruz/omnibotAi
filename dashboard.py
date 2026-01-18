#!/usr/bin/env python3
"""
AI Robot Dashboard
Live camera stream with detection overlays, LLM commands, and robot controls
"""

import io
import json
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

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
    'task': 'Explore and interact with objects',
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

    # Camera - pass IMX500 instance if using AI camera for hardware-accelerated detection
    print("[Dashboard] Starting camera...")
    imx500_instance = detector.get_imx500() if detector and detector_backend == 'imx500' else None
    camera = CameraCapture(resolution=(640, 480), framerate=30, imx500=imx500_instance)

    # LLM
    print(f"[Dashboard] Initializing LLM ({llm_model})...")
    llm = LLMCommandGenerator(model_name=llm_model)

    # Robot - now uses audio tones for Tomy Omnibot
    print(f"[Dashboard] Initializing audio commander (volume: {volume})...")
    robot = RobotCommandExecutor(volume=volume)
    robot.connect()

    print("[Dashboard] System ready!")


def process_loop():
    """Main processing loop - runs in background thread"""
    global current_frame, annotated_frame, system_state

    while True:
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
                system_state['detections'] = detections
                system_state['stats']['total_detections'] += len(detections)

            # Draw detections on frame
            annotated = draw_detections(frame, detections)
            with frame_lock:
                annotated_frame = annotated

            # Generate commands
            if system_state['use_llm'] and llm:
                commands = llm.generate_commands(
                    detections,
                    context=system_state['task'],
                    use_llm=system_state['use_llm']
                )
            else:
                commands = llm.generate_commands(detections, use_llm=False) if llm else []

            system_state['last_commands'] = commands
            system_state['stats']['total_commands'] += len(commands)
            system_state['stats']['iterations'] += 1
            system_state['stats']['fps'] = camera.get_fps()

            # Execute commands
            if commands and robot and robot.connected:
                for cmd in commands[:3]:  # Limit to 3 commands per cycle
                    robot.execute(cmd)

            # Emit update via WebSocket
            socketio.emit('update', {
                'detections': detections,
                'commands': commands,
                'stats': system_state['stats']
            })

            time.sleep(1.5)  # Process every 1.5 seconds

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
    status = f"Detections: {len(detections)} | Task: {system_state['task'][:30]}"
    cv2.putText(annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated


def generate_mjpeg():
    """Generate MJPEG stream with detection overlays"""
    while True:
        with frame_lock:
            frame = annotated_frame if annotated_frame is not None else current_frame

        if frame is not None:
            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

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
                <h2>Detections</h2>
                <div class="detections" id="detectionsList">
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

    <script>
        const socket = io();

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

            // Update detections
            const detList = document.getElementById('detectionsList');
            if (data.detections.length > 0) {
                detList.innerHTML = data.detections.map(d => `
                    <div class="detection-item">
                        <span class="label">${d.label}</span>
                        <span class="confidence">${(d.confidence * 100).toFixed(0)}%</span>
                    </div>
                `).join('');
            } else {
                detList.innerHTML = '<div style="color: #666; text-align: center; padding: 10px;">No objects detected</div>';
            }

            // Update commands
            const cmdList = document.getElementById('commandsList');
            if (data.commands.length > 0) {
                cmdList.innerHTML = data.commands.map(c => `
                    <div class="command-item">${c}</div>
                `).join('');
                log(`Executed: ${data.commands.join(', ')}`, 'info');
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


# Routes
@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


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
    if robot:
        robot.stop()
    return jsonify({'status': 'stopped'})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    system_state['paused'] = not system_state['paused']
    return jsonify({'status': 'paused' if system_state['paused'] else 'resumed'})


@app.route('/api/task', methods=['POST'])
def api_task():
    data = request.json
    system_state['task'] = data.get('task', 'Explore')
    return jsonify({'status': 'ok', 'task': system_state['task']})


@app.route('/api/command', methods=['POST'])
def api_command():
    data = request.json
    cmd = data.get('command', '')
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


if __name__ == '__main__':
    import argparse
    import ssl
    import os

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
