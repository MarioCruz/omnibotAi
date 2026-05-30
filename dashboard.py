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

# Task logging — captures every decision cycle when a mission is active.
# Rotates at 5MB with 3 backups so the SD card doesn't fill up on a long run.
from logging.handlers import RotatingFileHandler

task_logger = logging.getLogger('task')
task_logger.setLevel(logging.DEBUG)
task_logger.propagate = False  # Don't duplicate to root/stdout
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_log_dir, exist_ok=True)
_task_handler = RotatingFileHandler(
    os.path.join(_log_dir, 'task.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
)
_task_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
task_logger.addHandler(_task_handler)
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        return
    loaded = 0
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            # Strip surrounding matching quotes so FOO="bar" and FOO='bar' work.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ[key] = value
            loaded += 1
    # Log the count only — never the names, since they often reveal which
    # third-party services (Groq, Gemini, etc.) are in use.
    if loaded:
        print(f"[Env] Loaded {loaded} variable(s) from .env")


load_env()

# Import our modules
from camera_capture import CameraCapture
from object_detector import ObjectDetector
from navigation import NavigationEngine
from robot_executor import RobotCommandExecutor
from eye_display import EyeDisplay


def load_config():
    """Load robot configuration from config.json.

    A malformed or unreadable config file falls back to defaults instead of
    crashing startup — otherwise systemd would flap the service on a bad
    deploy before anyone could fix it.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
            if not isinstance(cfg, dict):
                print(f"[Config] {config_path} is not a JSON object, using defaults")
                return {}
            return cfg
    except (json.JSONDecodeError, OSError) as e:
        print(f"[Config] ERROR reading {config_path}: {e} — using defaults")
        return {}


def _cfg_int(cfg, key, default):
    """Read an int from config, falling back to default on any type error."""
    v = cfg.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        print(f"[Config] {key}={v!r} is not an int, using default {default}")
        return default


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
    'arrived_announced': False,  # True once we've spoken "found it" for the current task
    'detections': [],
    'last_commands': [],
    'stats': {
        'iterations': 0,
        'total_detections': 0,
        'total_commands': 0,
        'fps': 0
    },
    # Collected during init_system(). Non-empty means /healthz reports degraded
    # so an operator can see what's wrong without reading journalctl.
    'init_errors': [],
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
# Set once after the first frame is encoded. Stays set. New MJPEG clients
# wait on this before entering the steady-state 30fps polling loop so they
# don't busy-spin during cold start.
first_frame_event = threading.Event()

# Health / watchdog state
process_start_time = time.time()
last_detection_time = 0.0  # Wall-clock of most recent non-empty detection

# Cached Bluetooth status — the actual bluetoothctl call is done in a
# background thread (see bt_poll_loop) so /api/bluetooth never blocks
# the Flask worker on a hung BT stack. Kids dashboard polls this heavily.
bt_cache_lock = threading.Lock()
bt_cache = {'connected': False, 'devices': [], 'updated_at': 0.0}
BT_POLL_INTERVAL = 5.0  # Seconds between bluetoothctl probes

# Cached internet reachability — refreshed by internet_poll_loop. Used by
# /api/describe to skip the Groq call (and its multi-second TCP timeout) when
# WiFi is dead at a venue. Probes 1.1.1.1:53 because DNS-bypass + tiny payload.
internet_cache_lock = threading.Lock()
internet_cache = {'alive': False, 'updated_at': 0.0}
INTERNET_POLL_INTERVAL = 15.0
INTERNET_PROBE_TIMEOUT = 1.5  # seconds; bounded so the probe thread can't hang

# If the camera is stuck this long, exit 1 so systemd restarts the process.
# Below this threshold, /healthz reports "degraded" but stays alive.
CAMERA_STALE_DEGRADED_SEC = 10
CAMERA_STALE_FATAL_SEC = 60


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
                eye_brightness=15, eye_rotation=0, eye_offset_x=0, eye_offset_y=0,
                step_duration=500, turn_duration=750, nudge_duration=300):
    """Initialize all system components"""
    global camera, detector, robot

    print("[Dashboard] Initializing system...")

    # Detector first (for IMX500, we need it before camera to get the imx500 instance)
    print(f"[Dashboard] Loading detector ({detector_backend})...")
    try:
        detector = ObjectDetector(backend=detector_backend)
    except Exception as e:
        print(f"[Dashboard] Detector error: {e}")
        system_state['init_errors'].append(f"detector: {e}")
        detector = None

    # Camera - pass IMX500 instance and intrinsics for hardware-accelerated detection
    print("[Dashboard] Starting camera...")
    imx500_instance = detector.get_imx500() if detector and detector_backend == 'imx500' else None
    intrinsics = detector.get_intrinsics() if detector and detector_backend == 'imx500' else None
    camera_resolution = (640, 480)
    try:
        camera = CameraCapture(resolution=camera_resolution, framerate=30, imx500=imx500_instance, intrinsics=intrinsics)
    except Exception as e:
        print(f"[Dashboard] Camera error: {e}")
        system_state['init_errors'].append(f"camera: {e}")
        camera = None

    # Pass picam2 instance to detector for accurate coordinate conversion
    if detector and detector_backend == 'imx500' and camera and hasattr(camera, 'picam2'):
        detector.set_picam2(camera.picam2)

    # Navigation engine - rule-based, no LLM needed
    global nav
    print("[Dashboard] Initializing navigation engine...")
    nav = NavigationEngine(
        frame_width=camera_resolution[0],
        frame_height=camera_resolution[1]
    )

    # Robot - now uses audio tones for Tomy Omnibot
    print(f"[Dashboard] Initializing audio commander "
          f"(volume={volume}, step={step_duration}ms, turn={turn_duration}ms, nudge={nudge_duration}ms)...")
    try:
        robot = RobotCommandExecutor(
            volume=volume,
            step_duration=step_duration,
            turn_duration=turn_duration,
            nudge_duration=nudge_duration,
        )
        if not robot.connect():
            system_state['init_errors'].append("robot: connect() returned False")
    except Exception as e:
        print(f"[Dashboard] Robot error: {e}")
        system_state['init_errors'].append(f"robot: {e}")
        robot = None

    # Eye display - configurable via config.json: st7735, ssd1351, or none
    global eye_display
    if eye_display_type == 'none':
        print("[Dashboard] Eye display disabled")
        eye_display = None
    else:
        print(f"[Dashboard] Initializing eye display ({eye_display_type})...")
        partially_initialized = None
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
            # If start() throws after __init__ opened the SPI handle, we
            # need to tear the partial object down so the bus isn't leaked.
            partially_initialized = eye_display
            eye_display.start()
            partially_initialized = None  # Fully initialized, no cleanup needed
            print(f"[Dashboard] Eye display started ({eye_display_type})")
        except Exception as e:
            print(f"[Dashboard] Eye display not available: {e}")
            system_state['init_errors'].append(f"eye: {e}")
            if partially_initialized is not None:
                try:
                    partially_initialized.stop()
                except Exception:
                    pass
            eye_display = None

    if system_state['init_errors']:
        print(f"[Dashboard] Init completed with {len(system_state['init_errors'])} error(s); /healthz will report degraded")
    else:
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

            # Detect a dead/stuck capture thread — warn at most once per 10s,
            # and if it stays stuck past CAMERA_STALE_FATAL_SEC, exit so
            # systemd restarts the whole process cleanly.
            age = camera.frame_age() if hasattr(camera, 'frame_age') else None
            if age is not None and age > CAMERA_STALE_DEGRADED_SEC and time.time() - last_stale_warn > 10.0:
                print(f"[Dashboard] Warning: camera frame is {age:.1f}s old — capture thread may be stuck")
                last_stale_warn = time.time()
            if age is not None and age > CAMERA_STALE_FATAL_SEC:
                print(f"[Dashboard] FATAL: camera frame {age:.1f}s old, exiting for systemd restart")
                os._exit(1)

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
                    global last_detection_time
                    last_detection_time = time.time()

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

                    # Announce arrival once per task when nav decides we're close enough.
                    if 'step("stop")' in commands and not system_state['arrived_announced']:
                        commands.append('phrase("found_it")')
                        system_state['arrived_announced'] = True
                        task_logger.info("ARRIVED — announcing found_it")

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
    """Background thread that encodes JPEG once for all clients.

    Always encodes when there's a camera frame, regardless of system_state
    ['running']. The camera thread runs continuously after init_system, and
    operators should see the live feed the moment they open the dashboard
    (even before clicking Start). Sets first_frame_event once cached bytes
    are ready so generate_mjpeg unblocks.
    """
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
                    first_frame_event.set()
            except Exception as e:
                print(f"[MJPEG] Encode error: {e}")

        time.sleep(0.033)  # ~30 FPS


def generate_mjpeg():
    """Generate MJPEG stream — block on first_frame_event during cold start.

    Before the encoder has ever produced a frame, wait on an Event instead
    of busy-looping at 30Hz yielding nothing. Once any frame exists we go
    into steady-state 30fps polling; the event stays set, so multiple
    clients all proceed independently.
    """
    while not system_state['shutdown']:
        # Cold-start wait (cheap once set).
        if not first_frame_event.wait(timeout=1.0):
            continue

        with mjpeg_lock:
            frame_bytes = cached_mjpeg_bytes
        if frame_bytes is not None:
            yield frame_bytes

        time.sleep(0.033)  # ~30 FPS


# HTML dashboards live in templates/ (extracted from this file to keep
# dashboard.py focused on logic). Loaded once at import; served via
# render_template_string so existing routes are unchanged.
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


def _load_template(name):
    with open(os.path.join(_TEMPLATE_DIR, name), encoding='utf-8') as f:
        return f.read()


DASHBOARD_HTML = _load_template('dashboard.html')
KIDS_DASHBOARD_HTML = _load_template('kids.html')


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
    system_state['arrived_announced'] = False
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
    system_state['arrived_announced'] = False
    task_logger.info(f"=== TASK SET: {cleaned} ===")
    return jsonify({'status': 'ok', 'task': cleaned})


@app.route('/api/task/end', methods=['POST'])
def api_task_end():
    """End the current task without stopping the system (detection continues)."""
    if system_state['task']:
        task_logger.info(f"=== TASK ENDED: {system_state['task']} ===")
    system_state['task'] = None
    system_state['arrived_announced'] = False
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

    # Try Groq LLM for a natural description — but only if we have an API key
    # AND the background probe thinks the internet is reachable. Skipping the
    # call when offline avoids the TCP-connect timeout and gives the kid an
    # immediate (local) answer instead of a frozen UI.
    api_key = os.environ.get('GROQ_API_KEY')
    description = None
    with internet_cache_lock:
        net_alive = internet_cache['alive']
        net_age = time.time() - internet_cache['updated_at'] if internet_cache['updated_at'] else None
    # If the probe hasn't run yet (age=None) we tentatively try Groq once;
    # the short timeout below caps the worst case. If the probe is fresh and
    # says offline, we skip outright.
    can_call_groq = bool(api_key) and (net_alive or net_age is None)
    if not can_call_groq and api_key:
        print(f"[Describe] Skipping Groq — internet probe says offline (age={net_age})")

    if can_call_groq:
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
                timeout=3
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


def _eye_react_to_command(cmd_lower):
    """Update eye expression to match a command. Safe no-op if eye is disabled."""
    if not eye_display:
        return
    mark_activity()
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


@app.route('/api/command', methods=['POST'])
def api_command():
    data = request.json or {}
    cmd = data.get('command', '')
    if not cmd:
        return jsonify({'status': 'error', 'message': 'No command provided'})

    cmd_lower = cmd.lower()

    if not (robot and robot.connected):
        return jsonify({'status': 'error', 'message': 'Robot not connected'})

    # Preempt path: speaker_off and stop must interrupt whatever is running.
    # Checking executing_lock would make the Stop button useless mid-pattern.
    if cmd_lower == 'speaker_off':
        _eye_react_to_command(cmd_lower)
        if robot.audio:
            robot.audio.stop_speaking()
            return jsonify({'status': 'ok', 'result': True, 'message': 'Speaker off sent'})
        return jsonify({'status': 'error', 'message': 'Audio not available'})
    if cmd_lower == 'stop':
        _eye_react_to_command(cmd_lower)
        try:
            robot.stop()  # sets _cancel_pattern + audio.stop() — reapable mid-pattern
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'stop failed: {e}'}), 500
        return jsonify({'status': 'ok', 'result': True, 'message': 'Stopped'})

    # Everything else runs asynchronously under executing_lock so the Flask
    # request doesn't block on patterns (~5-10s) or speech (up to 30s) and
    # can't race with process_loop's autonomous dispatch.
    global is_executing
    with executing_lock:
        if is_executing:
            return jsonify({'status': 'busy', 'message': 'Robot executing another command'}), 409
        is_executing = True

    def _run_manual(c):
        global is_executing
        try:
            robot.execute(c)
        finally:
            with executing_lock:
                is_executing = False

    # Update the eye only once we're committed to dispatching — keeps the eye
    # from jerking around during a click storm that returns 409.
    _eye_react_to_command(cmd_lower)

    try:
        threading.Thread(target=_run_manual, args=(cmd,), daemon=True).start()
    except Exception as e:
        # Thread creation failed (extremely rare) — release the slot we reserved
        # so the next command isn't stuck at 409 forever.
        with executing_lock:
            is_executing = False
        return jsonify({'status': 'error', 'message': f'dispatch failed: {e}'}), 500

    return jsonify({'status': 'queued', 'command': cmd})


@app.route('/api/status', methods=['GET'])
def api_status():
    # Whitelist fields so future additions to system_state don't leak by
    # accident. Keep the response compact — dashboard JS just needs these.
    return jsonify({
        'running': bool(system_state.get('running')),
        'paused': bool(system_state.get('paused')),
        'task': system_state.get('task'),
        'detections': system_state.get('detections', [])[:20],
        'last_commands': system_state.get('last_commands', [])[:20],
        'stats': system_state.get('stats', {}),
        'init_errors': list(system_state.get('init_errors', [])),
    })


def _poll_bluetooth_once():
    """Shell out to bluetoothctl and return (connected, device_names)."""
    try:
        result = subprocess.run(
            ['bluetoothctl', 'devices', 'Connected'],
            capture_output=True, text=True, timeout=3,
        )
        devices = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        names = []
        for dev in devices:
            # Format: "Device XX:XX:XX:XX:XX:XX Name"
            parts = dev.split(' ', 2)
            if len(parts) >= 3:
                names.append(parts[2])
        return (len(devices) > 0, names)
    except Exception:
        return (False, [])


def _poll_internet_once():
    """Open a tiny TCP connection to 1.1.1.1:53 with a hard timeout."""
    import socket
    s = None
    try:
        s = socket.create_connection(("1.1.1.1", 53), timeout=INTERNET_PROBE_TIMEOUT)
        return True
    except Exception:
        return False
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


def internet_poll_loop():
    """Refresh internet-reachability flag every INTERNET_POLL_INTERVAL seconds."""
    while not system_state['shutdown']:
        alive = _poll_internet_once()
        with internet_cache_lock:
            internet_cache['alive'] = alive
            internet_cache['updated_at'] = time.time()
        # Sleep in small chunks so shutdown is responsive.
        for _ in range(int(INTERNET_POLL_INTERVAL * 5)):
            if system_state['shutdown']:
                return
            time.sleep(0.2)


def bt_poll_loop():
    """Refresh Bluetooth status in the background every BT_POLL_INTERVAL seconds."""
    while not system_state['shutdown']:
        connected, names = _poll_bluetooth_once()
        with bt_cache_lock:
            bt_cache['connected'] = connected
            bt_cache['devices'] = names
            bt_cache['updated_at'] = time.time()
        # Sleep in small chunks so shutdown is responsive.
        for _ in range(int(BT_POLL_INTERVAL * 5)):
            if system_state['shutdown']:
                return
            time.sleep(0.2)


@app.route('/api/bluetooth', methods=['GET'])
def api_bluetooth():
    """Return the last cached Bluetooth status — never blocks on subprocess."""
    with bt_cache_lock:
        age = time.time() - bt_cache['updated_at'] if bt_cache['updated_at'] else None
        return jsonify({
            'connected': bt_cache['connected'],
            'devices': list(bt_cache['devices']),
            'age_seconds': round(age, 1) if age is not None else None,
        })


@app.route('/health')
@app.route('/healthz')
def health():
    """Structured health snapshot for monitors and the dashboard badge.

    Returns 200 when ok, 503 when any subsystem is degraded (camera stuck,
    robot disconnected during an active task, or the process loop is idle).
    """
    now = time.time()
    cam_age = camera.frame_age() if (camera and hasattr(camera, 'frame_age')) else None
    eye_alive = bool(eye_display and getattr(eye_display, 'running', False))
    robot_connected = bool(robot and robot.connected)
    last_det_ago = (now - last_detection_time) if last_detection_time else None

    subsystems = {
        'camera': {
            'age_seconds': round(cam_age, 2) if cam_age is not None else None,
            'fps': camera.get_fps() if camera else 0,
            'stale': cam_age is not None and cam_age > CAMERA_STALE_DEGRADED_SEC,
        },
        'robot': {
            'connected': robot_connected,
        },
        'eye': {
            'alive': eye_alive,
        },
        'process': {
            'uptime_seconds': round(now - process_start_time, 1),
            'running': bool(system_state.get('running')),
            'paused': bool(system_state.get('paused')),
            'task': system_state.get('task'),
        },
        'detection': {
            'last_ago_seconds': round(last_det_ago, 1) if last_det_ago is not None else None,
        },
    }

    # Degraded if any core subsystem is unhealthy during active operation.
    degraded = False
    reasons = []
    if subsystems['camera']['stale']:
        degraded = True
        reasons.append(f"camera stale {subsystems['camera']['age_seconds']}s")
    if system_state.get('task') and not robot_connected:
        degraded = True
        reasons.append("robot disconnected during active task")
    init_errors = system_state.get('init_errors', [])
    if init_errors:
        degraded = True
        for err in init_errors:
            reasons.append(f"init: {err}")

    payload = {
        'status': 'degraded' if degraded else 'ok',
        'reasons': reasons,
        'subsystems': subsystems,
    }
    return jsonify(payload), (503 if degraded else 200)


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
        eye_dc_pin=_cfg_int(robot_config, 'eye_dc_pin', 24),
        eye_rst_pin=_cfg_int(robot_config, 'eye_rst_pin', 25),
        eye_cs_pin=_cfg_int(robot_config, 'eye_cs_pin', 0),
        eye_spi_port=_cfg_int(robot_config, 'eye_spi_port', 0),
        eye_brightness=_cfg_int(robot_config, 'eye_brightness', 15),
        eye_rotation=_cfg_int(robot_config, 'eye_rotation', 0),
        eye_offset_x=_cfg_int(robot_config, 'eye_offset_x', 0),
        eye_offset_y=_cfg_int(robot_config, 'eye_offset_y', 0),
        step_duration=_cfg_int(robot_config, 'step_duration', 500),
        turn_duration=_cfg_int(robot_config, 'turn_duration', 750),
        nudge_duration=_cfg_int(robot_config, 'nudge_duration', 300),
    )

    # Start processing thread
    process_thread = threading.Thread(target=process_loop, daemon=True)
    process_thread.start()

    # Start MJPEG encoder thread (encodes once, shared by all clients)
    encoder_thread = threading.Thread(target=mjpeg_encoder_loop, daemon=True)
    encoder_thread.start()

    # Start internet reachability probe so /api/describe can skip Groq when
    # WiFi is down without paying a TCP timeout.
    internet_thread = threading.Thread(target=internet_poll_loop, daemon=True)
    internet_thread.start()

    # Start Bluetooth poll thread (keeps /api/bluetooth non-blocking)
    bt_thread = threading.Thread(target=bt_poll_loop, daemon=True)
    bt_thread.start()

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
