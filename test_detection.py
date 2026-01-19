#!/usr/bin/env python3
"""
IMX500 Detection Test - Using official picamera2 approach
Based on: https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py
"""

import time
import threading
import ssl
import os
import sys
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# Configuration - Choose model:
# MobileNet SSD (fast): imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
# YOLOv8 nano (accurate): imx500_network_yolov8n_pp.rpk
# YOLO11 nano (latest): imx500_network_yolo11n_pp.rpk
# NanoDet Plus: imx500_network_nanodet_plus_416x416_pp.rpk
MODEL_PATH = '/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk'
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.65
MAX_DETECTIONS = 10

# COCO labels
LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Global state
picam2 = None
imx500 = None
intrinsics = None
last_detections = []
last_results = []
current_frame = None
frame_lock = threading.Lock()
fps = 0
detection_count = 0
current_detections = []
detection_history = []
inference_time_ms = 0
running = True

app = Flask(__name__)


class Detection:
    """Detection result with bounding box, category, and confidence."""
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        # Convert inference coordinates to ISP output coordinates
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


@lru_cache
def get_labels():
    """Get labels, filtering out dash placeholders if needed."""
    labels = LABELS
    if intrinsics and hasattr(intrinsics, 'ignore_dash_labels') and intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def parse_detections(metadata):
    """Parse the output tensor into detected objects, scaled to ISP output."""
    global last_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    input_w, input_h = imx500.get_input_size()

    # Check if using nanodet postprocessing
    if intrinsics and intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=CONFIDENCE_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            max_out_dets=MAX_DETECTIONS
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        # Standard SSD output format
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        # Handle bbox normalization
        bbox_normalization = intrinsics.bbox_normalization if intrinsics else True
        if bbox_normalization:
            boxes = boxes / input_h

        # Handle bbox order (xy vs yx)
        bbox_order = intrinsics.bbox_order if intrinsics else "yx"
        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > CONFIDENCE_THRESHOLD
    ]
    return last_detections


def draw_detections(request, stream="main"):
    """Draw detections onto the camera frame using MappedArray."""
    global last_results

    detections = last_results
    if detections is None or len(detections) == 0:
        return

    labels = get_labels()

    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            x, y, w, h = int(x), int(y), int(w), int(h)

            label_text = f"{labels[int(detection.category)]} ({detection.conf:.0%})"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x + 5
            text_y = y + 15

            # Semi-transparent background
            overlay = m.array.copy()
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height - 2),
                          (text_x + text_width + 2, text_y + baseline + 2),
                          (255, 255, 255),
                          cv2.FILLED)
            cv2.addWeighted(overlay, 0.3, m.array, 0.7, 0, m.array)

            # Draw label text
            cv2.putText(m.array, label_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw bounding box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)


def capture_loop():
    """Main capture and detection loop."""
    global current_frame, fps, detection_count, current_detections, detection_history
    global inference_time_ms, running, last_results
    from datetime import datetime

    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    last_logged = {}

    while running:
        try:
            # Capture metadata and parse detections
            metadata = picam2.capture_metadata()
            last_results = parse_detections(metadata)

            # Get hardware inference timing
            try:
                kpi = imx500.get_kpi_info(metadata)
                if kpi:
                    inference_time_ms = kpi[0]
            except:
                pass

            # Capture frame with detections drawn
            frame = picam2.capture_array("main")
            frame_count += 1

            # Convert detections to API format
            labels = get_labels()
            detections = []
            for det in last_results:
                x, y, w, h = det.box
                detections.append({
                    'label': labels[int(det.category)] if int(det.category) < len(labels) else f"class_{int(det.category)}",
                    'score': float(det.conf),
                    'box': (int(x), int(y), int(x + w), int(y + h))
                })

            detection_count = len(detections)
            current_detections = detections

            # Log detections to history (with 2-second cooldown per label)
            now = time.time()
            for det in detections:
                label = det['label']
                if label not in last_logged or (now - last_logged[label]) > 2.0:
                    last_logged[label] = now
                    detection_history.insert(0, {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'label': label,
                        'score': det['score']
                    })
                    if len(detection_history) > 100:
                        detection_history.pop()

            # Calculate FPS
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - start_time)
                last_fps_time = now

            # Draw FPS overlay
            cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {detection_count} | Inf: {inference_time_ms:.1f}ms",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            with frame_lock:
                current_frame = frame

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


def generate_mjpeg():
    """Generate MJPEG stream for web viewer."""
    global current_frame, running

    while running:
        with frame_lock:
            frame = current_frame

        if frame is not None:
            try:
                # Convert RGB to BGR for JPEG encoding
                _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except:
                pass

        time.sleep(0.033)


# HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IMX500 Detection Test (picamera2)</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a2e; color: #fff; font-family: monospace; margin: 0; padding: 20px; }
        h1 { color: #00d4ff; text-align: center; margin-bottom: 20px; }
        .subtitle { color: #888; text-align: center; margin-top: -15px; margin-bottom: 20px; font-size: 12px; }
        .container { display: flex; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .video-panel { flex: 2; }
        .video-panel img { width: 100%; border: 2px solid #00d4ff; border-radius: 8px; }
        .side-panels { flex: 1; display: flex; flex-direction: column; gap: 15px; min-width: 300px; }
        .panel { background: #16213e; border: 2px solid #00d4ff; border-radius: 8px; padding: 15px; }
        .panel h2 { color: #00d4ff; margin-top: 0; font-size: 16px; border-bottom: 1px solid #00d4ff; padding-bottom: 10px; }
        .stats { background: #0f3460; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
        .stats span { display: block; margin: 5px 0; }
        .fps { color: #00ff88; font-size: 18px; }
        .count { color: #ffaa00; }
        .detection-list { max-height: 200px; overflow-y: auto; }
        .detection-item { background: #0f3460; margin: 8px 0; padding: 10px; border-radius: 5px; border-left: 3px solid #00ff88; }
        .detection-item .label { color: #00ff88; font-weight: bold; font-size: 14px; }
        .detection-item .confidence { color: #ffaa00; font-size: 12px; }
        .detection-item .coords { color: #888; font-size: 11px; margin-top: 5px; }
        .no-detections { color: #666; font-style: italic; text-align: center; padding: 20px; }
        .history-list { max-height: 300px; overflow-y: auto; }
        .history-item { background: #0f3460; margin: 5px 0; padding: 8px; border-radius: 5px; font-size: 12px; border-left: 3px solid #00d4ff; }
        .history-item .time { color: #00d4ff; font-size: 10px; }
        .history-item .label { color: #00ff88; }
        .history-item .conf { color: #ffaa00; }
        .history-header { display: flex; justify-content: space-between; align-items: center; }
        .clear-btn { background: #e74c3c; color: #fff; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; }
        .clear-btn:hover { background: #c0392b; }
        .total-count { color: #888; font-size: 12px; margin-top: 10px; }
        @media (max-width: 900px) { .container { flex-direction: column; } }
    </style>
</head>
<body>
    <h1>IMX500 Detection Test</h1>
    <p class="subtitle">Using official picamera2 API (convert_inference_coords)</p>
    <div class="container">
        <div class="video-panel">
            <img src="/stream" alt="Camera Stream">
        </div>
        <div class="side-panels">
            <div class="panel">
                <h2>Live Detections</h2>
                <div class="stats">
                    <span class="fps">FPS: <span id="fps">--</span></span>
                    <span class="count">Objects: <span id="count">0</span></span>
                    <span style="color:#ff6b6b;">Inference: <span id="inference">--</span>ms</span>
                </div>
                <div class="detection-list" id="detections">
                    <div class="no-detections">Waiting for detections...</div>
                </div>
            </div>
            <div class="panel">
                <div class="history-header">
                    <h2 style="border:none; padding:0; margin:0;">Detection Log</h2>
                    <button class="clear-btn" onclick="clearHistory()">Clear</button>
                </div>
                <div class="history-list" id="history">
                    <div class="no-detections">No detection history yet...</div>
                </div>
                <div class="total-count">Total detections: <span id="total">0</span></div>
            </div>
        </div>
    </div>
    <script>
        function updateDetections() {
            fetch('/api/detections')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('count').textContent = data.count;
                    document.getElementById('inference').textContent = data.inference_ms.toFixed(1);

                    const list = document.getElementById('detections');
                    if (data.detections.length === 0) {
                        list.innerHTML = '<div class="no-detections">No objects detected</div>';
                    } else {
                        list.innerHTML = data.detections.map(d => `
                            <div class="detection-item">
                                <div class="label">${d.label}</div>
                                <div class="confidence">${(d.score * 100).toFixed(0)}% confidence</div>
                                <div class="coords">Box: (${d.box[0]}, ${d.box[1]}) to (${d.box[2]}, ${d.box[3]})</div>
                            </div>
                        `).join('');
                    }
                })
                .catch(err => console.error('Update error:', err));
        }

        function updateHistory() {
            fetch('/api/history')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('total').textContent = data.total;

                    const list = document.getElementById('history');
                    if (data.history.length === 0) {
                        list.innerHTML = '<div class="no-detections">No detection history yet...</div>';
                    } else {
                        list.innerHTML = data.history.map(h => `
                            <div class="history-item">
                                <span class="time">${h.time}</span>
                                <span class="label">${h.label}</span>
                                <span class="conf">(${(h.score * 100).toFixed(0)}%)</span>
                            </div>
                        `).join('');
                        list.scrollTop = 0;
                    }
                })
                .catch(err => console.error('History error:', err));
        }

        function clearHistory() {
            fetch('/api/history/clear', {method: 'POST'})
                .then(() => updateHistory())
                .catch(err => console.error('Clear error:', err));
        }

        setInterval(updateDetections, 500);
        setInterval(updateHistory, 1000);
        updateDetections();
        updateHistory();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/stream')
def stream():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/detections')
def api_detections():
    return jsonify({
        'fps': fps,
        'count': detection_count,
        'inference_ms': inference_time_ms,
        'detections': current_detections
    })


@app.route('/api/history')
def api_history():
    return jsonify({
        'total': len(detection_history),
        'history': detection_history[:50]
    })


@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    global detection_history
    detection_history = []
    return jsonify({'status': 'cleared'})


def main():
    global picam2, imx500, intrinsics, running

    print("=" * 50)
    print("  IMX500 Detection Test (picamera2 Official API)")
    print("=" * 50)

    # Initialize IMX500
    print(f"Loading model: {MODEL_PATH}")
    imx500 = IMX500(MODEL_PATH)

    # Get network intrinsics
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit(1)

    # Set labels
    intrinsics.labels = LABELS
    intrinsics.update_with_defaults()

    print(f"Model task: {intrinsics.task}")
    print(f"Inference rate: {intrinsics.inference_rate} fps")

    # Initialize camera
    print("Starting camera...")
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (640, 480)},
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12
    )
    picam2.configure(config)

    # Show model loading progress
    imx500.show_network_fw_progress_bar()

    # Set pre_callback for drawing detections directly on frames
    picam2.pre_callback = draw_detections

    picam2.start()
    print("Camera started.")

    # Start capture thread
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # Generate self-signed SSL cert if needed
    cert_file = '/tmp/test_cert.pem'
    key_file = '/tmp/test_key.pem'

    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("Generating self-signed SSL certificate...")
        os.system(f'openssl req -x509 -newkey rsa:2048 -keyout {key_file} -out {cert_file} -days 365 -nodes -subj "/CN=omniai.local"')

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, key_file)

    print("-" * 50)
    print("Open in browser: https://omniai.local:8080")
    print("(Accept the self-signed certificate warning)")
    print("-" * 50)

    try:
        app.run(host='0.0.0.0', port=8080, threaded=True, ssl_context=ssl_context)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        picam2.stop()
        print("Done.")


if __name__ == '__main__':
    main()
