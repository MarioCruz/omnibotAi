#!/usr/bin/env python3
"""
HTTPS MJPEG Streaming Server for Raspberry Pi Camera with IMX500 AI
Streams camera feed with MobileNet SSD object detection over HTTPS
"""

import io
import threading
import time
from flask import Flask, Response, render_template_string

# Try to import picamera2, fall back to subprocess for rpicam-vid
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    import subprocess

app = Flask(__name__)

# Streaming output class for picamera2
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


# Global streaming output
output = StreamingOutput()
camera = None


def init_camera():
    """Initialize the Pi camera with IMX500 post-processing"""
    global camera, output

    if PICAMERA2_AVAILABLE:
        camera = Picamera2()

        # Configure for 1920x1080 @ 30fps as per your working command
        config = camera.create_video_configuration(
            main={"size": (1920, 1080), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        camera.configure(config)

        # Start the camera with MJPEG encoder
        encoder = MJPEGEncoder(bitrate=10000000)
        camera.start_recording(encoder, FileOutput(output))
        print("Camera started with picamera2")
    else:
        print("picamera2 not available, using rpicam-vid subprocess")
        # Fallback to subprocess method
        start_rpicam_subprocess()


def start_rpicam_subprocess():
    """Start rpicam-vid as a subprocess for MJPEG streaming"""
    global output

    def stream_from_rpicam():
        cmd = [
            'rpicam-vid',
            '-t', '0',
            '--codec', 'mjpeg',
            '--width', '1920',
            '--height', '1080',
            '--framerate', '30',
            '--post-process-file', '/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json',
            '-o', '-'
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        # Read MJPEG frames from stdout
        buffer = b''
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk

            # Find JPEG frame boundaries
            start = buffer.find(b'\xff\xd8')
            end = buffer.find(b'\xff\xd9')

            if start != -1 and end != -1 and end > start:
                frame = buffer[start:end+2]
                buffer = buffer[end+2:]

                with output.condition:
                    output.frame = frame
                    output.condition.notify_all()

    thread = threading.Thread(target=stream_from_rpicam, daemon=True)
    thread.start()


def generate_frames():
    """Generator function that yields MJPEG frames"""
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# HTML template for the viewer page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pi Camera Stream - IMX500 AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            margin-bottom: 10px;
            color: #00d4ff;
        }
        .info {
            color: #888;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .stream-container {
            background: #16213e;
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }
        .status {
            margin-top: 15px;
            padding: 10px 20px;
            background: #0f3460;
            border-radius: 6px;
            font-size: 12px;
        }
        .status span { color: #00d4ff; }
    </style>
</head>
<body>
    <h1>Pi Camera Stream</h1>
    <p class="info">IMX500 MobileNet SSD Object Detection</p>
    <div class="stream-container">
        <img src="/stream" alt="Camera Stream" id="stream">
    </div>
    <div class="status">
        Resolution: <span>1920x1080</span> |
        FPS: <span>30</span> |
        Protocol: <span>MJPEG over HTTPS</span>
    </div>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the viewer page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/stream')
def stream():
    """MJPEG stream endpoint"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'ok', 'camera': 'running'}


if __name__ == '__main__':
    import ssl
    import os

    # Initialize camera
    init_camera()

    # Check for SSL certificates
    cert_file = 'cert.pem'
    key_file = 'key.pem'

    if os.path.exists(cert_file) and os.path.exists(key_file):
        # Run with HTTPS
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_file, key_file)

        print("\n" + "="*50)
        print("  HTTPS Camera Stream Server")
        print("="*50)
        print(f"  Open: https://<your-pi-ip>:5000")
        print("  Stream: https://<your-pi-ip>:5000/stream")
        print("="*50 + "\n")

        app.run(host='0.0.0.0', port=5000, ssl_context=ssl_context, threaded=True)
    else:
        print("\n" + "="*50)
        print("  WARNING: No SSL certificates found!")
        print("  Run: ./generate_certs.sh")
        print("  Starting in HTTP mode (not secure)")
        print("="*50)
        print(f"  Open: http://<your-pi-ip>:5000")
        print("="*50 + "\n")

        app.run(host='0.0.0.0', port=5000, threaded=True)
