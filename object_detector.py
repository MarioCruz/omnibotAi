#!/usr/bin/env python3
"""
Object Detection Module
Supports multiple backends: IMX500 (native), YOLOv5, MediaPipe
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import subprocess
import json
import threading
import time


class ObjectDetector:
    """Base object detector with multiple backend support"""

    def __init__(self, backend='imx500', model_path=None, confidence_threshold=0.5):
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path

        # Detection results cache for IMX500
        self._detections = []
        self._detections_lock = threading.Lock()

        if backend == 'imx500':
            self._init_imx500()
        elif backend == 'yolov5':
            self._init_yolov5()
        elif backend == 'mediapipe':
            self._init_mediapipe()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"[Detector] Initialized with backend: {backend}")

    def _init_imx500(self):
        """Initialize IMX500 native detection via rpicam-detect"""
        self.imx500_process = None
        self.imx500_running = False

        # COCO class labels for MobileNet SSD
        self.labels = [
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

    def _init_yolov5(self):
        """Initialize YOLOv5 detection"""
        try:
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = self.confidence_threshold
            print("[Detector] YOLOv5 loaded successfully")
        except ImportError:
            raise ImportError("YOLOv5 requires: pip install yolov5 torch")

    def _init_mediapipe(self):
        """Initialize MediaPipe object detection"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = self.model_path or 'efficientdet_lite0.tflite'

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=self.confidence_threshold
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            self.mp_image = mp.Image
            print("[Detector] MediaPipe loaded successfully")
        except ImportError:
            raise ImportError("MediaPipe requires: pip install mediapipe")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame

        Args:
            frame: BGR or RGB numpy array

        Returns:
            List of detections with keys: label, confidence, bbox
        """
        if self.backend == 'imx500':
            return self._detect_imx500(frame)
        elif self.backend == 'yolov5':
            return self._detect_yolov5(frame)
        elif self.backend == 'mediapipe':
            return self._detect_mediapipe(frame)

    def _detect_imx500(self, frame: np.ndarray) -> List[Dict]:
        """
        For IMX500, detection happens on-chip.
        This returns cached detections from the IMX500 output parser.
        Call start_imx500_stream() to begin detection.
        """
        with self._detections_lock:
            return self._detections.copy()

    def start_imx500_stream(self, callback=None):
        """
        Start IMX500 detection stream using rpicam-detect

        The IMX500 runs detection on-chip and outputs results via metadata.
        This starts a subprocess that parses detection output.
        """
        if self.imx500_running:
            return

        def parse_imx500_output():
            cmd = [
                'rpicam-hello',
                '-t', '0',
                '--post-process-file', '/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json',
                '--lores-width', '640',
                '--lores-height', '480',
                '-n',  # No preview window
                '--metadata', '-'  # Output metadata to stdout
            ]

            try:
                self.imx500_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self.imx500_running = True

                for line in self.imx500_process.stdout:
                    if not self.imx500_running:
                        break

                    # Parse detection metadata
                    if 'detection' in line.lower() or 'object' in line.lower():
                        detections = self._parse_imx500_line(line)
                        with self._detections_lock:
                            self._detections = detections
                        if callback:
                            callback(detections)

            except Exception as e:
                print(f"[Detector] IMX500 error: {e}")
            finally:
                self.imx500_running = False

        thread = threading.Thread(target=parse_imx500_output, daemon=True)
        thread.start()

    def _parse_imx500_line(self, line: str) -> List[Dict]:
        """Parse IMX500 metadata output line"""
        detections = []
        try:
            # IMX500 outputs JSON-like detection data
            # Format varies by model, this handles MobileNet SSD output
            if '{' in line:
                data = json.loads(line)
                for det in data.get('detections', []):
                    detections.append({
                        'label': self.labels[det.get('class_id', 0)],
                        'confidence': det.get('confidence', 0),
                        'bbox': {
                            'x': det.get('x', 0),
                            'y': det.get('y', 0),
                            'width': det.get('width', 0),
                            'height': det.get('height', 0)
                        }
                    })
        except:
            pass
        return detections

    def stop_imx500_stream(self):
        """Stop IMX500 detection stream"""
        self.imx500_running = False
        if self.imx500_process:
            self.imx500_process.terminate()
            self.imx500_process = None

    def _detect_yolov5(self, frame: np.ndarray) -> List[Dict]:
        """Detect using YOLOv5"""
        results = self.model(frame)
        detections = []

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                'label': results.names[int(cls)],
                'confidence': float(conf),
                'bbox': {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            })

        return detections

    def _detect_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """Detect using MediaPipe"""
        import mediapipe as mp

        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        mp_image = self.mp_image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)

        detections = []
        for detection in result.detections:
            bbox = detection.bounding_box
            detections.append({
                'label': detection.categories[0].category_name,
                'confidence': detection.categories[0].score,
                'bbox': {
                    'x': bbox.origin_x,
                    'y': bbox.origin_y,
                    'width': bbox.width,
                    'height': bbox.height
                }
            })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()

        for det in detections:
            bbox = det['bbox']
            x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])

            # Draw box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = f"{det['label']} {det['confidence']:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated


# For testing
if __name__ == '__main__':
    print("Testing object detector...")

    # Test with YOLOv5 (requires torch)
    try:
        detector = ObjectDetector(backend='yolov5')
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(test_frame)
        print(f"YOLOv5 test: {len(results)} detections")
    except ImportError as e:
        print(f"YOLOv5 not available: {e}")

    print("Detector test complete")
