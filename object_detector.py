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
        """Initialize IMX500 using picamera2's native IMX500 support (hardware accelerated)"""
        try:
            from picamera2.devices.imx500 import IMX500
            from picamera2.devices.imx500 import postprocess_nanodet_detection

            # Default model path for MobileNet SSD on IMX500
            model_path = self.model_path or '/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'

            # Initialize IMX500 with the neural network model
            self.imx500 = IMX500(model_path)
            self.postprocess = postprocess_nanodet_detection

            print(f"[Detector] IMX500 initialized with model: {model_path}")
            print("[Detector] Detection runs on-chip (hardware accelerated)")

        except ImportError as e:
            print(f"[Detector] IMX500 import failed: {e}")
            print("[Detector] Make sure picamera2 is installed: sudo apt install python3-picamera2")
            raise

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
        Get detections from IMX500 hardware accelerator.

        The IMX500 runs inference on-chip. We get results from the
        camera metadata after each frame capture.
        """
        detections = []

        try:
            # Get the last inference results from IMX500
            # This requires the picamera2 instance to pass metadata
            metadata = getattr(self, '_last_metadata', None)
            if metadata is None:
                return detections

            # Get raw output tensors from IMX500
            np_outputs = self.imx500.get_outputs(metadata)
            if np_outputs is None:
                return detections

            # Get input/output tensor info for coordinate scaling
            input_w, input_h = self.imx500.get_input_size()

            # Post-process the neural network output
            boxes, scores, classes = self.postprocess(
                np_outputs,
                self.confidence_threshold,
                iou_thres=0.65,
                max_out_dets=10
            )

            # Convert to our detection format
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / input_w
            scale_y = frame_h / input_h

            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                # Scale coordinates to frame size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                label = self.labels[int(cls_id)] if int(cls_id) < len(self.labels) else f"class_{cls_id}"

                detections.append({
                    'label': label,
                    'confidence': float(score),
                    'bbox': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                })

        except (ValueError, IndexError, TypeError) as e:
            # Handle expected errors during detection parsing
            print(f"[Detector] Detection parse error: {e}")
        except Exception as e:
            # Log unexpected errors
            print(f"[Detector] Unexpected error: {e}")

        return detections

    def set_metadata(self, metadata):
        """
        Set the latest camera metadata containing IMX500 inference results.
        Call this after each frame capture from picamera2.
        """
        self._last_metadata = metadata

    def get_imx500(self):
        """Return the IMX500 instance for camera configuration"""
        return getattr(self, 'imx500', None)

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
