#!/usr/bin/env python3
"""
Object Detection Module using MediaPipe
Lightweight and efficient for Raspberry Pi
"""

import cv2
import numpy as np
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. Install with: pip install mediapipe")

class ObjectDetector:
    """Detect objects using MediaPipe Object Detection"""

    def __init__(self, confidence_threshold=0.5):
        """
        Initialize object detector

        Args:
            confidence_threshold: Minimum confidence for detections
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available")

        self.confidence_threshold = confidence_threshold
        self.detector = None
        logger.info("MediaPipe Object Detector initialized")

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame

        Args:
            frame: BGR numpy array from camera

        Returns:
            List of detected objects with labels, confidence, and bounding boxes
        """
        try:
            if frame is None or frame.size == 0:
                return []

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # For now, return simulated detections
            # In production, integrate actual MediaPipe detection
            objects = self._simulate_detections(frame)

            return objects

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def _simulate_detections(self, frame: np.ndarray) -> List[Dict]:
        """Simulate detections for testing"""
        # This is a placeholder - replace with actual MediaPipe detection
        height, width = frame.shape[:2]

        detections = [
            {
                'label': 'person',
                'confidence': 0.85,
                'bbox': {
                    'x': width * 0.3,
                    'y': height * 0.2,
                    'width': width * 0.2,
                    'height': height * 0.4
                }
            },
            {
                'label': 'ball',
                'confidence': 0.72,
                'bbox': {
                    'x': width * 0.6,
                    'y': height * 0.5,
                    'width': width * 0.1,
                    'height': height * 0.1
                }
            }
        ]

        return [d for d in detections if d['confidence'] >= self.confidence_threshold]

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detections on frame"""
        result = frame.copy()

        for det in detections:
            bbox = det['bbox']
            label = det['label']
            conf = det['confidence']

            x = int(bbox['x'])
            y = int(bbox['y'])
            w = int(bbox['width'])
            h = int(bbox['height'])

            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            text = f"{label} {conf:.1%}"
            cv2.putText(result, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result

class YOLODetector:
    """Alternative detector using YOLOv5"""

    def __init__(self, model_size='s'):
        """
        Initialize YOLO detector

        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium)
        """
        try:
            import yolov5
            self.model = yolov5.load(f'yolov5{model_size}.pt')
            self.model.conf = 0.45
            logger.info(f"YOLO{model_size} loaded")
        except ImportError:
            raise ImportError("YOLOv5 not installed. Install with: pip install yolov5")

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO"""
        try:
            results = self.model(frame)
            detections = []

            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                detections.append({
                    'label': self.model.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': {
                        'x': float(box[0]),
                        'y': float(box[1]),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1])
                    }
                })

            return detections
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

if __name__ == "__main__":
    import time
    detector = ObjectDetector()
    print(f"Detector ready. Confidence threshold: {detector.confidence_threshold}")
