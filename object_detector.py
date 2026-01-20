#!/usr/bin/env python3
"""
Object Detection Module - IMX500 YOLO11 with Official picamera2 API
Supports multiple backends: IMX500 (native hardware), YOLOv5, MediaPipe
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from functools import lru_cache


class ObjectDetector:
    """Object detector with IMX500 hardware acceleration using official picamera2 API"""

    # Default to YOLOv8 for proven performance
    DEFAULT_MODEL = '/usr/share/imx500-models/imx500_network_yolov8n_pp.rpk'

    def __init__(self, backend='imx500', model_path=None, confidence_threshold=0.3):
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path

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
        """Initialize IMX500 using official picamera2 API with NetworkIntrinsics"""
        try:
            from picamera2.devices import IMX500
            from picamera2.devices.imx500 import NetworkIntrinsics

            # Use YOLOv8 by default for better accuracy
            model_path = self.model_path or self.DEFAULT_MODEL

            # Initialize IMX500 with the neural network model
            # This uploads the model firmware to the camera's AI chip
            self.imx500 = IMX500(model_path)

            # Get network intrinsics (model metadata)
            self.intrinsics = self.imx500.network_intrinsics
            if not self.intrinsics:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"
            elif self.intrinsics.task != "object detection":
                print(f"[Detector] Warning: Model task is '{self.intrinsics.task}', expected 'object detection'")

            # Set labels
            self.intrinsics.labels = self.labels
            self.intrinsics.update_with_defaults()

            print(f"[Detector] IMX500 initialized with model: {model_path}")
            print(f"[Detector] Model task: {self.intrinsics.task}")
            print(f"[Detector] Inference rate: {self.intrinsics.inference_rate} fps")
            print("[Detector] Detection runs on-chip (hardware accelerated)")

        except ImportError as e:
            print(f"[Detector] IMX500 import failed: {e}")
            print("[Detector] Make sure picamera2 is installed: sudo apt install python3-picamera2")
            raise

    # COCO class labels (80 classes)
    labels = [
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
        Get detections from IMX500 hardware accelerator using official picamera2 API.

        Uses imx500.get_outputs() with add_batch=True and handles model-specific
        bbox_normalization and bbox_order from NetworkIntrinsics.
        """
        detections = []

        try:
            metadata = getattr(self, '_last_metadata', None)
            if metadata is None:
                return detections

            # Validate metadata is a dict (picamera2 metadata format)
            if not isinstance(metadata, dict):
                if not getattr(self, '_warned_metadata_type', False):
                    print(f"[Detector] Warning: metadata is {type(metadata).__name__}, expected dict")
                    self._warned_metadata_type = True
                return detections

            # Get outputs with batch dimension added
            np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is None:
                return detections

            input_w, input_h = self.imx500.get_input_size()

            # Check for nanodet postprocessing
            if self.intrinsics and self.intrinsics.postprocess == "nanodet":
                from picamera2.devices.imx500 import postprocess_nanodet_detection
                from picamera2.devices.imx500.postprocess import scale_boxes
                boxes, scores, classes = postprocess_nanodet_detection(
                    outputs=np_outputs[0],
                    conf=self.confidence_threshold,
                    iou_thres=0.65,
                    max_out_dets=10
                )[0]
                boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            else:
                # Standard SSD/YOLO output format
                boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

                # Handle bbox normalization based on model intrinsics
                bbox_normalization = self.intrinsics.bbox_normalization if self.intrinsics else True
                if bbox_normalization:
                    boxes = boxes / input_h

                # Handle bbox order (xy vs yx) - YOLOv8 uses xy
                bbox_order = self.intrinsics.bbox_order if self.intrinsics else "yx"
                if bbox_order == "xy":
                    boxes = boxes[:, [1, 0, 3, 2]]

                boxes = np.array_split(boxes, 4, axis=1)
                boxes = list(zip(*boxes))

            # Get picam2 reference for coordinate conversion
            picam2 = getattr(self, '_picam2', None)

            # Convert detections
            for box, score, category in zip(boxes, scores, classes):
                if score < self.confidence_threshold:
                    continue

                cls_id = int(category)
                label = self.labels[cls_id] if cls_id < len(self.labels) else f"class_{cls_id}"

                # Use official convert_inference_coords if picam2 is available
                if picam2 is not None:
                    try:
                        x, y, w, h = self.imx500.convert_inference_coords(box, metadata, picam2)
                        detections.append({
                            'label': label,
                            'confidence': float(score),
                            'bbox': {
                                'x': int(x),
                                'y': int(y),
                                'width': int(w),
                                'height': int(h)
                            }
                        })
                        continue
                    except (ValueError, TypeError, IndexError) as e:
                        # Log first coordinate conversion failure, then fall back silently
                        if not getattr(self, '_warned_coord_convert', False):
                            print(f"[Detector] Coordinate conversion failed: {e}, using fallback")
                            self._warned_coord_convert = True

                # Manual coordinate conversion (fallback)
                frame_h, frame_w = frame.shape[:2]
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    y1, x1, y2, x2 = [float(b) for b in box]
                else:
                    y1, x1, y2, x2 = box.flatten()

                # Clamp normalized coordinates to valid range [0, 1]
                y1 = max(0.0, min(1.0, y1))
                x1 = max(0.0, min(1.0, x1))
                y2 = max(0.0, min(1.0, y2))
                x2 = max(0.0, min(1.0, x2))

                detections.append({
                    'label': label,
                    'confidence': float(score),
                    'bbox': {
                        'x': int(x1 * frame_w),
                        'y': int(y1 * frame_h),
                        'width': int((x2 - x1) * frame_w),
                        'height': int((y2 - y1) * frame_h)
                    }
                })

        except (ValueError, TypeError, IndexError, KeyError) as e:
            # Log specific errors that indicate data format issues
            if not getattr(self, '_warned_detection_error', False):
                print(f"[Detector] Detection parsing error: {e}")
                self._warned_detection_error = True
        except Exception as e:
            # Log unexpected errors with full details for debugging
            print(f"[Detector] Unexpected detection error: {type(e).__name__}: {e}")

        return detections

    def set_metadata(self, metadata):
        """
        Set the latest camera metadata containing IMX500 inference results.
        Call this after each frame capture from picamera2.

        Args:
            metadata: dict from picam2.capture_metadata() or request.get_metadata()
        """
        # Accept None (clears metadata) or dict (picamera2 metadata format)
        if metadata is not None and not isinstance(metadata, dict):
            if not getattr(self, '_warned_set_metadata', False):
                print(f"[Detector] Warning: set_metadata expects dict, got {type(metadata).__name__}")
                self._warned_set_metadata = True
            return
        self._last_metadata = metadata

    def set_picam2(self, picam2):
        """
        Set the Picamera2 instance for coordinate conversion.
        This enables using imx500.convert_inference_coords() for accurate bbox mapping.

        Args:
            picam2: Picamera2 instance (must have camera_configuration attribute)
        """
        # Validate it looks like a Picamera2 instance
        if picam2 is not None and not hasattr(picam2, 'camera_configuration'):
            print(f"[Detector] Warning: set_picam2 expects Picamera2 instance, got {type(picam2).__name__}")
            return
        self._picam2 = picam2

    def get_imx500(self):
        """Return the IMX500 instance for camera configuration"""
        return getattr(self, 'imx500', None)

    def get_intrinsics(self):
        """Return the NetworkIntrinsics for model configuration"""
        return getattr(self, 'intrinsics', None)

    def stop(self):
        """
        Clean up detector resources.
        Call this when shutting down to release IMX500 and other backend resources.
        """
        # Clear references to allow garbage collection
        self._last_metadata = None
        self._picam2 = None

        # Reset warning flags for potential reuse
        for attr in ['_warned_detection_error', '_warned_metadata_type',
                     '_warned_coord_convert', '_warned_set_metadata']:
            if hasattr(self, attr):
                delattr(self, attr)

        if self.backend == 'imx500':
            # IMX500 doesn't have explicit cleanup - it's tied to Picamera2 lifecycle
            self.imx500 = None
            self.intrinsics = None
            print("[Detector] IMX500 resources released")
        elif self.backend == 'yolov5':
            self.model = None
            print("[Detector] YOLOv5 model released")
        elif self.backend == 'mediapipe':
            if hasattr(self, 'detector'):
                self.detector.close()
                self.detector = None
            print("[Detector] MediaPipe detector closed")

        print("[Detector] Stopped")

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

        # Convert BGR to RGB if needed (MediaPipe expects RGB)
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
