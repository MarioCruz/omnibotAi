#!/usr/bin/env python3
"""
Camera Capture Module for Raspberry Pi AI Camera
Handles continuous frame capture in a background thread
"""

from picamera2 import Picamera2
import numpy as np
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraCapture:
    """Manages continuous camera frame capture"""

    def __init__(self, resolution=(640, 480), fps=30):
        """
        Initialize camera capture

        Args:
            resolution: (width, height) tuple
            fps: Frames per second
        """
        logger.info(f"Initializing camera at {resolution}...")
        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": resolution}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.fps_target = fps

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Camera ready")

    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        frame_time = 1.0 / self.fps_target

        while self.running:
            try:
                start_time = time.time()
                frame = self.picam2.capture_array()

                with self.frame_lock:
                    self.current_frame = frame
                    self.frame_count += 1

                # Maintain target FPS
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """Get current frame (non-blocking)"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def get_frame_info(self):
        """Get frame info"""
        return {
            'count': self.frame_count,
            'is_active': self.running
        }

    def stop(self):
        """Stop camera and cleanup"""
        self.running = False
        self.capture_thread.join()
        self.picam2.stop()
        logger.info("Camera stopped")

if __name__ == "__main__":
    cap = CameraCapture()
    for i in range(30):
        frame = cap.get_frame()
        if frame is not None:
            print(f"Frame {i}: {frame.shape}")
        time.sleep(0.1)
    cap.stop()
