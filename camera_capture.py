#!/usr/bin/env python3
"""
Camera Capture Module
Threaded frame capture from Raspberry Pi Camera
Supports IMX500 AI Camera with hardware-accelerated inference
"""

from picamera2 import Picamera2
import numpy as np
import threading
import time


class CameraCapture:
    def __init__(self, resolution=(640, 480), framerate=30, imx500=None):
        """
        Initialize camera capture.

        Args:
            resolution: Tuple of (width, height)
            framerate: Target frames per second
            imx500: Optional IMX500 instance for AI camera configuration
        """
        self.picam2 = Picamera2()
        self.imx500 = imx500

        # Configure camera - use IMX500 config if available
        if imx500 is not None:
            # IMX500 AI Camera configuration
            config = self.picam2.create_preview_configuration(
                main={"format": 'RGB888', "size": resolution},
                controls={"FrameRate": framerate}
            )
            # Apply IMX500-specific configuration
            imx500.configure(self.picam2, config)
            print("[Camera] Configured for IMX500 AI Camera (hardware accelerated)")
        else:
            # Standard camera configuration
            config = self.picam2.create_preview_configuration(
                main={"format": 'RGB888', "size": resolution},
                controls={"FrameRate": framerate}
            )
            self.picam2.configure(config)

        self.picam2.start()

        self.current_frame = None
        self.current_metadata = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        print(f"[Camera] Started at {resolution[0]}x{resolution[1]} @ {framerate}fps")

    def _capture_loop(self):
        while self.running:
            # Capture frame with metadata (needed for IMX500 inference results)
            frame, metadata = self.picam2.capture_array(), self.picam2.capture_metadata()

            with self.frame_lock:
                self.current_frame = frame
                self.current_metadata = metadata
                self.frame_count += 1

            # Calculate FPS every second
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = now

            time.sleep(0.033)  # ~30 FPS

    def get_frame(self):
        """Get the current frame (thread-safe copy)"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def get_frame_and_metadata(self):
        """Get current frame and metadata (for IMX500 inference results)"""
        with self.frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
            metadata = self.current_metadata
            return frame, metadata

    def get_fps(self):
        """Get current frames per second"""
        return self.fps

    def stop(self):
        """Stop the camera capture"""
        self.running = False
        self.capture_thread.join(timeout=2.0)
        self.picam2.stop()
        print("[Camera] Stopped")


# For testing
if __name__ == '__main__':
    print("Testing camera capture...")
    cam = CameraCapture(resolution=(640, 480))

    try:
        for i in range(10):
            frame = cam.get_frame()
            if frame is not None:
                print(f"Frame {i+1}: shape={frame.shape}, fps={cam.get_fps()}")
            time.sleep(0.5)
    finally:
        cam.stop()
