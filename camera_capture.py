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

        # Configure camera
        config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": resolution},
            controls={"FrameRate": framerate}
        )
        self.picam2.configure(config)

        if imx500 is not None:
            print("[Camera] Configured for IMX500 AI Camera (hardware accelerated)")

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
        frames_this_second = 0
        last_fps_update = time.time()

        while self.running:
            job = None
            try:
                # Capture frame and metadata together using capture_request()
                # This returns the array and we get metadata from the same capture
                job = self.picam2.capture_request()
                frame = job.make_array("main")
                metadata = job.get_metadata()

                with self.frame_lock:
                    self.current_frame = frame
                    self.current_metadata = metadata

                frames_this_second += 1

                # Calculate FPS every second
                now = time.time()
                elapsed = now - last_fps_update
                if elapsed >= 1.0:
                    self.fps = int(frames_this_second / elapsed)
                    frames_this_second = 0
                    last_fps_update = now

            except Exception as e:
                print(f"[Camera] Capture error: {e}")
                time.sleep(0.1)
                continue
            finally:
                # Always release the capture request to prevent resource leak
                if job is not None:
                    job.release()

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
            # Copy metadata dict to ensure thread safety (metadata is a dict reference)
            metadata = dict(self.current_metadata) if self.current_metadata is not None else None
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
