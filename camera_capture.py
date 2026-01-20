#!/usr/bin/env python3
"""
Camera Capture Module
Threaded frame capture from Raspberry Pi Camera
Supports IMX500 AI Camera with hardware-accelerated inference (YOLOv8)
"""

from picamera2 import Picamera2
import numpy as np
import threading
import time


class CameraCapture:
    def __init__(self, resolution=(640, 480), framerate=30, imx500=None, intrinsics=None):
        """
        Initialize camera capture.

        Args:
            resolution: Tuple of (width, height) - must be positive integers
            framerate: Target frames per second (1-120)
            imx500: Optional IMX500 instance for AI camera configuration
            intrinsics: Optional NetworkIntrinsics for model-specific settings
        """
        # Validate resolution
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValueError(f"resolution must be (width, height) tuple, got {resolution}")
        width, height = resolution
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError(f"resolution values must be integers, got ({type(width).__name__}, {type(height).__name__})")
        if width <= 0 or height <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        # Validate framerate
        if not isinstance(framerate, (int, float)) or framerate <= 0:
            raise ValueError(f"framerate must be positive number, got {framerate}")
        framerate = max(1, min(120, int(framerate)))  # Clamp to 1-120 fps

        # Use IMX500's camera_num if available
        if imx500 is not None:
            self.picam2 = Picamera2(imx500.camera_num)
        else:
            self.picam2 = Picamera2()

        self.imx500 = imx500
        self.intrinsics = intrinsics

        # Use model's inference rate if available
        if intrinsics and hasattr(intrinsics, 'inference_rate') and intrinsics.inference_rate:
            framerate = intrinsics.inference_rate

        # Store target framerate AFTER any intrinsics override
        self.target_framerate = framerate

        # Configure camera with larger buffer for IMX500
        buffer_count = 12 if imx500 else 4
        config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": resolution},
            controls={"FrameRate": framerate},
            buffer_count=buffer_count
        )
        self.picam2.configure(config)

        if imx500 is not None:
            # Show model loading progress bar
            imx500.show_network_fw_progress_bar()
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
        frame_interval = 1.0 / self.target_framerate

        while self.running:
            loop_start = time.time()
            job = None

            try:
                # Capture frame and metadata together
                job = self.picam2.capture_request()
                frame = job.make_array("main")
                metadata = job.get_metadata()

                # Store data BEFORE releasing the request
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
                # Always release the request if we got one
                if job is not None:
                    try:
                        job.release()
                    except Exception:
                        pass

            # Sleep for remaining time to hit target framerate
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame(self):
        """Get the current frame (thread-safe copy to prevent race conditions)"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def get_frame_and_metadata(self):
        """Get current frame and metadata (thread-safe copies for IMX500 inference results)"""
        with self.frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
            # Metadata is a dict - make a shallow copy to prevent mutation
            metadata = dict(self.current_metadata) if self.current_metadata else None
            return frame, metadata

    def get_fps(self):
        """Get current frames per second"""
        return self.fps

    def stop(self):
        """Stop the camera capture and release resources"""
        self.running = False
        self.capture_thread.join(timeout=2.0)

        if self.capture_thread.is_alive():
            # Thread is stuck - likely blocked on capture_request()
            # Force stop picam2 first to unblock it, then try joining again
            print("[Camera] Warning: capture thread blocked, forcing camera stop")
            try:
                self.picam2.stop()
            except Exception as e:
                print(f"[Camera] Error stopping camera: {e}")

            # Give thread one more chance to exit
            self.capture_thread.join(timeout=1.0)
            if self.capture_thread.is_alive():
                print("[Camera] Warning: capture thread did not stop - may leak resources")
            else:
                print("[Camera] Stopped (after force)")
        else:
            # Thread stopped cleanly, now stop camera
            try:
                self.picam2.stop()
            except Exception as e:
                print(f"[Camera] Error stopping camera: {e}")
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
