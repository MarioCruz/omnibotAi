#!/usr/bin/env python3
"""
AI Robot Control System
Main integration script that ties together camera, detection, LLM, and robot execution
"""

import argparse
import time
import signal
import sys
import threading
from typing import Optional

from camera_capture import CameraCapture
from object_detector import ObjectDetector
from llm_command_generator import LLMCommandGenerator
from robot_executor import RobotCommandExecutor


class AIRobotSystem:
    """Main AI Robot Control System"""

    def __init__(self,
                 volume: float = 0.5,
                 detector_backend: str = 'yolov5',
                 llm_model: str = 'mistral',
                 resolution: tuple = (640, 480),
                 use_llm: bool = True):

        print("=" * 50)
        print("  AI Robot Control System")
        print("=" * 50)

        self.use_llm = use_llm
        self.running = False
        self.paused = False
        self.task_context = "Explore and interact with objects"

        # Statistics
        self.iteration_count = 0
        self.total_detections = 0
        self.total_commands = 0

        # Initialize components
        print("[Init] Loading object detector...")
        try:
            self.detector = ObjectDetector(backend=detector_backend)
        except ImportError as e:
            print(f"[Init] Warning: {e}")
            print("[Init] Falling back to rule-based detection")
            self.detector = None

        print("\n[Init] Starting camera...")
        imx500_instance = self.detector.get_imx500() if self.detector and detector_backend == 'imx500' else None
        self.camera = CameraCapture(resolution=resolution, imx500=imx500_instance)

        print("[Init] Initializing LLM command generator...")
        self.llm = LLMCommandGenerator(
            model_name=llm_model,
            frame_width=resolution[0],
            frame_height=resolution[1]
        )

        if use_llm:
            if not self.llm.check_ollama_status():
                print("[Init] Warning: Ollama not available, using rule-based commands")
                self.use_llm = False

        print("[Init] Connecting to robot...")
        self.robot = RobotCommandExecutor(volume=volume)
        self.robot_connected = self.robot.connect()

        if not self.robot_connected:
            print("[Init] Warning: Robot not connected. Commands will be logged but not sent.")

        print("\n[Init] System ready!")
        print("=" * 50)

    def set_task(self, task: str):
        """Set the current task context for LLM"""
        self.task_context = task
        print(f"[Task] {task}")

    def process_frame(self) -> tuple:
        """
        Process a single frame: detect objects and generate commands

        Returns:
            (detections, commands) tuple
        """
        frame, metadata = self.camera.get_frame_and_metadata()
        if frame is None:
            return [], []

        # Detect objects
        if self.detector:
            if hasattr(self.detector, 'set_metadata'):
                self.detector.set_metadata(metadata)
            detections = self.detector.detect(frame)
        else:
            detections = []

        self.total_detections += len(detections)

        # Generate commands
        commands = self.llm.generate_commands(
            detections,
            context=self.task_context,
            use_llm=self.use_llm
        )

        self.total_commands += len(commands)

        return detections, commands

    def run(self, interval: float = 2.0, max_iterations: int = 0):
        """
        Main control loop

        Args:
            interval: Time between processing cycles in seconds
            max_iterations: Maximum iterations (0 = unlimited)
        """
        self.running = True
        print(f"\n[Run] Starting control loop (interval: {interval}s)")
        print("[Run] Press Ctrl+C to stop\n")

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.5)
                    continue

                self.iteration_count += 1

                # Check iteration limit
                if max_iterations > 0 and self.iteration_count > max_iterations:
                    print(f"[Run] Reached max iterations ({max_iterations})")
                    break

                # Process frame
                timestamp = time.strftime('%H:%M:%S')
                print(f"\n[{self.iteration_count}] {timestamp}")
                print("-" * 30)

                detections, commands = self.process_frame()

                # Log detections
                if detections:
                    print(f"[Detect] Found {len(detections)} objects:")
                    for det in detections:
                        print(f"    - {det['label']} ({det['confidence']:.0%})")
                else:
                    print("[Detect] No objects detected")

                # Execute commands
                if commands:
                    print(f"[Command] Generated {len(commands)} commands:")
                    for cmd in commands:
                        print(f"    > {cmd}")

                    if self.robot_connected:
                        for cmd in commands:
                            self.robot.execute(cmd)
                            time.sleep(0.3)
                    else:
                        print("[Command] (Robot not connected, commands logged only)")
                else:
                    print("[Command] No commands generated")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[Run] Interrupted by user")
        finally:
            self.stop()

    def run_once(self) -> tuple:
        """Run a single detection/command cycle"""
        return self.process_frame()

    def pause(self):
        """Pause the control loop"""
        self.paused = True
        print("[Control] Paused")

    def resume(self):
        """Resume the control loop"""
        self.paused = False
        print("[Control] Resumed")

    def emergency_stop(self):
        """Send emergency stop to robot"""
        print("[Control] EMERGENCY STOP")
        if self.robot_connected:
            self.robot.stop()

    def stop(self):
        """Stop the system"""
        print("\n[Shutdown] Stopping system...")

        self.running = False

        # Stop robot
        if self.robot_connected:
            self.robot.stop()
            self.robot.disconnect()

        # Stop camera
        self.camera.stop()

        # Print stats
        self._print_stats()

        print("[Shutdown] Complete")

    def _print_stats(self):
        """Print session statistics"""
        print("\n" + "=" * 50)
        print("  Session Statistics")
        print("=" * 50)
        print(f"  Iterations:     {self.iteration_count}")
        print(f"  Total detections: {self.total_detections}")
        print(f"  Total commands:   {self.total_commands}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="AI Robot Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robot_control.py
  python robot_control.py --task "Find and approach people"
  python robot_control.py --no-llm --interval 1.0
  python robot_control.py --detector yolov5 --robot-url http://192.168.1.100:5000
        """
    )

    parser.add_argument('--task', type=str,
                       default="Explore and interact with objects",
                       help='Task context for command generation')

    parser.add_argument('--volume', type=float,
                       default=0.5,
                       help='Audio volume (0.0-1.0)')

    parser.add_argument('--detector', type=str,
                       choices=['yolov5', 'mediapipe', 'imx500'],
                       default='yolov5',
                       help='Object detection backend')

    parser.add_argument('--llm-model', type=str,
                       default='mistral',
                       help='Ollama model name')

    parser.add_argument('--no-llm', action='store_true',
                       help='Use rule-based commands instead of LLM')

    parser.add_argument('--interval', type=float,
                       default=2.0,
                       help='Processing interval in seconds')

    parser.add_argument('--max-iterations', type=int,
                       default=0,
                       help='Maximum iterations (0 = unlimited)')

    parser.add_argument('--resolution', type=str,
                       default='640x480',
                       help='Camera resolution (WxH)')

    args = parser.parse_args()

    # Parse resolution
    try:
        w, h = map(int, args.resolution.lower().split('x'))
        resolution = (w, h)
    except:
        print(f"Invalid resolution format: {args.resolution}")
        resolution = (640, 480)

    # Create and run system
    system = AIRobotSystem(
        volume=args.volume,
        detector_backend=args.detector,
        llm_model=args.llm_model,
        resolution=resolution,
        use_llm=not args.no_llm
    )

    # Handle signals
    def signal_handler(sig, frame):
        system.emergency_stop()
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set task and run
    system.set_task(args.task)
    system.run(interval=args.interval, max_iterations=args.max_iterations)


if __name__ == '__main__':
    main()
