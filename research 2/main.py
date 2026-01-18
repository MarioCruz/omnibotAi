#!/usr/bin/env python3
"""
Main AI Robot Control System
Integrates camera, detection, LLM, and robot execution
"""

import argparse
import time
import signal
import sys
import logging
from typing import Optional, Tuple, List, Dict

from camera_capture import CameraCapture
from object_detector import ObjectDetector
from llm_command_generator import LLMCommandGenerator
from robot_executor import RobotCommandExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIRobotSystem:
    """Complete AI-powered robot control system"""

    def __init__(self, robot_url: str = "http://localhost:5000",
                 detector_type: str = "mediapipe",
                 skip_camera: bool = False):
        """
        Initialize AI robot system

        Args:
            robot_url: Robot control interface URL
            detector_type: Object detector type (mediapipe, yolo)
            skip_camera: Skip camera (for testing)
        """
        logger.info("=" * 50)
        logger.info("AI Robot Control System Starting")
        logger.info("=" * 50)

        self.skip_camera = skip_camera

        if not skip_camera:
            logger.info("[1/4] Initializing camera...")
            self.camera = CameraCapture()

            logger.info("[2/4] Loading object detector...")
            self.detector = ObjectDetector()
        else:
            self.camera = None
            self.detector = None
            logger.info("Camera/detector skipped (test mode)")

        logger.info("[3/4] Initializing LLM...")
        self.llm = LLMCommandGenerator()

        logger.info("[4/4] Connecting to robot...")
        self.robot = RobotCommandExecutor(robot_url)

        self.running = False
        self.task_context = "Explore the field"
        self.iteration_count = 0
        self.total_objects_detected = 0
        self.total_commands_executed = 0

        logger.info("✓ System initialized successfully")

    def set_task(self, task: str):
        """Set the task context for robot operation"""
        self.task_context = task
        logger.info(f"Task set: {task}")

    def process_frame(self) -> Optional[Tuple[List[Dict], List[str], str]]:
        """
        Process single frame: detect objects and generate commands

        Returns:
            Tuple of (objects, commands, reasoning) or None
        """
        if self.skip_camera:
            return None

        # Capture frame
        frame = self.camera.get_frame()
        if frame is None:
            logger.warning("No frame available")
            return None

        # Detect objects
        objects = self.detector.detect_objects(frame)
        logger.info(f"Detected {len(objects)} objects")

        for obj in objects:
            logger.info(f"  - {obj['label']} (confidence: {obj['confidence']:.1%})")
            self.total_objects_detected += 1

        # Generate commands
        commands, reasoning = self.llm.generate_commands(objects, self.task_context)
        logger.info(f"Generated {len(commands)} commands: {reasoning}")

        return objects, commands, reasoning

    def run(self, processing_interval: float = 2.0,
            command_delay: float = 0.5,
            max_iterations: int = 0):
        """
        Run main processing loop

        Args:
            processing_interval: Time between detection cycles (seconds)
            command_delay: Delay between command execution (seconds)
            max_iterations: Max iterations (0 = infinite)
        """
        self.running = True
        logger.info(f"Starting main loop (interval: {processing_interval}s)")
        logger.info("Press Ctrl+C to stop\n")

        try:
            while self.running:
                self.iteration_count += 1

                if max_iterations > 0 and self.iteration_count > max_iterations:
                    logger.info(f"Max iterations ({max_iterations}) reached")
                    break

                logger.info(f"{'='*50}")
                logger.info(f"ITERATION {self.iteration_count} | {time.strftime('%H:%M:%S')}")
                logger.info(f"{'='*50}")

                # Process frame
                result = self.process_frame()

                if result is None:
                    logger.info("Skipping iteration (no frame)")
                    time.sleep(processing_interval)
                    continue

                objects, commands, reasoning = result

                # Execute commands
                if commands:
                    logger.info(f"Executing {len(commands)} commands...")
                    executed = self.robot.execute_sequence(
                        commands, 
                        delay=command_delay
                    )
                    self.total_commands_executed += executed
                else:
                    logger.info("No commands generated")

                # Log stats
                logger.info(f"Stats - Iteration: {self.iteration_count}, "
                           f"Total objects: {self.total_objects_detected}, "
                           f"Total commands: {self.total_commands_executed}")

                time.sleep(processing_interval)

        except KeyboardInterrupt:
            logger.info("\nShutdown signal received...")
            self.stop()

    def stop(self):
        """Stop system and cleanup"""
        self.running = False

        logger.info("\n" + "="*50)
        logger.info("SYSTEM SHUTDOWN")
        logger.info("="*50)

        # Emergency stop robot
        logger.info("Sending emergency stop to robot...")
        self.robot.emergency_stop()

        # Cleanup
        if self.camera is not None:
            self.camera.stop()

        # Final stats
        logger.info(f"Final Statistics:")
        logger.info(f"  - Iterations: {self.iteration_count}")
        logger.info(f"  - Objects detected: {self.total_objects_detected}")
        logger.info(f"  - Commands executed: {self.total_commands_executed}")
        logger.info(f"  - Avg objects/iteration: {self.total_objects_detected/max(self.iteration_count, 1):.1f}")

        logger.info("✓ System stopped cleanly")

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Robot Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py --task "Find and collect balls"
  python3 main.py --task "Patrol the area" --interval 3.0
  python3 main.py --test --max-iter 10
        """
    )

    parser.add_argument('--task', type=str,
                       default="Explore the field",
                       help='Task context for robot operation')

    parser.add_argument('--robot-url', type=str,
                       default="http://localhost:5000",
                       help='Robot control interface URL')

    parser.add_argument('--detector', type=str,
                       choices=['mediapipe', 'yolo'],
                       default='mediapipe',
                       help='Object detector type')

    parser.add_argument('--interval', type=float,
                       default=2.0,
                       help='Processing interval in seconds')

    parser.add_argument('--cmd-delay', type=float,
                       default=0.5,
                       help='Delay between command execution')

    parser.add_argument('--max-iter', type=int,
                       default=0,
                       help='Maximum iterations (0 = infinite)')

    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (no camera)')

    args = parser.parse_args()

    # Initialize system
    system = AIRobotSystem(
        robot_url=args.robot_url,
        detector_type=args.detector,
        skip_camera=args.test
    )

    # Set task
    system.set_task(args.task)

    # Run main loop
    try:
        system.run(
            processing_interval=args.interval,
            command_delay=args.cmd_delay,
            max_iterations=args.max_iter
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        system.stop()
        sys.exit(1)

if __name__ == '__main__':
    main()
