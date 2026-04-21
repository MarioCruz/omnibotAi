#!/usr/bin/env python3
"""Camera + AI sanity test.

Runs the production CameraCapture + ObjectDetector pipeline for N seconds and
prints every detection to stdout. No web UI, no MJPEG, no Flask. Designed
for quick "is the AI actually seeing things right now?" checks over SSH.

Usage:
  python3 util/test_camera_ai.py                # 10 seconds, default conf 0.3
  python3 util/test_camera_ai.py --seconds 30
  python3 util/test_camera_ai.py --conf 0.5
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from camera_capture import CameraCapture        # noqa: E402
from object_detector import ObjectDetector      # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=10,
                        help='How long to run (default: 10)')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold 0.0-1.0 (default: 0.3)')
    args = parser.parse_args()

    print("=== OmniBot camera + AI sanity test ===")
    print(f"Running for {args.seconds}s, confidence >= {args.conf:.0%}")
    print()

    detector = ObjectDetector(backend='imx500', confidence_threshold=args.conf)
    imx500 = detector.get_imx500()
    intrinsics = detector.get_intrinsics()

    camera = CameraCapture(
        resolution=(640, 480), framerate=30,
        imx500=imx500, intrinsics=intrinsics,
    )
    detector.set_picam2(camera.picam2)

    # Wait for the first frame.
    deadline = time.time() + 5
    while time.time() < deadline and camera.get_frame() is None:
        time.sleep(0.1)
    if camera.get_frame() is None:
        print("ERROR: no camera frame within 5s", file=sys.stderr)
        camera.stop()
        sys.exit(1)

    seen_labels = set()
    total_detections = 0
    frames = 0
    end = time.time() + args.seconds

    try:
        while time.time() < end:
            frame, metadata = camera.get_frame_and_metadata()
            if frame is None or metadata is None:
                time.sleep(0.05)
                continue

            detector.set_metadata(metadata)
            detections = detector.detect(frame)
            frames += 1

            for d in detections:
                total_detections += 1
                seen_labels.add(d['label'])
                b = d['bbox']
                print(f"  {d['label']:<12} {d['confidence']*100:>3.0f}%  "
                      f"bbox=({b['x']},{b['y']},{b['width']}x{b['height']})")

            time.sleep(0.1)  # ~10 Hz print rate, plenty for a sanity check
    finally:
        camera.stop()

    print()
    print(f"=== {args.seconds}s done ===")
    print(f"  camera fps:   {camera.get_fps():.1f}")
    print(f"  loop frames:  {frames}")
    print(f"  detections:   {total_detections}")
    if seen_labels:
        print(f"  unique objects seen: {', '.join(sorted(seen_labels))}")
    else:
        print("  unique objects seen: (none)")


if __name__ == '__main__':
    main()
