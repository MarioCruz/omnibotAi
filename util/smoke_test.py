#!/usr/bin/env python3
"""
Pre-launch smoke test for the OmniBot dashboard.

Catches broken imports, missing hardware, and missing dependencies BEFORE
systemd (or start.sh) launches the dashboard — so a bad deploy doesn't
flap the service in a restart loop.

Exit codes:
  0  All critical checks passed.
  1  One or more critical checks failed — do not launch.

Optional checks (eye display) print a warning but never cause failure.

Usage:
  python3 util/smoke_test.py
  python3 util/smoke_test.py --skip-hardware   # imports only (dev machines)
"""

import argparse
import os
import sys
import time
import traceback

# Make the project root importable when run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAIL = []
WARN = []


def _check(name, fn, critical=True):
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"  OK   {name} ({dt*1000:.0f}ms)")
        return True
    except Exception as e:
        dt = time.time() - t0
        target = FAIL if critical else WARN
        target.append(f"{name}: {e}")
        tag = "FAIL" if critical else "WARN"
        print(f"  {tag} {name} ({dt*1000:.0f}ms): {e}")
        if os.environ.get('SMOKE_VERBOSE'):
            traceback.print_exc()
        return False


def check_pure_imports():
    """Modules that don't need Pi-specific packages."""
    import audio_commander  # noqa: F401
    import navigation  # noqa: F401
    import robot_executor  # noqa: F401


def check_hardware_imports():
    """Modules that pull in picamera2 / st7735 / luma.oled — Pi-only."""
    import camera_capture  # noqa: F401
    import dashboard  # noqa: F401
    import eye_display  # noqa: F401
    import object_detector  # noqa: F401


def check_config():
    """config.json must parse if present (missing is fine, defaults apply)."""
    import json
    cfg_path = os.path.join(ROOT, 'config.json')
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path) as f:
        json.load(f)


def check_camera():
    """Open the camera, grab one frame, stop cleanly."""
    from camera_capture import CameraCapture
    cam = CameraCapture(resolution=(640, 480), framerate=30)
    try:
        # Capture thread is async — give it up to 3s to produce a frame.
        for _ in range(30):
            if cam.get_frame() is not None:
                break
            time.sleep(0.1)
        frame = cam.get_frame()
        if frame is None:
            raise RuntimeError("no frame captured within 3s")
    finally:
        cam.stop()


def check_audio():
    """sox + pw-play must be available and successfully connect to PipeWire."""
    import subprocess
    for tool in ('sox', 'pw-play'):
        r = subprocess.run(['which', tool], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{tool} not on PATH")
    # 0.1s tone at 1000 Hz, silenced with gain -60 so nothing audible.
    # Capture stderr so we can report a PipeWire connection failure rather
    # than letting pw-play exit non-zero silently.
    p = subprocess.Popen(
        ['sox', '-n', '-t', 'wav', '-', 'synth', '0.1', 'sine', '1000', 'gain', '-60'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    q = subprocess.Popen(
        ['pw-play', '-'], stdin=p.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    p.stdout.close()
    try:
        _, err = q.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        q.kill()
        raise RuntimeError("pw-play hung (Bluetooth speaker offline?)")
    if q.returncode != 0:
        msg = (err or b'').decode(errors='replace').strip() or 'unknown error'
        # Common case: systemd service missing XDG_RUNTIME_DIR so pw-play
        # can't find the PipeWire socket.
        raise RuntimeError(f"pw-play exit {q.returncode}: {msg}")


def check_eye_display():
    """Optional: open the configured eye display, draw once, close."""
    import json
    cfg_path = os.path.join(ROOT, 'config.json')
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    if cfg.get('eye_display', 'st7735') == 'none':
        return  # explicitly disabled
    from eye_display import EyeDisplay
    eye = EyeDisplay(
        display_type=cfg.get('eye_display', 'st7735'),
        dc_pin=cfg.get('eye_dc_pin', 24),
        rst_pin=cfg.get('eye_rst_pin', 25),
        cs_pin=cfg.get('eye_cs_pin', 0),
        spi_port=cfg.get('eye_spi_port', 0),
        brightness=cfg.get('eye_brightness', 15),
        rotation=cfg.get('eye_rotation', 0),
        offset_x=cfg.get('eye_offset_x', 0),
        offset_y=cfg.get('eye_offset_y', 0),
    )
    try:
        eye.start()
        time.sleep(0.2)
    finally:
        eye.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-hardware', action='store_true',
                        help='Run import/config checks only (no camera, audio, eye)')
    args = parser.parse_args()

    print("=== OmniBot smoke test ===")
    _check("pure imports", check_pure_imports, critical=True)
    _check("config.json parses", check_config, critical=True)

    if not args.skip_hardware:
        _check("hardware imports", check_hardware_imports, critical=True)
        _check("camera captures a frame", check_camera, critical=True)
        _check("audio pipeline (sox + pw-play)", check_audio, critical=True)
        _check("eye display init", check_eye_display, critical=False)

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print(f"  - {f}")
        if WARN:
            print(f"Warnings ({len(WARN)}):")
            for w in WARN:
                print(f"  - {w}")
        sys.exit(1)

    if WARN:
        print(f"Passed with warnings ({len(WARN)}):")
        for w in WARN:
            print(f"  - {w}")
    else:
        print("All checks passed.")
    sys.exit(0)


if __name__ == '__main__':
    main()
