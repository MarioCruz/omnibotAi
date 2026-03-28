#!/usr/bin/env python3
"""
Test script for Eye Display (ST7735S or SSD1351)
Reads display type from config.json, or pass --display to override.
Cycles through all expressions, blinks, pupil tracking, and idle mode.
"""

import sys
import os
import time
import json
import math
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eye_display import EyeDisplay


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description='Test Eye Display')
    parser.add_argument('--display', choices=['st7735', 'ssd1351'],
                        help='Display type (default: from config.json)')
    args = parser.parse_args()

    cfg = load_config()
    display_type = args.display or cfg.get('eye_display', 'st7735')

    print("=" * 60)
    print(f"  Eye Display Test  —  {display_type.upper()}")
    print("=" * 60)
    print()
    if display_type == 'ssd1351':
        print("  Display: SSD1351 1.5\" Color OLED (128x128)")
    else:
        print("  Display: ST7735S 1.8\" TFT (128x160)")
    print("  Wiring:")
    print("    VCC  -> 3.3V")
    print("    GND  -> GND")
    print("    SCL  -> GPIO11 (SCLK)")
    print("    SDA  -> GPIO10 (MOSI)")
    print("    RES  -> GPIO25")
    print("    DC   -> GPIO24")
    print("    CS   -> GPIO8 (CE0)")
    if display_type == 'st7735':
        print("    BLK  -> 3.3V (backlight)")
    print()

    eye = EyeDisplay(
        display_type=display_type,
        dc_pin=cfg.get('eye_dc_pin', 24),
        rst_pin=cfg.get('eye_rst_pin', 25),
        cs_pin=cfg.get('eye_cs_pin', 0),
        spi_port=cfg.get('eye_spi_port', 0),
        brightness=cfg.get('eye_brightness', 15),
    )
    eye.start()
    print()

    try:
        # --- Expressions ---
        expressions = [
            ('Normal',        EyeDisplay.EXPR_NORMAL,        2),
            ('Happy',         EyeDisplay.EXPR_HAPPY,         2),
            ('Surprised',     EyeDisplay.EXPR_SURPRISED,     2),
            ('Sleepy',        EyeDisplay.EXPR_SLEEPY,        2),
            ('Angry',         EyeDisplay.EXPR_ANGRY,         2),
            ('Looking Left',  EyeDisplay.EXPR_LOOKING_LEFT,  1.5),
            ('Looking Right', EyeDisplay.EXPR_LOOKING_RIGHT, 1.5),
            ('Looking Up',    EyeDisplay.EXPR_LOOKING_UP,    1.5),
            ('Looking Down',  EyeDisplay.EXPR_LOOKING_DOWN,  1.5),
        ]

        print("Expressions")
        print("-" * 40)
        for name, expr, duration in expressions:
            print(f"  {name}...")
            eye.set_expression(expr)
            time.sleep(duration)

        # --- Blinks ---
        print()
        print("Manual Blinks")
        print("-" * 40)
        eye.set_expression(EyeDisplay.EXPR_NORMAL)
        for i in range(3):
            print(f"  Blink {i + 1}...")
            eye.blink()
            time.sleep(1)

        # --- Pupil Tracking ---
        print()
        print("Pupil Tracking")
        print("-" * 40)
        eye.set_expression(EyeDisplay.EXPR_NORMAL)

        # Circular sweep
        print("  Circular sweep...")
        for i in range(80):
            angle = (i / 80) * 2 * math.pi
            eye.look_at(math.cos(angle), math.sin(angle))
            time.sleep(0.04)

        # Horizontal sweep
        print("  Horizontal sweep...")
        for x in range(-10, 11):
            eye.look_at(x / 10, 0)
            time.sleep(0.06)
        for x in range(10, -11, -1):
            eye.look_at(x / 10, 0)
            time.sleep(0.06)

        # Vertical sweep
        print("  Vertical sweep...")
        for y in range(-10, 11):
            eye.look_at(0, y / 10)
            time.sleep(0.06)
        for y in range(10, -11, -1):
            eye.look_at(0, y / 10)
            time.sleep(0.06)

        eye.look_at(0, 0)

        # --- Expression rapid cycle ---
        print()
        print("Rapid Cycle")
        print("-" * 40)
        rapid = [
            EyeDisplay.EXPR_NORMAL,
            EyeDisplay.EXPR_HAPPY,
            EyeDisplay.EXPR_SURPRISED,
            EyeDisplay.EXPR_ANGRY,
            EyeDisplay.EXPR_HAPPY,
            EyeDisplay.EXPR_SLEEPY,
            EyeDisplay.EXPR_NORMAL,
        ]
        for expr in rapid:
            print(f"  {expr}...")
            eye.set_expression(expr)
            time.sleep(0.8)

        # --- Idle ---
        print()
        print("Idle mode (auto-blink). Press Ctrl+C to exit.")
        eye.set_expression(EyeDisplay.EXPR_NORMAL)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("Done!")

    finally:
        eye.stop()


if __name__ == '__main__':
    main()
