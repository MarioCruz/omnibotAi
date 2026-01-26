#!/usr/bin/env python3
"""
Test script for ST7735S Eye Display
Run this to verify wiring and test all expressions
"""

import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eye_display import EyeDisplay

def main():
    print("=" * 60)
    print("  ST7735S Eye Display Test")
    print("=" * 60)
    print()
    print("Wiring for ST7735S 1.8\" (128x160):")
    print("  VCC  -> 3.3V")
    print("  GND  -> GND")
    print("  SCL  -> GPIO11 (SCLK)")
    print("  SDA  -> GPIO10 (MOSI)")
    print("  RES  -> GPIO25")
    print("  DC   -> GPIO24")
    print("  CS   -> GPIO8 (CE0)")
    print("  BLK  -> 3.3V (backlight)")
    print()
    print("Starting eye display...")
    print()

    eye = EyeDisplay()
    eye.start()

    try:
        expressions = [
            ('Normal', EyeDisplay.EXPR_NORMAL, 2),
            ('Happy', EyeDisplay.EXPR_HAPPY, 2),
            ('Surprised', EyeDisplay.EXPR_SURPRISED, 2),
            ('Sleepy', EyeDisplay.EXPR_SLEEPY, 2),
            ('Angry', EyeDisplay.EXPR_ANGRY, 2),
            ('Looking Left', EyeDisplay.EXPR_LOOKING_LEFT, 1.5),
            ('Looking Right', EyeDisplay.EXPR_LOOKING_RIGHT, 1.5),
            ('Looking Up', EyeDisplay.EXPR_LOOKING_UP, 1.5),
            ('Looking Down', EyeDisplay.EXPR_LOOKING_DOWN, 1.5),
        ]

        print("Testing expressions:")
        print("-" * 40)

        for name, expr, duration in expressions:
            print(f"  {name}...")
            eye.set_expression(expr)
            time.sleep(duration)

        print()
        print("Testing manual blink...")
        eye.set_expression(EyeDisplay.EXPR_NORMAL)
        for i in range(3):
            print(f"  Blink {i+1}...")
            eye.blink()
            time.sleep(1)

        print()
        print("Testing pupil tracking...")
        eye.set_expression(EyeDisplay.EXPR_NORMAL)

        # Smooth circular motion
        import math
        for i in range(60):
            angle = (i / 60) * 2 * math.pi
            x = math.cos(angle)
            y = math.sin(angle)
            eye.look_at(x, y)
            time.sleep(0.05)

        eye.look_at(0, 0)  # Center

        print()
        print("Entering idle mode with random blinks...")
        print("Press Ctrl+C to exit")
        print()

        eye.set_expression(EyeDisplay.EXPR_NORMAL)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("Test complete!")

    finally:
        eye.stop()


if __name__ == '__main__':
    main()
