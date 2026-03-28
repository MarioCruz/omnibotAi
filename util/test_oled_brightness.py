#!/usr/bin/env python3
"""
OLED Brightness Test - cycles SSD1351 from dim to max
Helps find the right brightness level for your setup.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from luma.oled.device import ssd1351
    from luma.core.interface.serial import spi as luma_spi
except ImportError:
    print("Need luma.oled: pip install luma.oled")
    sys.exit(1)

from PIL import Image, ImageDraw


def draw_test_pattern(device):
    """Draw a simple eye-like pattern to judge brightness"""
    img = Image.new(device.mode, (128, 128), (30, 25, 55))
    draw = ImageDraw.Draw(img)
    # White circle
    draw.ellipse([24, 24, 104, 104], fill=(255, 255, 255))
    # Cyan iris
    draw.ellipse([39, 39, 89, 89], fill=(50, 255, 255))
    # Black pupil
    draw.ellipse([49, 49, 79, 79], fill=(0, 0, 0))
    # Highlight
    draw.ellipse([52, 42, 60, 50], fill=(255, 255, 255))
    device.display(img)


def main():
    import json
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)

    spi_port = cfg.get('eye_spi_port', 0)
    cs_pin = cfg.get('eye_cs_pin', 0)
    dc_pin = cfg.get('eye_dc_pin', 24)
    rst_pin = cfg.get('eye_rst_pin', 25)

    print("Connecting to SSD1351...")
    serial = luma_spi(port=spi_port, device=cs_pin, gpio_DC=dc_pin, gpio_RST=rst_pin)
    device = ssd1351(serial, width=128, height=128)

    draw_test_pattern(device)

    print()
    print("=" * 50)
    print("  SSD1351 Brightness Test")
    print("=" * 50)
    print()

    # Sweep from low to high
    levels = [
        (0x20, 0x01, "Very dim"),
        (0x40, 0x03, "Dim"),
        (0x60, 0x05, "Low"),
        (0x80, 0x07, "Medium-low"),
        (0xA0, 0x09, "Medium"),
        (0xC0, 0x0B, "Medium-high"),
        (0xD0, 0x0D, "High"),
        (0xE0, 0x0E, "Very high"),
        (0xFF, 0x0F, "Maximum"),
    ]

    print("Cycling through brightness levels...")
    print("Watch the OLED and note which level you like.")
    print()

    for channel, master, label in levels:
        serial.command(0xC1, channel, channel, channel)  # Per-channel contrast
        serial.command(0xC7, master)                      # Master contrast
        device.contrast(channel)
        print(f"  [{channel:#04x} / {master:#04x}]  {label}")
        time.sleep(2)

    print()
    print("-" * 50)
    print("Now stepping back down...")
    print()

    for channel, master, label in reversed(levels):
        serial.command(0xC1, channel, channel, channel)
        serial.command(0xC7, master)
        device.contrast(channel)
        print(f"  [{channel:#04x} / {master:#04x}]  {label}")
        time.sleep(2)

    # End at max
    serial.command(0xC1, 0xFF, 0xFF, 0xFF)
    serial.command(0xC7, 0x0F)
    device.contrast(0xFF)

    print()
    print("Back to maximum. Done!")
    print()
    print("To set a level in eye_display.py, use these values:")
    print("  serial.command(0xC1, LEVEL, LEVEL, LEVEL)  # channel contrast")
    print("  serial.command(0xC7, MASTER)                # master contrast")


if __name__ == '__main__':
    main()
