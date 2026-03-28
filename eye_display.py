#!/usr/bin/env python3
"""
Eye Display Module - supports ST7735S 1.8" TFT (128x160) and SSD1351 OLED (128x128)
Animated robot eye for personality and feedback.

Display type is set via config.json "eye_display": "st7735" or "ssd1351"
"""

import time
import threading
import random
from PIL import Image, ImageDraw

# Try to import display libraries (only work on Pi)
HAS_ST7735 = False
HAS_SSD1351 = False

try:
    import st7735 as ST7735
    HAS_ST7735 = True
except ImportError:
    pass

try:
    from luma.oled.device import ssd1351
    from luma.core.interface.serial import spi as luma_spi
    HAS_SSD1351 = True
except ImportError:
    pass

if not HAS_ST7735 and not HAS_SSD1351:
    print("[EyeDisplay] No display library found - running in simulation mode")

HAS_DISPLAY = HAS_ST7735 or HAS_SSD1351


class EyeDisplay:
    """Animated eye display - supports ST7735S TFT (128x160) and SSD1351 OLED (128x128)"""

    # Display dimensions per type
    DISPLAY_SIZES = {
        'st7735':  (128, 160),
        'ssd1351': (128, 128),
    }

    # Eye parameters
    EYE_COLOR = (0, 200, 255)       # Cyan iris
    PUPIL_COLOR = (0, 0, 0)        # Black pupil
    SCLERA_COLOR = (255, 255, 255) # White sclera
    BG_COLOR = (20, 20, 40)        # Dark blue background

    # Eye dimensions (centered on display)
    EYE_RADIUS = 45
    PUPIL_RADIUS = 18
    IRIS_RADIUS = 30

    # Expressions
    EXPR_NORMAL = 'normal'
    EXPR_HAPPY = 'happy'
    EXPR_SURPRISED = 'surprised'
    EXPR_SLEEPY = 'sleepy'
    EXPR_ANGRY = 'angry'
    EXPR_LOOKING_LEFT = 'look_left'
    EXPR_LOOKING_RIGHT = 'look_right'
    EXPR_LOOKING_UP = 'look_up'
    EXPR_LOOKING_DOWN = 'look_down'
    EXPR_BLINK = 'blink'

    def __init__(self, display_type='st7735', dc_pin=24, rst_pin=25, cs_pin=0, spi_port=0, brightness=15):
        """
        Initialize the eye display.

        Args:
            display_type: 'st7735' for ST7735S TFT or 'ssd1351' for SSD1351 OLED
            brightness: SSD1351 master contrast 0-15 (default 15 = max)
            dc_pin: GPIO pin for DC (data/command)
            rst_pin: GPIO pin for RST (reset)
            cs_pin: SPI chip select (0=CE0, 1=CE1)
            spi_port: SPI bus (0 or 1)
        """
        self.display_type = display_type
        self.WIDTH, self.HEIGHT = self.DISPLAY_SIZES.get(display_type, (128, 160))

        self.expression = self.EXPR_NORMAL
        self.pupil_offset_x = 0
        self.pupil_offset_y = 0
        self.eyelid_position = 0  # 0 = open, 1 = closed
        self.running = False
        self.lock = threading.Lock()

        self.display = None
        self._luma_device = None  # Only used for SSD1351

        if display_type == 'ssd1351' and HAS_SSD1351:
            try:
                serial = luma_spi(port=spi_port, device=cs_pin, gpio_DC=dc_pin, gpio_RST=rst_pin)
                self._luma_device = ssd1351(serial, width=self.WIDTH, height=self.HEIGHT)
                self._spi_serial = serial
                # Set brightness using proper SSD1351 command/data sequence
                self._set_oled_brightness(brightness)
                self.display = self._luma_device
                print(f"[EyeDisplay] SSD1351 OLED initialized ({self.WIDTH}x{self.HEIGHT}, brightness={brightness})")
            except Exception as e:
                print(f"[EyeDisplay] Failed to initialize SSD1351: {e}")
        elif display_type == 'st7735' and HAS_ST7735:
            try:
                self.display = ST7735.ST7735(
                    port=spi_port,
                    cs=cs_pin,
                    dc=dc_pin,
                    rst=rst_pin,
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    rotation=0,
                    invert=False,
                    offset_left=2,
                    offset_top=1
                )
                self.display._spi.max_speed_hz = 24000000
                print(f"[EyeDisplay] ST7735S initialized ({self.WIDTH}x{self.HEIGHT})")
            except Exception as e:
                print(f"[EyeDisplay] Failed to initialize ST7735S: {e}")
                self.display = None
        else:
            print(f"[EyeDisplay] No driver for '{display_type}' - running in simulation mode")

        # Create image buffer
        self.image = Image.new('RGB', (self.WIDTH, self.HEIGHT), self.BG_COLOR)
        self.draw = ImageDraw.Draw(self.image)

        # Animation thread
        self.animation_thread = None
        self.blink_time = 0
        self.next_blink = time.time() + random.uniform(2, 5)

    def _set_oled_brightness(self, level):
        """Set SSD1351 brightness via command + data (like Arduino's sendCommand).
        level: 0-15 for master contrast (0x00-0x0F)
        """
        if not self._luma_device or not self._spi_serial:
            return
        level = max(0, min(15, level))
        channel = int(level * 255 / 15)  # Scale 0-15 to 0-255 for per-channel
        # SSD1351 command 0xC1: Set Contrast for A, B, C (needs command then 3 data bytes)
        self._spi_serial.command(0xC1)
        self._spi_serial.data([channel, channel, channel])
        # SSD1351 command 0xC7: Master Contrast (needs command then 1 data byte)
        self._spi_serial.command(0xC7)
        self._spi_serial.data([level])
        print(f"[EyeDisplay] SSD1351 brightness: master={level}/15, channel={channel}/255")

    def start(self):
        """Start the eye animation loop"""
        if self.running:
            return

        self.running = True
        self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self.animation_thread.start()
        print("[EyeDisplay] Animation started")

    def stop(self):
        """Stop the eye animation"""
        self.running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=2.0)
        print("[EyeDisplay] Animation stopped")

    def set_expression(self, expression):
        """Set the eye expression"""
        with self.lock:
            self.expression = expression

            # Reset pupil position for directional expressions
            if expression == self.EXPR_LOOKING_LEFT:
                self.pupil_offset_x = -12
                self.pupil_offset_y = 0
            elif expression == self.EXPR_LOOKING_RIGHT:
                self.pupil_offset_x = 12
                self.pupil_offset_y = 0
            elif expression == self.EXPR_LOOKING_UP:
                self.pupil_offset_x = 0
                self.pupil_offset_y = -8
            elif expression == self.EXPR_LOOKING_DOWN:
                self.pupil_offset_x = 0
                self.pupil_offset_y = 8
            else:
                self.pupil_offset_x = 0
                self.pupil_offset_y = 0

    def look_at(self, x_offset, y_offset):
        """Move pupil to look in a direction (-1 to 1 range)"""
        with self.lock:
            self.pupil_offset_x = int(x_offset * 12)
            self.pupil_offset_y = int(y_offset * 8)

    def blink(self):
        """Trigger a blink"""
        with self.lock:
            self.blink_time = time.time()

    def _animation_loop(self):
        """Main animation loop"""
        while self.running:
            try:
                # Check for random blinks
                now = time.time()
                if now >= self.next_blink:
                    self.blink()
                    self.next_blink = now + random.uniform(3, 7)

                # Calculate eyelid position for blink animation
                with self.lock:
                    if self.blink_time > 0:
                        blink_elapsed = now - self.blink_time
                        if blink_elapsed < 0.1:
                            self.eyelid_position = blink_elapsed / 0.1  # Closing
                        elif blink_elapsed < 0.2:
                            self.eyelid_position = 1.0 - (blink_elapsed - 0.1) / 0.1  # Opening
                        else:
                            self.eyelid_position = 0
                            self.blink_time = 0
                    elif self.expression == self.EXPR_SLEEPY:
                        self.eyelid_position = 0.6  # Half closed
                    elif self.expression == self.EXPR_BLINK:
                        self.eyelid_position = 1.0  # Fully closed
                    else:
                        self.eyelid_position = 0

                # Draw the eye
                self._draw_eye()

                # Update display
                if self.display:
                    if self._luma_device:
                        self._luma_device.display(self.image.convert(self._luma_device.mode))
                    else:
                        self.display.display(self.image)

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"[EyeDisplay] Animation error: {e}")
                time.sleep(0.1)

    def _draw_eye(self):
        """Draw the eye with current expression"""
        # Clear background
        self.draw.rectangle([0, 0, self.WIDTH, self.HEIGHT], fill=self.BG_COLOR)

        # Eye center
        cx = self.WIDTH // 2
        cy = self.HEIGHT // 2

        with self.lock:
            expression = self.expression
            pupil_x = self.pupil_offset_x
            pupil_y = self.pupil_offset_y
            eyelid = self.eyelid_position

        # Draw sclera (white of eye)
        if expression == self.EXPR_SURPRISED:
            # Bigger eye when surprised
            radius = self.EYE_RADIUS + 8
        elif expression == self.EXPR_ANGRY:
            # Slightly smaller, more intense
            radius = self.EYE_RADIUS - 5
        else:
            radius = self.EYE_RADIUS

        self.draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=self.SCLERA_COLOR
        )

        # Draw iris
        iris_x = cx + pupil_x
        iris_y = cy + pupil_y
        iris_r = self.IRIS_RADIUS

        if expression == self.EXPR_SURPRISED:
            iris_r = self.IRIS_RADIUS - 3  # Smaller iris = bigger whites showing

        self.draw.ellipse(
            [iris_x - iris_r, iris_y - iris_r, iris_x + iris_r, iris_y + iris_r],
            fill=self.EYE_COLOR
        )

        # Draw pupil
        pupil_r = self.PUPIL_RADIUS
        if expression == self.EXPR_SURPRISED:
            pupil_r = self.PUPIL_RADIUS - 5  # Smaller pupil when surprised
        elif expression == self.EXPR_HAPPY:
            pupil_r = self.PUPIL_RADIUS + 3  # Dilated when happy

        self.draw.ellipse(
            [iris_x - pupil_r, iris_y - pupil_r, iris_x + pupil_r, iris_y + pupil_r],
            fill=self.PUPIL_COLOR
        )

        # Draw highlight (makes eye look alive)
        highlight_x = iris_x - 8
        highlight_y = iris_y - 8
        self.draw.ellipse(
            [highlight_x - 4, highlight_y - 4, highlight_x + 4, highlight_y + 4],
            fill=(255, 255, 255)
        )

        # Draw eyelid
        if eyelid > 0:
            lid_height = int(radius * 2 * eyelid)
            self.draw.rectangle(
                [cx - radius - 5, cy - radius - 5, cx + radius + 5, cy - radius + lid_height],
                fill=self.BG_COLOR
            )
            # Bottom eyelid for sleepy/blink
            if eyelid > 0.3:
                bottom_lid = int(radius * eyelid * 0.5)
                self.draw.rectangle(
                    [cx - radius - 5, cy + radius - bottom_lid, cx + radius + 5, cy + radius + 5],
                    fill=self.BG_COLOR
                )

        # Draw happy expression (curved line under eye)
        if expression == self.EXPR_HAPPY:
            self.draw.arc(
                [cx - 30, cy + 10, cx + 30, cy + 40],
                start=0, end=180,
                fill=self.BG_COLOR, width=8
            )

        # Draw angry expression (angled eyebrow)
        if expression == self.EXPR_ANGRY:
            # Angry eyebrow
            self.draw.polygon(
                [(cx - radius, cy - radius - 10),
                 (cx + radius, cy - radius + 10),
                 (cx + radius, cy - radius + 5),
                 (cx - radius, cy - radius - 15)],
                fill=self.BG_COLOR
            )


# Convenience functions for integration
_eye_display = None

def init_eye_display(**kwargs):
    """Initialize the global eye display"""
    global _eye_display
    _eye_display = EyeDisplay(**kwargs)
    return _eye_display

def get_eye_display():
    """Get the global eye display instance"""
    return _eye_display

def eye_look_left():
    if _eye_display:
        _eye_display.set_expression(EyeDisplay.EXPR_LOOKING_LEFT)

def eye_look_right():
    if _eye_display:
        _eye_display.set_expression(EyeDisplay.EXPR_LOOKING_RIGHT)

def eye_happy():
    if _eye_display:
        _eye_display.set_expression(EyeDisplay.EXPR_HAPPY)

def eye_surprised():
    if _eye_display:
        _eye_display.set_expression(EyeDisplay.EXPR_SURPRISED)

def eye_normal():
    if _eye_display:
        _eye_display.set_expression(EyeDisplay.EXPR_NORMAL)

def eye_sleepy():
    if _eye_display:
        _eye_display.set_expression(EyeDisplay.EXPR_SLEEPY)

def eye_blink():
    if _eye_display:
        _eye_display.blink()


# Test mode
if __name__ == '__main__':
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description='Test Eye Display')
    parser.add_argument('--display', choices=['st7735', 'ssd1351'],
                        help='Display type (default: from config.json)')
    test_args = parser.parse_args()

    # Load from config.json if no CLI arg
    display_type = test_args.display
    if not display_type:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            display_type = cfg.get('eye_display', 'st7735')
        else:
            display_type = 'st7735'

    print(f"Testing Eye Display ({display_type})...")
    if display_type == 'ssd1351':
        print("Wiring for SSD1351 1.5\" OLED (128x128):")
    else:
        print("Wiring for ST7735S 1.8\" TFT (128x160):")
    print("  VCC  -> 3.3V")
    print("  GND  -> GND")
    print("  SCL  -> GPIO11 (SCLK)")
    print("  SDA  -> GPIO10 (MOSI)")
    print("  RES  -> GPIO25")
    print("  DC   -> GPIO24")
    print("  CS   -> GPIO8 (CE0)")
    print("  BLK  -> 3.3V")
    print()

    eye = EyeDisplay(display_type=display_type)
    eye.start()

    try:
        expressions = [
            ('Normal', EyeDisplay.EXPR_NORMAL),
            ('Happy', EyeDisplay.EXPR_HAPPY),
            ('Surprised', EyeDisplay.EXPR_SURPRISED),
            ('Sleepy', EyeDisplay.EXPR_SLEEPY),
            ('Angry', EyeDisplay.EXPR_ANGRY),
            ('Looking Left', EyeDisplay.EXPR_LOOKING_LEFT),
            ('Looking Right', EyeDisplay.EXPR_LOOKING_RIGHT),
            ('Looking Up', EyeDisplay.EXPR_LOOKING_UP),
            ('Looking Down', EyeDisplay.EXPR_LOOKING_DOWN),
        ]

        for name, expr in expressions:
            print(f"Expression: {name}")
            eye.set_expression(expr)
            time.sleep(2)

        print("Back to normal with random blinks...")
        eye.set_expression(EyeDisplay.EXPR_NORMAL)
        time.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        eye.stop()
        print("Done!")
