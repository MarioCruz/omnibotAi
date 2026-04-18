#!/usr/bin/env python3
"""Render every EyeDisplay expression into a single animated GIF for the blog.

Runs on any machine with Pillow installed (no display hardware required) — the
EyeDisplay class falls back to simulation mode when st7735/luma.oled are
missing, so `_draw_eye()` just paints into its PIL buffer.
"""

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eye_display import EyeDisplay  # noqa: E402


SEQUENCE = [
    (EyeDisplay.EXPR_NORMAL,         0.0),
    (EyeDisplay.EXPR_LOOKING_LEFT,   0.0),
    (EyeDisplay.EXPR_LOOKING_RIGHT,  0.0),
    (EyeDisplay.EXPR_LOOKING_UP,     0.0),
    (EyeDisplay.EXPR_LOOKING_DOWN,   0.0),
    (EyeDisplay.EXPR_HAPPY,          0.0),
    (EyeDisplay.EXPR_SURPRISED,      0.0),
    (EyeDisplay.EXPR_ANGRY,          0.0),
    (EyeDisplay.EXPR_SLEEPY,         0.6),
    (EyeDisplay.EXPR_BLINK,          1.0),
]

FRAME_MS = 700
SCALE = 4
ROTATION = 180  # degrees clockwise
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'docs', 'images', 'ring-o-eye-expressions.gif')


def render_frame(eye, expression, eyelid):
    eye.set_expression(expression)
    eye.eyelid_position = eyelid
    eye._draw_eye()
    frame = eye.image.copy()
    if ROTATION:
        frame = frame.rotate(-ROTATION, expand=True)
    if SCALE != 1:
        frame = frame.resize(
            (frame.width * SCALE, frame.height * SCALE),
            resample=Image.NEAREST,
        )
    return frame


def main():
    eye = EyeDisplay(display_type='ssd1351')
    frames = [render_frame(eye, expr, lid) for expr, lid in SEQUENCE]

    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUTPUT} ({len(frames)} frames, {frames[0].size})")


if __name__ == '__main__':
    main()
