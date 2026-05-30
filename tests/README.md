# Tests

Fast, hardware-free unit tests for the pure-logic parts of OmniAI. They use the
stdlib `unittest` module only — **no pytest, no Pi, no camera/robot required** —
so they run anywhere in well under a second.

## Run them

```bash
# All tests (stdlib discovery — no dependencies)
python -m unittest discover -s tests -v

# A single file
python tests/test_navigation.py

# With pytest, if you have it
python -m pytest tests/ -v
```

## What's covered

| File | Module under test | Why it matters |
|------|-------------------|----------------|
| `test_navigation.py` | `navigation.py` | Movement decisions + context target filtering — locks in the "Find the laptop won't chase a person" regression fix. |
| `test_audio_commander.py` | `audio_commander.sanitize_speech` | Command-injection boundary for text sent to `speak_pi.sh`. |
| `test_robot_executor.py` | `robot_executor.py` | Duration clamps + the quarter-turn 100ms relay floor. |
| `test_templates.py` | `templates/` | Guards the HTML extraction (files exist, markers present). |

## What's *not* here

Anything needing the IMX500, audio relay, SPI display, or Bluetooth. Those are
exercised by the manual smoke scripts under `util/` (`smoke_test.py`,
`test_detection.py`, `test_eye_display.py`) on the Pi.
