# 👁️ BlinkRate

Real-time fatigue detection using just a webcam.

## What It Does

BlinkRate watches your eyes through your webcam and tells you how tired you are.
It runs at 30+ FPS and displays a live fatigue score on screen.

Most systems count how often you blink. BlinkRate also measures **how** you blink
— a new metric called **EARD** (Eye Aspect Ratio Dynamics) that captures
changes in eyelid movement speed.

## Features

- Real-time face and eye tracking
- Live fatigue score (green → orange → red)
- Novel EARD metric
- PERCLOS (clinical drowsiness standard)
- Session logging to JSON
- Post-session reports with graphs

## Requirements

- Python 3.10 or 3.11 (not 3.12+)
- A webcam
- Windows, macOS, or Linux

## Files

- BlinkRate/
├── blinkrate.py          # Main app
├── analyze_session.py    # Report generator
├── requirements.txt
├── README.md
└── sessions/             # Saved sessions

## Author

- Thuvaraka Yogeswaran
- GitHub: @ThuvarakaYogeswaran
