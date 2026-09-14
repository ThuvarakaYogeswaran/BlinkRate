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

## Install

```bash
git clone https://github.com/ThuvarakaYogeswaran/BlinkRate.git
cd BlinkRate

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install numpy==1.26.4
pip install opencv-python==4.9.0.80 --no-deps
pip install mediapipe==0.10.14 --no-deps
pip install absl-py attrs flatbuffers protobuf sounddevice matplotlib
Important: Do not upgrade NumPy to 2.x. MediaPipe 0.10.14 needs NumPy 1.x.


## Run

```bash
python blinkrate.py
Controls:

- q — Quit and save session

- r — Reset session

- s — Save snapshot

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
