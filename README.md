# 👻 GhostDesk

**Control your computer without touching your mouse.**

GhostDesk is a real-time **touchless desktop control system** built with Python, OpenCV, MediaPipe, and PyAutoGUI. It uses your webcam to track hand landmarks and translates specific hand gestures into mouse actions such as cursor movement, clicking, scrolling, and locking the gesture interface.

## ✨ Features

* 🖐️ Real-time hand tracking
* 🖱️ Touchless cursor movement
* 🤏 Left-click gesture
* 🤏 Right-click gesture
* ✌️ Gesture-based scrolling
* ✊ Gesture control lock
* 🖐️ Open-palm unlock
* 🎯 Cursor smoothing
* 📐 Distance-normalized gesture detection
* 🖥️ Real-time futuristic HUD
* 📊 FPS and tracking status display
* 🔒 Accidental-action protection

## 🎮 Gesture Controls

| Gesture                  | Action          |
| ------------------------ | --------------- |
| ☝️ Move index finger     | Move cursor     |
| 🤏 Thumb + Index         | Left Click      |
| 🤏 Thumb + Middle        | Right Click     |
| ✌️ Index + Middle raised | Scroll Mode     |
| ✊ Hold Fist              | Lock Controls   |
| 🖐️ Hold Open Palm       | Unlock Controls |

### Keyboard Controls

| Key | Action         |
| --- | -------------- |
| `H` | Toggle HUD     |
| `Q` | Quit GhostDesk |

## 🧠 How It Works

GhostDesk captures live video from your webcam using **OpenCV**.

Each frame is processed by **MediaPipe Hands**, which detects 21 landmarks across your hand.

The program then analyzes:

* Finger positions
* Finger extension states
* Thumb-to-finger distances
* Palm movement
* Hand scale
* Gesture duration

Recognized gestures are translated into desktop actions using **PyAutoGUI**.

### Basic Pipeline

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Tracking
   ↓
21 Hand Landmarks
   ↓
Gesture Recognition
   ↓
Gesture State Machine
   ↓
PyAutoGUI
   ↓
Desktop Control
```

## 🛠️ Tech Stack

* **Python 3.12**
* **OpenCV**
* **MediaPipe**
* **PyAutoGUI**
* **NumPy**

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/GhostDesk.git
cd GhostDesk
```

Install the dependencies:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

## 🚀 Run GhostDesk

```bash
python ghostdesk.py
```

On Windows with Python 3.12:

```bash
py -3.12 ghostdesk.py
```

Your webcam should open and the **GhostDesk // Vision Interface** HUD will appear.

## 🔐 Gesture Lock

GhostDesk includes an internal gesture-control lock to reduce accidental actions.

Hold a:

```text
✊ FIST
```

for approximately **0.35 seconds** to lock gesture controls.

To unlock, hold:

```text
🖐️ OPEN PALM
```

for approximately **0.5 seconds**.

> This locks GhostDesk's gesture controls. It does **not** lock the Windows operating system.

## 🎯 Cursor Mapping

GhostDesk uses an active region inside the webcam frame and maps your index-finger position to the full desktop.

Cursor movement also includes:

* Exponential moving-average smoothing
* Adjustable sensitivity
* Dead-zone filtering

This helps reduce small hand movements and cursor jitter.

## ⚙️ Customization

Important parameters can be modified near the top of `ghostdesk.py`:

```python
CURSOR_SMOOTHING = 0.35
CURSOR_SENSITIVITY = 1.3

PINCH_THRESHOLD = 0.45

SCROLL_SPEED = 900

LOCK_HOLD_TIME = 0.35
UNLOCK_HOLD_TIME = 0.5

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.6
```

These values can be tuned depending on your webcam, lighting conditions, and preferred gesture sensitivity.

## 🔮 Future Ideas

Potential upgrades for GhostDesk:

* 🫳 Drag-and-drop gestures
* 🔊 Gesture-based volume control
* 🔍 Pinch-to-zoom
* 🪟 Window switching gestures
* 🎵 Media controls
* ✋ Custom gesture profiles
* 🖥️ Multi-monitor support
* 🎨 Improved sci-fi HUD
* 🤖 ML-based custom gesture recognition

## 📸 Demo

Add a GIF or video of GhostDesk here:

```markdown
![GhostDesk Demo](assets/ghostdesk-demo.gif)
```

A short demo showing cursor movement, clicking, scrolling and the lock/unlock gesture will make the repository much easier to understand.

## ⚠️ Notes

Performance may vary depending on:

* Webcam quality
* Lighting
* Background
* Hand visibility
* System performance

For best results, keep your hand clearly visible to the webcam and use GhostDesk in a well-lit environment.

## 📄 License

This project is intended for learning, experimentation, and personal development.

---

## 👻 GhostDesk

**Your hand is the controller.**
