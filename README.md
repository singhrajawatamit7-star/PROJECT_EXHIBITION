# Edge-AI Driver Drowsiness Detection and Alert System

An end-to-end Edge-AI embedded driver monitoring system that detects early signs of driver fatigue and microsleep in real-time using driver-facing computer vision and multi-modal sensory fusion.

---

## 1. System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                         │
│  Camera (720p/1080p) | Wearable (ECG/EEG) | CAN-Bus (Future)│
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    PREPROCESSING                            │
│ Face Mesh (468 landmarks) | EAR | MAR | PERCLOS | Head Pose │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    EDGE INFERENCE                           │
│ Rule-Based Classifier → Baseline RF → Temporal LSTM/ONNX    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    ALERT CONTROLLER                         │
│ Normal → Watch (Visual) → Warning (Audio) → Critical (All)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              OUTPUTS & NON-BLOCKING LOGGING                 │
│ LED Indicator | Buzzer/Speaker | Haptic Motor | SQLite DB   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Features & Metrics

1. **Eye Aspect Ratio (EAR)**:
   Measures vertical eye aperture relative to horizontal width across 6 contour landmarks per eye:
   $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
   Default closed threshold: `EAR < 0.21`.

2. **PERCLOS (Percentage of Eye Closure)**:
   Calculates the proportion of time eyes remain classified as closed across a rolling 30-second window:
   - **Normal**: $\text{PERCLOS} < 0.25$
   - **Watch**: $0.25 \le \text{PERCLOS} < 0.40$ (sustained $\ge 3$ s)
   - **Warning**: $0.40 \le \text{PERCLOS} < 0.60$ (sustained $\ge 5$ s)
   - **Critical**: $\text{PERCLOS} \ge 0.60$ (sustained $\ge 10$ s) or immediate microsleep ($\ge 2.5$ s continuous closure).

3. **Mouth Aspect Ratio (MAR)**:
   Quantifies mouth aperture to identify yawning and prolonged mouth openings.

4. **3D Head Pose Estimation**:
   Uses OpenCV `solvePnP` with standard 3D anthropometric face models to track pitch (head nodding/drooping) and yaw (inattention/distraction).

5. **Staged Escalating Alert Controller**:
   - Built-in **hysteresis**, **sustained duration timers**, and **alarm cooldown** to eliminate false triggers caused by normal blinking and prevent alarm fatigue.

6. **Non-Blocking Asynchronous Database Logging**:
   - Queued thread worker guarantees $< 0.1$ ms database write overhead, preventing pipeline stutter or dropped video frames.

---

## 3. Directory Structure

```text
driver-drowsiness-ai/
├── app/
│   ├── __init__.py
│   ├── config.py             # Global thresholds, timing, and path configurations
│   ├── camera.py             # Camera capture with synthetic fallback simulation
│   ├── landmarks.py          # MediaPipe FaceMesh landmark detector
│   ├── features.py           # EAR, MAR, Euclidean, and 3D Head Pose algorithms
│   ├── perclos.py            # Sliding window PERCLOS and blink tracker
│   ├── alert_controller.py   # Staged alert state machine with hysteresis
│   ├── alerts.py             # Audio tone synthesis and multi-modal actuator
│   ├── database.py           # Asynchronous SQLite event and metrics recorder
│   ├── main.py               # Main application entry point with HUD overlay
│   └── simulate.py           # Driver fatigue session simulator & dataset generator
│
├── data/
│   ├── raw/                  # Raw input videos / recordings
│   ├── processed/            # Extracted feature CSVs
│   ├── logs/                 # SQLite database (drowsiness.db)
│   └── warning.wav           # Auto-synthesized warning audio tone
│
├── models/
│   ├── trained/              # Trained models (Random Forest, ONNX)
│   ├── exported/             # Optimized edge binaries (TensorRT, TFLite)
│   └── train_baseline.py     # Machine learning training pipeline
│
├── tests/
│   ├── __init__.py
│   ├── test_features.py      # Unit tests for EAR, MAR, distance metrics
│   ├── test_perclos.py       # Unit tests for rolling window and eviction
│   ├── test_alert_controller.py # Tests for staged alerts and cooldowns
│   └── test_database.py      # Tests for synchronous and asynchronous logging
│
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore
```

---

## 4. Setup and Installation

### Step 1: Create and Activate Virtual Environment

**Windows:**
```powershell
cd driver-drowsiness-ai
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
cd driver-drowsiness-ai
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Usage Guide

### 1. Test Camera Hardware

```bash
# Test camera capture window
python app/test_camera.py
```

### 2. Run Real-Time Detection (Default Webcam)

```bash
python app/main.py
```
*Press `q` anytime to safely quit.*

### 2. Run with a Specific Camera or Video File

```bash
# Camera index 1
python app/main.py --camera 1

# Video file
python app/main.py --camera path/to/driver_video.mp4
```

### 3. Run in Headless Mode (e.g., Embedded Edge or Testing)

```bash
python app/main.py --no-gui
```

### 4. Run Fatigue Simulation & Dataset Generator

```bash
python app/simulate.py
```
This simulates a fatigue progression and generates `data/processed/features.csv`.

### 5. Train Baseline Machine Learning Classifier

```bash
python models/train_baseline.py
```
Trains a Random Forest classifier with 300 estimators on the engineered feature vector and exports it to `models/trained/drowsiness_random_forest.joblib`.

### 6. Inspect SQLite Database Logs

```bash
# Query recent predictions and alerts directly from SQLite
sqlite3 data/logs/drowsiness.db "SELECT timestamp, perclos, ear, alert_state FROM predictions ORDER BY id DESC LIMIT 10;"
sqlite3 data/logs/drowsiness.db "SELECT timestamp, level, channel FROM alerts ORDER BY id DESC LIMIT 10;"
```

### 7. Run Unit & Integration Tests

```bash
# Standard unittest runner (zero extra dependencies)
python -m unittest discover -s tests -p "test_*.py"

# Or with pytest
pytest tests/ -v
```

---

## 6. Privacy & Safety Guidelines

- **Edge-First Processing**: All facial landmarks and video frames are analyzed locally in device memory and immediately discarded.
- **Biometric Minimization**: Only extracted numerical features (`EAR`, `PERCLOS`, `MAR`) and alert logs are saved to SQLite. Raw camera video is never persisted unless explicitly configured for calibration.
- **Safety Boundary**: This software is intended for research, simulator, and stationary vehicle testing. Do not rely upon this prototype as a certified primary safety system on public roads.
