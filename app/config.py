import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure runtime directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "trained").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "exported").mkdir(parents=True, exist_ok=True)

# Video Stream Configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEFAULT_CAMERA_INDEX = 0

# Feature Extraction Thresholds
EAR_CLOSED_THRESHOLD = 0.21
MAR_YAWN_THRESHOLD = 0.65
HEAD_PITCH_NOD_THRESHOLD = 18.0   # Degrees down
HEAD_YAW_DISTRACTION_THRESHOLD = 25.0 # Degrees sideways

# PERCLOS Rolling Window Settings
PERCLOS_WINDOW_SECONDS = 30
PERCLOS_WATCH_THRESHOLD = 0.25
PERCLOS_WARNING_THRESHOLD = 0.40
PERCLOS_CRITICAL_THRESHOLD = 0.60

# Escalation & Hysteresis Timing (Sustained duration required)
WATCH_CONSECUTIVE_SECONDS = 3.0
WARNING_CONSECUTIVE_SECONDS = 5.0
CRITICAL_CONSECUTIVE_SECONDS = 10.0

# Alerting & Feedback
ALERT_COOLDOWN_SECONDS = 10.0
DATABASE_PATH = str(LOGS_DIR / "drowsiness.db")
ALERT_SOUND_PATH = str(DATA_DIR / "warning.wav")

# Landmark Indices for MediaPipe FaceMesh
# 6 key points per eye for Eye Aspect Ratio (EAR)
LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDS = [362, 385, 387, 263, 373, 380]

# Key points for Mouth Aspect Ratio (MAR)
# Upper lip top, lower lip bottom, inner corners
MOUTH_OUTER_IDS = [61, 291, 0, 17, 37, 267, 84, 314]
MOUTH_INNER_IDS = [78, 308, 13, 14, 82, 312, 87, 317]

# 3D Head Pose Landmark Indices (MediaPipe Face Mesh IDs)
# Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
