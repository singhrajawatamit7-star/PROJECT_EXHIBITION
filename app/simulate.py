"""
Driver Fatigue Simulation & Dataset Generator
Simulates realistic driver telemetry, eye closure sequences, yawning, and head nod patterns.
Generates training dataset data/processed/features.csv and exercises the full edge pipeline.
"""

import time
import random
import csv
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from perclos import PerclosTracker
from alert_controller import AlertController, AlertState
from database import Database
from config import DATABASE_PATH, DATA_DIR


def generate_feature_dataset(output_path=None, num_samples=2000):
    """
    Generates a realistic synthetic dataset for ML model training and evaluation.
    Columns: ear, perclos_30s, mar, blink_duration, head_pitch, head_yaw, label (0=alert, 1=drowsy)
    """
    if output_path is None:
        output_path = DATA_DIR / "processed" / "features.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # 60% Alert, 40% Drowsy
    for i in range(num_samples):
        is_drowsy = random.random() < 0.40

        if not is_drowsy:
            ear = random.gauss(0.33, 0.04)
            ear = max(0.24, min(0.45, ear))
            perclos = random.gauss(0.08, 0.04)
            perclos = max(0.0, min(0.22, perclos))
            mar = random.gauss(0.22, 0.06)
            mar = max(0.10, min(0.40, mar))
            blink_dur = random.gauss(0.18, 0.05)
            blink_dur = max(0.08, min(0.35, blink_dur))
            head_pitch = random.gauss(0.0, 4.0)
            head_yaw = random.gauss(0.0, 5.0)
            label = 0
        else:
            ear = random.gauss(0.18, 0.05)
            ear = max(0.05, min(0.26, ear))
            perclos = random.gauss(0.52, 0.12)
            perclos = max(0.30, min(0.95, perclos))
            is_yawn = random.random() < 0.35
            mar = random.gauss(0.72, 0.08) if is_yawn else random.gauss(0.25, 0.08)
            mar = max(0.15, min(1.10, mar))
            blink_dur = random.gauss(0.65, 0.25)
            blink_dur = max(0.35, min(2.5, blink_dur))
            head_pitch = random.gauss(14.0, 7.0)  # Head nodding downward
            head_yaw = random.gauss(6.0, 8.0)
            label = 1

        rows.append({
            "ear": round(ear, 4),
            "perclos_30s": round(perclos, 4),
            "mar": round(mar, 4),
            "blink_duration": round(blink_dur, 4),
            "head_pitch": round(head_pitch, 2),
            "head_yaw": round(head_yaw, 2),
            "label": label
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Dataset Generator] Generated {num_samples} labeled feature samples at: {output_path}")
    return str(output_path)


def run_pipeline_simulation(duration_seconds=15):
    """
    Simulates a 15-second accelerated driving session through AlertController and Database.
    Demonstrates Normal -> Watch -> Warning -> Critical escalation.
    """
    print(f"\n[Simulation] Running {duration_seconds}-second fatigue progression simulation...")
    tracker = PerclosTracker(window_seconds=30)
    controller = AlertController(
        watch_threshold=0.25,
        warning_threshold=0.40,
        critical_threshold=0.60,
        watch_duration=1.0,    # Accelerated for fast validation
        warning_duration=2.0,
        critical_duration=3.0,
        cooldown_seconds=3.0
    )
    db = Database(DATABASE_PATH, async_mode=False)

    start_sim = 1000.0
    for sec in range(duration_seconds):
        t = start_sim + sec
        # Progressively worsen closure:
        if sec < 4:
            # Alert phase
            ear = 0.35
            closed = False
        elif sec < 8:
            # Mild fatigue
            ear = 0.20
            closed = True
        elif sec < 12:
            # Significant drowsiness
            ear = 0.16
            closed = True
        else:
            # Microsleep
            ear = 0.10
            closed = True

        perclos = tracker.update(closed, timestamp=t)
        closed_dur = tracker.get_current_closed_duration(current_time=t)
        state = controller.update(perclos, continuous_closed_seconds=closed_dur, current_time=t)

        db.log_prediction(
            perclos=perclos,
            ear=ear,
            alert_state=state.value,
            mar=0.25,
            model_version="sim_v1"
        )
        print(f"  Sec {sec:02d} | EAR: {ear:.2f} | PERCLOS: {perclos*100:.1f}% | Closed Dur: {closed_dur:.1f}s | State: {state.value.upper()}")

    db.close()
    print("[Simulation] Completed simulation successfully.")


if __name__ == "__main__":
    generate_feature_dataset()
    run_pipeline_simulation()
