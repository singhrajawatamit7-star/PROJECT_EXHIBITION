import sys
import argparse
import time
from pathlib import Path

# Ensure current directory and project root are in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Safe import of CV libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from camera import Camera
from landmarks import FaceLandmarkDetector
from features import calculate_ear, calculate_mar, estimate_head_pose, landmark_to_pixel
from perclos import PerclosTracker
from alert_controller import AlertController, AlertState
from alerts import AlertActuator
from database import Database
from fatigue_monitor import FatigueMonitor
from hardware_alerts import HardwareAlerts
from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DEFAULT_CAMERA_INDEX,
    EAR_CLOSED_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    PERCLOS_WINDOW_SECONDS,
    PERCLOS_WATCH_THRESHOLD,
    PERCLOS_WARNING_THRESHOLD,
    PERCLOS_CRITICAL_THRESHOLD,
    WATCH_CONSECUTIVE_SECONDS,
    WARNING_CONSECUTIVE_SECONDS,
    CRITICAL_CONSECUTIVE_SECONDS,
    ALERT_COOLDOWN_SECONDS,
    DATABASE_PATH,
    ALERT_SOUND_PATH,
    LEFT_EYE_IDS,
    RIGHT_EYE_IDS,
    MOUTH_INNER_IDS
)


def get_eye_points(landmarks, eye_ids, width, height):
    return [
        landmark_to_pixel(landmarks[idx], width, height)
        for idx in eye_ids
    ]


def get_mouth_points(landmarks, width, height):
    p_left = landmark_to_pixel(landmarks[78], width, height)
    p_right = landmark_to_pixel(landmarks[308], width, height)
    p_top = landmark_to_pixel(landmarks[13], width, height)
    p_bottom = landmark_to_pixel(landmarks[14], width, height)
    return [p_left, p_right, p_top, p_bottom]


def draw_hud(frame, ear, mar, perclos, pitch, yaw, state_str, loop_ms, fps):
    """
    Renders a modern, high-contrast Edge-AI telemetry dashboard overlay on the frame.
    """
    if not HAS_CV2 or not hasattr(frame, "shape"):
        return

    height, width = frame.shape[:2]

    # Semi-transparent dark header banner for telemetry
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 82), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status badge styling based on alert level
    color_map = {
        "normal": (50, 205, 50),     # Vibrant Green
        "watch": (0, 215, 255),      # Amber/Yellow
        "warning": (0, 140, 255),    # Vibrant Orange
        "critical": (30, 30, 255),   # Neon Red
        "no_face": (160, 160, 160),  # Neutral Grey (face unavailable)
    }
    badge_color = color_map.get(state_str, (200, 200, 200))

    # Alert status pill
    status_text = f"STATE: {state_str.upper()}"
    cv2.rectangle(frame, (15, 12), (185, 42), badge_color, -1)
    cv2.putText(
        frame,
        status_text,
        (25, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Core metrics row 1
    metrics_str1 = f"EAR: {ear:.2f} (Thresh: {EAR_CLOSED_THRESHOLD:.2f})  |  MAR: {mar:.2f}"
    cv2.putText(
        frame,
        metrics_str1,
        (200, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA
    )

    # Core metrics row 2: PERCLOS progress and Head Pose
    metrics_str2 = f"PERCLOS: {perclos * 100:.1f}%  |  Pose: Pitch {pitch:+.1f} deg, Yaw {yaw:+.1f} deg"
    cv2.putText(
        frame,
        metrics_str2,
        (200, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    # PERCLOS visual gauge bar
    bar_x, bar_y, bar_w, bar_h = 15, 52, 170, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
    fill_w = int(bar_w * min(1.0, max(0.0, perclos)))
    cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1), badge_color, -1)

    # Performance bottom bar (Latency & FPS)
    perf_text = f"Edge Latency: {loop_ms:.1f}ms  |  FPS: {fps:.1f}  |  Press 'q' to exit"
    cv2.putText(
        frame,
        perf_text,
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser(description="Edge-AI Driver Drowsiness Detection System")
    parser.add_argument("--camera", default=DEFAULT_CAMERA_INDEX, help="Camera index or video file path")
    parser.add_argument("--no-gui", action="store_true", help="Run without graphical display (headless mode)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop execution after N frames (testing)")
    parser.add_argument("--telemetry-url", default=None, help="Remote HTTP API endpoint to POST live telemetry (e.g. http://127.0.0.1:8000/api/predictions)")
    args = parser.parse_args()

    if not HAS_CV2:
        print("\n[Driver Drowsiness AI] Notice: 'opencv-python' is not installed in the active Python environment.")
        print("To enable webcam capture, HUD rendering, and real-time inference:")
        print("  1. Create/activate a virtual environment: python -m venv .venv && .venv\\Scripts\\activate")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("For offline simulation and verification without OpenCV, run: python app/simulate.py\n")
        if not args.no_gui:
            return

    # Parse camera input
    camera_source = args.camera
    try:
        camera_source = int(camera_source)
    except ValueError:
        pass

    # Load trained machine learning model if available
    rf_model = None
    rf_model_path = current_dir.parent / "models" / "trained" / "drowsiness_random_forest.joblib"
    if rf_model_path.exists():
        try:
            import joblib
            rf_model = joblib.load(rf_model_path)
            print(f"[Driver Drowsiness AI] Loaded trained Random Forest model from {rf_model_path}")
        except Exception as e:
            print(f"[Driver Drowsiness AI] Could not load ML model ({e}). Continuing with rule-based engine.")

    print("[Driver Drowsiness AI] Initializing Edge-AI runtime modules...")
    camera = Camera(camera_source=camera_source, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    detector = FaceLandmarkDetector()
    perclos_tracker = PerclosTracker(window_seconds=PERCLOS_WINDOW_SECONDS)
    controller = AlertController(
        watch_threshold=PERCLOS_WATCH_THRESHOLD,
        warning_threshold=PERCLOS_WARNING_THRESHOLD,
        critical_threshold=PERCLOS_CRITICAL_THRESHOLD,
        watch_duration=WATCH_CONSECUTIVE_SECONDS,
        warning_duration=WARNING_CONSECUTIVE_SECONDS,
        critical_duration=CRITICAL_CONSECUTIVE_SECONDS,
        cooldown_seconds=ALERT_COOLDOWN_SECONDS
    )
    fatigue_monitor = FatigueMonitor(yawn_mar_threshold=MAR_YAWN_THRESHOLD)
    hardware_alerts = HardwareAlerts()
    actuator = AlertActuator(sound_path=ALERT_SOUND_PATH)
    database = Database(database_path=DATABASE_PATH, async_mode=True)

    frame_count = 0
    prev_state = AlertState.NORMAL
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0

    print("[Driver Drowsiness AI] Loop active. Ready for driver monitoring.")

    try:
        while True:
            loop_start = time.time()
            frame = camera.read()
            if frame is None:
                print("[Driver Drowsiness AI] Stream completed or source unavailable.")
                break

            height = getattr(frame, "shape", [480, 640])[0]
            width = getattr(frame, "shape", [480, 640])[1]

            landmarks = detector.detect(frame) if HAS_CV2 else None
            ear = 0.0
            mar = 0.0
            pitch, yaw, roll = 0.0, 0.0, 0.0
            eyes_closed = False
            perclos = 0.0
            fatigue_risk = 0.0
            fatigue_reasons = []

            face_present = landmarks is not None
            if face_present:
                # 1. Calculate EAR from the existing facial landmarks.
                left_eye = get_eye_points(landmarks, LEFT_EYE_IDS, width, height)
                right_eye = get_eye_points(landmarks, RIGHT_EYE_IDS, width, height)
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                eyes_closed = ear < EAR_CLOSED_THRESHOLD

                # 2. Calculate mouth opening and head pose.
                mouth_pts = get_mouth_points(landmarks, width, height)
                mar = calculate_mar(mouth_pts)
                pitch, yaw, roll = estimate_head_pose(landmarks, width, height)

                # 3. Track eye closure only when a face is available.
                perclos = perclos_tracker.update(eyes_closed)

                # 4. Draw facial contours for active tracking visualization
                if HAS_CV2 and HAS_NUMPY:
                    cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(mouth_pts, dtype=np.int32)], True, (255, 150, 0), 1)

            # Update supplementary fatigue indicators (yawn & head-droop)
            fatigue_risk, fatigue_reasons = fatigue_monitor.update(mar, pitch, face_present=face_present)

            # Evaluate state safely based on face presence and ML classifier
            model_tag = "rule_v2"
            if not face_present:
                alert_state_str = "no_face"
                state = AlertState.NORMAL
            else:
                closed_duration = perclos_tracker.get_current_closed_duration()

                # ML Random Forest Inference scoring if model is loaded
                ml_risk = None
                if rf_model is not None:
                    try:
                        feat_row = [[ear, perclos, mar, closed_duration, pitch, yaw]]
                        prob = float(rf_model.predict_proba(feat_row)[0][1])
                        ml_risk = prob
                        model_tag = "random_forest_v1"
                    except Exception:
                        pass

                effective_risk = max(fatigue_risk, ml_risk) if ml_risk is not None else fatigue_risk

                state = controller.update(
                    perclos,
                    continuous_closed_seconds=closed_duration,
                    external_risk_score=effective_risk,
                )
                alert_state_str = state.value

            # Update physical GPIO hardware output if equipped
            hardware_alerts.set_state(alert_state_str)

            # 6. Physical & Acoustic Actuation
            if state != AlertState.NORMAL and (state != prev_state or (time.time() - controller.last_alert_time) < 0.1):
                actuator.trigger(state)
                database.log_alert(alert_state_str, "audio")
                if hardware_alerts.enabled:
                    database.log_alert(alert_state_str, "led")

            prev_state = state

            # 7. Asynchronous Telemetry & Predictions Logging
            database.log_prediction(
                perclos=perclos,
                ear=ear,
                alert_state=alert_state_str,
                mar=mar,
                head_pitch=pitch,
                head_yaw=yaw,
                model_version=model_tag
            )

            # Remote API Telemetry Uplink (if --telemetry-url is provided)
            if args.telemetry_url and frame_count % 3 == 0:
                try:
                    import urllib.request
                    uplink_data = json.dumps({
                        "ear": ear,
                        "perclos": perclos,
                        "mar": mar,
                        "head_pitch": pitch,
                        "head_yaw": yaw,
                        "alert_state": alert_state_str,
                        "model_version": model_tag
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        args.telemetry_url,
                        data=uplink_data,
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(req, timeout=0.1)
                except Exception:
                    pass


            # Latency and FPS computation
            elapsed_ms = (time.time() - loop_start) * 1000.0
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            # 8. Render HUD
            if HAS_CV2:
                draw_hud(frame, ear, mar, perclos, pitch, yaw, alert_state_str, elapsed_ms, fps)

                if not args.no_gui:
                    cv2.imshow("Driver Drowsiness AI Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[Driver Drowsiness AI] User requested quit.")
                        break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                print(f"[Driver Drowsiness AI] Reached target frame limit ({args.max_frames}).")
                break

    finally:
        camera.release()
        detector.close()
        database.close()
        actuator.close()
        hardware_alerts.close()
        if HAS_CV2 and not args.no_gui:
            cv2.destroyAllWindows()
        print("[Driver Drowsiness AI] Shutdown completed successfully.")


# ---------------------------------------------------------------------------
# Top-level Web / Serverless WSGI Application Exports (Vercel, Gunicorn, AWS)
# ---------------------------------------------------------------------------
import json


DASHBOARD_HTML = """<!-- Telemetry dashboard served via root index.html -->"""


def create_wsgi_app():
    """Delegates WSGI requests to the dedicated api.index service."""
    from api.index import application
    return application


# Export top-level variables required by Vercel, AWS Lambda, and WSGI servers:
app = create_wsgi_app()
application = app
handler = app


if __name__ == "__main__":
    main()
