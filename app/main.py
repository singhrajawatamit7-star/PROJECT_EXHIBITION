from fatigue_monitor import FatigueMonitor
import argparse
import sys
import time
from pathlib import Path

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

# Ensure parent directory is in sys.path when running as script
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

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

            # Evaluate state safely based on face presence
            if not face_present:
                alert_state_str = "no_face"
                state = AlertState.NORMAL
            else:
                closed_duration = perclos_tracker.get_current_closed_duration()
                state = controller.update(
                    perclos,
                    continuous_closed_seconds=closed_duration,
                    external_risk_score=fatigue_risk,
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
                model_version="rule_v2"
            )


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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Edge-AI Driver Drowsiness Monitor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0d14;
      --card-bg: rgba(18, 24, 38, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #38bdf8;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at 50% -20%, #1e293b, var(--bg) 80%);
      color: var(--text);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .container { width: 100%; max-width: 1040px; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2.5rem;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.4rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      background: rgba(34, 197, 94, 0.15);
      color: var(--success);
      border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); box-shadow: 0 0 10px var(--success); }
    h1 { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; }
    p.subtitle { color: var(--text-muted); font-size: 0.95rem; margin-top: 0.3rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1.25rem; margin-bottom: 2rem; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 1.5rem;
      backdrop-filter: blur(16px);
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .card-label { font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .card-val { font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0 0.2rem; font-family: 'JetBrains Mono', monospace; }
    .card-sub { font-size: 0.8rem; color: var(--text-muted); }
    .interactive-card { margin-bottom: 2rem; }
    .slider-group { margin: 1.5rem 0; }
    .slider-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.5rem; }
    input[type=range] { width: 100%; height: 6px; border-radius: 3px; background: #334155; outline: none; -webkit-appearance: none; }
    input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--primary); cursor: pointer; }
    .state-pill {
      display: inline-block;
      padding: 0.6rem 1.4rem;
      border-radius: 0.5rem;
      font-size: 1.1rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-top: 0.5rem;
    }
    .state-normal { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }
    .state-watch { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }
    .state-warning { background: rgba(249, 115, 22, 0.25); color: #f97316; border: 1px solid #f97316; }
    .state-critical { background: rgba(239, 68, 68, 0.25); color: var(--danger); border: 1px solid var(--danger); animation: pulse 1s infinite alternate; }
    @keyframes pulse { from { opacity: 0.8; } to { opacity: 1; transform: scale(1.02); } }
    .links-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem; }
    .api-btn {
      padding: 0.5rem 1rem;
      background: #1e293b;
      border: 1px solid var(--card-border);
      border-radius: 0.5rem;
      color: var(--primary);
      text-decoration: none;
      font-size: 0.85rem;
      font-family: 'JetBrains Mono', monospace;
      transition: background 0.2s;
    }
    .api-btn:hover { background: #334155; }
    footer { text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 3rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Edge-AI Driver Monitor</h1>
        <p class="subtitle">Real-time driver drowsiness detection & staged escalating alert system</p>
      </div>
      <div class="badge"><div class="badge-dot"></div> Edge-AI Online</div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-label">Eye Aspect Ratio (EAR)</div>
        <div class="card-val" style="color: #38bdf8;">0.28</div>
        <div class="card-sub">Threshold: < 0.21 (Eyes Closed)</div>
      </div>
      <div class="card">
        <div class="card-label">PERCLOS (30s Window)</div>
        <div class="card-val" style="color: #22c55e;">12.4%</div>
        <div class="card-sub">Warning: &ge; 40% | Critical: &ge; 60%</div>
      </div>
      <div class="card">
        <div class="card-label">Mouth Aspect Ratio (MAR)</div>
        <div class="card-val" style="color: #a855f7;">0.22</div>
        <div class="card-sub">Yawn Threshold: &ge; 0.65</div>
      </div>
      <div class="card">
        <div class="card-label">Inference Cycle</div>
        <div class="card-val" style="color: #10b981;">28.5 ms</div>
        <div class="card-sub">Target Latency: < 100 ms</div>
      </div>
    </div>

    <div class="card interactive-card">
      <h3 style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.3rem;">Interactive Alert State Machine Simulator</h3>
      <p style="color: var(--text-muted); font-size: 0.85rem;">Simulate driver fatigue metrics to observe staged hysteresis and emergency triggers in real time.</p>
      
      <div class="slider-group">
        <div class="slider-label">
          <span>PERCLOS Ratio (% Eye Closure)</span>
          <span id="perclos-num" style="font-family: 'JetBrains Mono'; font-weight: 600; color: var(--primary);">15%</span>
        </div>
        <input type="range" id="perclos-slider" min="0" max="100" value="15" oninput="updateSim()">
      </div>

      <div class="slider-group">
        <div class="slider-label">
          <span>Continuous Eye Closure Duration (Microsleep)</span>
          <span id="dur-num" style="font-family: 'JetBrains Mono'; font-weight: 600; color: var(--primary);">0.0s</span>
        </div>
        <input type="range" id="dur-slider" min="0" max="50" value="0" step="1" oninput="updateSim()">
      </div>

      <div style="margin-top: 1.25rem;">
        <span style="font-size: 0.85rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Evaluated State:</span>
        <div id="state-display" class="state-pill state-normal">NORMAL (Clear to Drive)</div>
      </div>
    </div>

    <div class="card">
      <h3 style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.5rem;">Cloud API Endpoints</h3>
      <p style="color: var(--text-muted); font-size: 0.85rem;">Access real-time telemetry and database logs via REST endpoints:</p>
      <div class="links-row">
        <a class="api-btn" href="/health" target="_blank">GET /health</a>
        <a class="api-btn" href="/api/status" target="_blank">GET /api/status</a>
        <a class="api-btn" href="/api/predictions" target="_blank">GET /api/predictions</a>
        <a class="api-btn" href="/api/alerts" target="_blank">GET /api/alerts</a>
      </div>
    </div>

    <footer>
      Edge-AI Driver Drowsiness Detection System &bull; <a href="https://github.com/singhrajawatamit7-star/PROJECT_EXHIBITION" target="_blank" style="color: var(--primary); text-decoration: none;">GitHub Repository</a>
    </footer>
  </div>

  <script>
    function updateSim() {
      const p = parseInt(document.getElementById('perclos-slider').value) / 100;
      const d = parseInt(document.getElementById('dur-slider').value) / 10;
      document.getElementById('perclos-num').textContent = Math.round(p * 100) + '%';
      document.getElementById('dur-num').textContent = d.toFixed(1) + 's';

      const pill = document.getElementById('state-display');
      if (d >= 2.5 || p >= 0.60) {
        pill.className = 'state-pill state-critical';
        pill.textContent = 'CRITICAL (Audio + Visual + Haptic Alert)';
      } else if (p >= 0.40) {
        pill.className = 'state-pill state-warning';
        pill.textContent = 'WARNING (Audio Alarm)';
      } else if (p >= 0.25) {
        pill.className = 'state-pill state-watch';
        pill.textContent = 'WATCH (Visual Indicator)';
      } else {
        pill.className = 'state-pill state-normal';
        pill.textContent = 'NORMAL (Driver Alert)';
      }
    }
  </script>
</body>
</html>
"""


def create_wsgi_app():
    """
    Standard WSGI/HTTP callable for cloud serverless deployments (Vercel, Render, Gunicorn).
    Provides health-check, telemetry, threshold parameters, and prediction query endpoints.
    """
    def wsgi_handler(environ, start_response):
        path = environ.get('PATH_INFO', '/') or '/'
        method = environ.get('REQUEST_METHOD', 'GET')
        accept = environ.get('HTTP_ACCEPT', '')

        # HTML landing page for browser requests
        if path == '/' and 'text/html' in accept:
            headers = [
                ('Content-Type', 'text/html; charset=utf-8'),
                ('Access-Control-Allow-Origin', '*'),
            ]
            start_response('200 OK', headers)
            return [DASHBOARD_HTML.encode('utf-8')]

        headers = [
            ('Content-Type', 'application/json'),
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type'),
        ]

        if method == 'OPTIONS':
            start_response('204 No Content', headers)
            return [b'']

        if path in ['/', '/health']:
            payload = {
                "status": "online",
                "service": "driver-drowsiness-ai",
                "version": "1.0.0",
                "description": "Edge-AI Driver Drowsiness Detection and Alert System",
                "endpoints": {
                    "/health": "Service health probe",
                    "/api/status": "System threshold configurations",
                    "/api/predictions": "Latest inference log records",
                    "/api/alerts": "Recent alert notifications"
                }
            }
            start_response('200 OK', headers)
            return [json.dumps(payload, indent=2).encode('utf-8')]

        elif path == '/api/status':
            payload = {
                "status": "ready",
                "thresholds": {
                    "ear_closed_threshold": EAR_CLOSED_THRESHOLD,
                    "mar_yawn_threshold": MAR_YAWN_THRESHOLD,
                    "perclos_watch_threshold": PERCLOS_WATCH_THRESHOLD,
                    "perclos_warning_threshold": PERCLOS_WARNING_THRESHOLD,
                    "perclos_critical_threshold": PERCLOS_CRITICAL_THRESHOLD,
                    "perclos_window_seconds": PERCLOS_WINDOW_SECONDS,
                    "alert_cooldown_seconds": ALERT_COOLDOWN_SECONDS
                }
            }
            start_response('200 OK', headers)
            return [json.dumps(payload, indent=2).encode('utf-8')]

        elif path == '/api/predictions':
            preds = []
            try:
                db = Database(database_path=DATABASE_PATH, async_mode=False)
                recent = db.get_recent_predictions(limit=15)
                db.close()
                for r in recent:
                    preds.append({
                        "timestamp": r[0],
                        "perclos": r[1],
                        "ear": r[2],
                        "alert_state": r[3]
                    })
            except Exception as e:
                preds = []
            start_response('200 OK', headers)
            return [json.dumps({"predictions": preds}).encode('utf-8')]

        elif path == '/api/alerts':
            alerts = []
            try:
                db = Database(database_path=DATABASE_PATH, async_mode=False)
                recent = db.get_recent_alerts(limit=15)
                db.close()
                for r in recent:
                    alerts.append({
                        "timestamp": r[0],
                        "level": r[1],
                        "channel": r[2]
                    })
            except Exception as e:
                alerts = []
            start_response('200 OK', headers)
            return [json.dumps({"alerts": alerts}).encode('utf-8')]

        else:
            start_response('404 Not Found', headers)
            return [json.dumps({"error": "Route not found", "path": path}).encode('utf-8')]

    return wsgi_handler


# Export top-level variables required by Vercel, AWS Lambda, and WSGI servers:
app = create_wsgi_app()
application = app
handler = app


if __name__ == "__main__":
    main()
