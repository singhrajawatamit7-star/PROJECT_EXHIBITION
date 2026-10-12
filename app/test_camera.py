"""
Camera Hardware Diagnostic Utility
Verifies real webcam hardware access directly with OpenCV VideoCapture.
Unlike the main application's synthetic fallback, it fails clearly if the camera cannot be opened.
"""

import argparse

try:
    import cv2
except ImportError:
    raise SystemExit("ERROR: OpenCV ('cv2') is required for camera diagnostic. Install it via 'pip install opencv-python'.")


def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Camera Diagnostic")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"ERROR: Cannot open camera index {args.camera}")

    print(f"Camera index {args.camera} opened successfully. Press 'q' in the window to exit.")
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("ERROR: Failed to read camera frame.")
                break
            cv2.putText(
                frame,
                "CAMERA TEST: OK",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Diagnostic", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
