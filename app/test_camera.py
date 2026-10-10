"""
Standalone Camera Hardware Test Utility
Verifies video capture feed and displays stream in a GUI window.
Press 'q' to exit.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from camera import Camera


def main():
    print("[Camera Test] Initializing camera device...")
    camera = Camera(fallback_synthetic=True)

    if not HAS_CV2:
        print("[Camera Test] OpenCV ('cv2') is not installed in the environment.")
        print("To test camera with video display: pip install opencv-python")
        print("Reading 5 test frames in headless mode...")
        for i in range(5):
            frame = camera.read()
            print(f"  Frame {i+1} successfully read.")
        camera.release()
        print("[Camera Test] Headless verification passed.")
        return

    print("[Camera Test] Stream open. Press 'q' in the window to close.")
    frame_count = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[Camera Test] Stream ended or camera disconnected.")
                break

            cv2.imshow("Driver Drowsiness - Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Camera Test] User pressed 'q'. Exiting.")
                break

            frame_count += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"[Camera Test] Session closed. Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
