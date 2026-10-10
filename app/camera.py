import time

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


class Camera:
    """
    Driver-facing camera capture wrapper with support for hardware webcams,
    video files, and synthetic fallback test streams.
    """

    def __init__(self, camera_source=0, width=640, height=480, fallback_synthetic=True):
        self.width = width
        self.height = height
        self.fallback_synthetic = fallback_synthetic
        self.is_synthetic = False
        self.cap = None

        if HAS_CV2 and (isinstance(camera_source, str) or isinstance(camera_source, int)):
            try:
                self.cap = cv2.VideoCapture(camera_source)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception as e:
                print(f"[Camera] Error initializing capture: {e}")

        if self.cap is None or not (HAS_CV2 and self.cap.isOpened()):
            if fallback_synthetic:
                print(f"[Camera] Notice: Hardware camera '{camera_source}' unavailable. Initializing synthetic telemetry stream.")
                self.is_synthetic = True
                self._synthetic_start = time.time()
            else:
                raise RuntimeError(f"Unable to open camera source: {camera_source}")

    def read(self):
        """
        Reads next frame as BGR numpy array. Returns None if stream ends.
        """
        if self.is_synthetic:
            return self._generate_synthetic_frame()

        if not HAS_CV2 or self.cap is None:
            return None

        success, frame = self.cap.read()
        if not success:
            return None

        # Resize if dimensions don't match requested
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def _generate_synthetic_frame(self):
        """
        Generates an animated test card when no physical webcam is present,
        allowing end-to-end loop verification without crashing.
        """
        if not HAS_NUMPY or not HAS_CV2:
            time.sleep(0.03)
            # Lightweight dummy object with shape attribute
            class DummyFrame:
                shape = (self.height, self.width, 3)
            return DummyFrame()

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background dark blue gradient
        for y in range(self.height):
            frame[y, :] = [int(30 + 20 * (y / self.height)), int(20), int(15)]

        elapsed = time.time() - self._synthetic_start
        cv2.putText(
            frame,
            "CAMERA SIMULATION MODE (Synthetic Driver Stream)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2
        )
        cv2.putText(
            frame,
            f"Elapsed: {elapsed:.1f}s | 640x480 Virtual Sensor",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

        # Silhouette
        center_x = self.width // 2
        center_y = self.height // 2 + 20
        cv2.ellipse(frame, (center_x, center_y), (80, 105), 0, 0, 360, (70, 70, 90), -1)
        eye_y = center_y - 20
        blink = (int(elapsed) % 4 == 0) and ((elapsed % 1.0) < 0.3)
        eye_h = 2 if blink else 10
        cv2.ellipse(frame, (center_x - 35, eye_y), (14, eye_h), 0, 0, 360, (220, 220, 240), -1)
        cv2.ellipse(frame, (center_x + 35, eye_y), (14, eye_h), 0, 0, 360, (220, 220, 240), -1)
        mouth_y = center_y + 45
        cv2.ellipse(frame, (center_x, mouth_y), (25, 6), 0, 0, 360, (150, 100, 100), -1)

        time.sleep(0.03)  # Approx 30 FPS
        return frame

    def is_opened(self):
        return self.is_synthetic or (HAS_CV2 and self.cap is not None and self.cap.isOpened())

    def release(self):
        if HAS_CV2 and self.cap is not None:
            self.cap.release()
            self.cap = None
