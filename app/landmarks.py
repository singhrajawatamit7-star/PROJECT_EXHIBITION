try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class FaceLandmarkDetector:
    """
    Detects 468+ high-density 3D facial landmarks using MediaPipe FaceMesh.
    Provides fallback handling if MediaPipe is not yet installed.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.face_mesh = None
        if HAS_MEDIAPIPE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            except Exception as e:
                print(f"[FaceLandmarkDetector] Initialization notice: {e}")
        else:
            print("[FaceLandmarkDetector] Notice: 'mediapipe' package not installed in environment.")

    def detect(self, frame):
        """
        Processes BGR frame and returns normalized landmark list or None.
        """
        if self.face_mesh is None or not HAS_CV2:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        return results.multi_face_landmarks[0].landmark

    def close(self):
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.face_mesh = None
