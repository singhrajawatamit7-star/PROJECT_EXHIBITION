import math

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from landmarks import FaceLandmarkDetector
except ImportError:
    try:
        from app.landmarks import FaceLandmarkDetector
    except ImportError:
        FaceLandmarkDetector = None


def euclidean(point_a, point_b):
    """Euclidean distance between two 2D or 3D points."""
    if len(point_a) == 2:
        return float(math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))
    return float(math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b))))


# Alias for backwards compatibility
distance = euclidean


def landmark_to_pixel(landmark, width, height):
    """Converts a normalized MediaPipe landmark (0..1) to integer pixel coordinates."""
    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


def calculate_ear(eye_points):
    """
    Computes Eye Aspect Ratio (EAR) from six ordered eye coordinates:
    p1 (outer corner), p2 (top-outer), p3 (top-inner),
    p4 (inner corner), p5 (bottom-inner), p6 (bottom-outer).

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal <= 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def calculate_mar(mouth_points):
    """
    Computes Mouth Aspect Ratio (MAR) to monitor yawning behavior.
    Expects ordered coordinates:
    p_left, p_right, p_top, p_bottom (or extended multi-point mouth set).
    """
    if len(mouth_points) < 4:
        return 0.0

    p_left, p_right, p_top, p_bottom = mouth_points[:4]

    vertical = euclidean(p_top, p_bottom)
    horizontal = euclidean(p_left, p_right)

    if horizontal <= 1e-6:
        return 0.0

    return float(vertical / horizontal)


# Generic 3D model landmarks for solvePnP head pose estimation
MODEL_3D_POINTS = [
    (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
    (0.0, -330.0, -65.0),     # Chin (Landmark 152)
    (-225.0, 170.0, -135.0),  # Left eye outer corner (Landmark 33)
    (225.0, 170.0, -135.0),   # Right eye outer corner (Landmark 263)
    (-150.0, -150.0, -125.0), # Left mouth corner (Landmark 61)
    (150.0, -150.0, -125.0)   # Right mouth corner (Landmark 291)
]


def estimate_head_pose(landmarks, width, height):
    """
    Estimates 3D Head Pose (pitch, yaw, roll) in degrees using cv2.solvePnP.
    - Pitch: > 0 means looking down / nodding; < 0 means looking up
    - Yaw: > 0 means turned right; < 0 means turned left
    - Roll: sideways head tilt
    """
    if landmarks is None or cv2 is None or np is None:
        return 0.0, 0.0, 0.0

    key_ids = [1, 152, 33, 263, 61, 291]
    image_points = []
    for idx in key_ids:
        lm = landmarks[idx]
        image_points.append([lm.x * width, lm.y * height])

    image_points = np.array(image_points, dtype=np.float64)
    model_points = np.array(MODEL_3D_POINTS, dtype=np.float64)

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0

    return (
        float(np.degrees(pitch)),
        float(np.degrees(yaw)),
        float(np.degrees(roll))
    )
