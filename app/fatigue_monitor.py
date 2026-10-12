"""
Fatigue Monitor Module
Tracks sustained yawn and head droop indicators to calculate a supplementary heuristic risk score.
"""

import time


class FatigueMonitor:
    """
    Combines supplementary fatigue indicators such as sustained yawning and head drooping.
    Provides a heuristic risk score (0 to 1), NOT a probability. It is separate from
    the actual PERCLOS measurement.
    """

    def __init__(
        self,
        yawn_mar_threshold=0.65,
        yawn_duration=1.5,
        head_pitch_threshold=15.0,
        head_droop_duration=2.0,
        mar_threshold=None,
        yawn_seconds=None,
        head_down_degrees=None,
        head_down_seconds=None,
        **kwargs,
    ):
        self.yawn_mar_threshold = mar_threshold if mar_threshold is not None else yawn_mar_threshold
        self.yawn_duration = yawn_seconds if yawn_seconds is not None else yawn_duration
        self.head_pitch_threshold = head_down_degrees if head_down_degrees is not None else head_pitch_threshold
        self.head_droop_duration = head_down_seconds if head_down_seconds is not None else head_droop_duration
        self._yawn_start = None
        self._droop_start = None

    def reset(self):
        """Clear temporal indicators when the face is unavailable."""
        self._yawn_start = None
        self._droop_start = None

    def update(self, mar, pitch, face_present=True, now=None, face_detected=None):
        """
        Return (risk_score, reasons).
        risk_score is a heuristic from 0 to 1, NOT a probability.
        It is separate from the actual PERCLOS measurement.
        """
        if face_detected is not None:
            face_present = face_detected

        now = time.monotonic() if now is None else now
        if not face_present:
            self.reset()
            return 0.0, ["face_unavailable"]

        # Track a sustained mouth opening.
        if mar >= self.yawn_mar_threshold:
            if self._yawn_start is None:
                self._yawn_start = now
        else:
            self._yawn_start = None

        # Track a sustained downward head angle.
        if pitch >= self.head_pitch_threshold:
            if self._droop_start is None:
                self._droop_start = now
        else:
            self._droop_start = None

        score = 0.0
        reasons = []

        if (
            self._yawn_start is not None
            and now - self._yawn_start >= self.yawn_duration
        ):
            score = max(score, 0.30)
            reasons.append("sustained_yawn")

        if (
            self._droop_start is not None
            and now - self._droop_start >= self.head_droop_duration
        ):
            score = max(score, 0.42)
            reasons.append("sustained_head_droop")

        return score, reasons
