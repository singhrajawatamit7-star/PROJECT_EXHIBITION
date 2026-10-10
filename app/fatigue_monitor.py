
# app/fatigue_monitor.py
import time
from dataclasses import dataclass


@dataclass
class FatigueSignals:
    yawning: bool = False
    head_drooping: bool = False
    no_face: bool = False


class FatigueMonitor:
    """
    Tracks sustained fatigue-related signals.

    Important:
    - Calibrate MAR and pitch thresholds for your camera.
    - This is a heuristic, not a medical or safety-certified detector.
    """

    def __init__(
        self,
        mar_threshold=0.65,
        yawn_seconds=1.2,
        head_down_degrees=20.0,
        head_down_seconds=1.5,
        recovery_seconds=0.5,
    ):
        self.mar_threshold = mar_threshold
        self.yawn_seconds = yawn_seconds
        self.head_down_degrees = head_down_degrees
        self.head_down_seconds = head_down_seconds
        self.recovery_seconds = recovery_seconds

        self._yawn_since = None
        self._head_down_since = None
        self._yawn_below_since = None
        self._head_up_since = None

    def update(self, mar, pitch, face_detected, now=None):
        now = time.monotonic() if now is None else now

        if not face_detected:
            # Do not interpret missing landmarks as an alert-free driver.
            self._yawn_since = None
            self._head_down_since = None
            self._yawn_below_since = None
            self._head_up_since = None

            return FatigueSignals(no_face=True)

        # Yawning: mouth opening sustained over time.
        if mar >= self.mar_threshold:
            if self._yawn_since is None:
                self._yawn_since = now
            self._yawn_below_since = None
        else:
            if self._yawn_below_since is None:
                self._yawn_below_since = now
            elif now - self._yawn_below_since >= self.recovery_seconds:
                self._yawn_since = None

        yawning = (
            self._yawn_since is not None
            and now - self._yawn_since >= self.yawn_seconds
        )

        # Head-down detection. Verify pitch sign and angle calibration
        # with your own camera before using this rule.
        if pitch >= self.head_down_degrees:
            if self._head_down_since is None:
                self._head_down_since = now
            self._head_up_since = None
        else:
            if self._head_up_since is None:
                self._head_up_since = now
            elif now - self._head_up_since >= self.recovery_seconds:
                self._head_down_since = None

        head_drooping = (
            self._head_down_since is not None
            and now - self._head_down_since >= self.head_down_seconds
        )

        return FatigueSignals(
            yawning=yawning,
            head_drooping=head_drooping,
            no_face=False,
        )
