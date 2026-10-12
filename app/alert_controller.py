from enum import Enum
import time


class AlertState(Enum):
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertController:
    """
    Multi-stage staged alert state machine with hysteresis, sustained duration timers,
    and cooldown protection to prevent false alarms and alarm fatigue.
    """

    def __init__(
        self,
        watch_threshold=0.25,
        warning_threshold=0.40,
        critical_threshold=0.60,
        watch_duration=3.0,
        warning_duration=5.0,
        critical_duration=10.0,
        cooldown_seconds=10.0
    ):
        self.watch_threshold = watch_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.watch_duration = watch_duration
        self.warning_duration = warning_duration
        self.critical_duration = critical_duration
        self.cooldown_seconds = cooldown_seconds

        self.state = AlertState.NORMAL
        self.last_alert_time = 0.0

        # Timers tracking continuous breach of thresholds
        self.threshold_entry_times = {
            AlertState.WATCH: None,
            AlertState.WARNING: None,
            AlertState.CRITICAL: None
        }

    def get_state(self, perclos):
        """
        Direct threshold evaluation mapping matching Section 12 rule definition:
        < 0.25 -> NORMAL, 0.25..0.39 -> WATCH, 0.40..0.59 -> WARNING, >= 0.60 -> CRITICAL.
        """
        if perclos >= self.critical_threshold:
            return AlertState.CRITICAL
        if perclos >= self.warning_threshold:
            return AlertState.WARNING
        if perclos >= self.watch_threshold:
            return AlertState.WATCH
        return AlertState.NORMAL

    def update(
        self,
        perclos_score,
        continuous_closed_seconds=0.0,
        current_time=None,
        external_risk_score=0.0,
    ):
        """
        Evaluates current score and sustained duration, returning current active AlertState.
        """
        now = time.time() if current_time is None else current_time

        # PERCLOS remains a separate measured metric.
        # External risk is a supplementary heuristic, not PERCLOS.
        decision_score = max(
            float(perclos_score),
            float(external_risk_score),
        )

        # Fast-track microsleep emergency: eyes continuously closed >= 2.5 seconds
        if continuous_closed_seconds >= 2.5:
            proposed = AlertState.CRITICAL
        elif decision_score >= self.critical_threshold:
            proposed = AlertState.CRITICAL
        elif decision_score >= self.warning_threshold:
            proposed = AlertState.WARNING
        elif decision_score >= self.watch_threshold:
            proposed = AlertState.WATCH
        else:
            proposed = AlertState.NORMAL

        # Update entry times for sustained requirement check
        for level in [AlertState.WATCH, AlertState.WARNING, AlertState.CRITICAL]:
            if proposed == level or (level == AlertState.WATCH and proposed in [AlertState.WARNING, AlertState.CRITICAL]) \
                                 or (level == AlertState.WARNING and proposed == AlertState.CRITICAL):
                if self.threshold_entry_times[level] is None:
                    self.threshold_entry_times[level] = now
            else:
                self.threshold_entry_times[level] = None

        # Determine sustained candidate state
        candidate_state = AlertState.NORMAL

        if self.threshold_entry_times[AlertState.CRITICAL] is not None:
            sustained = now - self.threshold_entry_times[AlertState.CRITICAL]
            if sustained >= self.critical_duration or continuous_closed_seconds >= 2.5:
                candidate_state = AlertState.CRITICAL

        if candidate_state == AlertState.NORMAL and self.threshold_entry_times[AlertState.WARNING] is not None:
            sustained = now - self.threshold_entry_times[AlertState.WARNING]
            if sustained >= self.warning_duration:
                candidate_state = AlertState.WARNING

        if candidate_state == AlertState.NORMAL and self.threshold_entry_times[AlertState.WATCH] is not None:
            sustained = now - self.threshold_entry_times[AlertState.WATCH]
            if sustained >= self.watch_duration:
                candidate_state = AlertState.WATCH

        # If candidate drops back to normal, reset state immediately (hysteresis drop)
        if candidate_state == AlertState.NORMAL:
            self.state = AlertState.NORMAL
            return self.state

        # Cooldown check for re-triggering repeated acoustic alarms
        if (
            candidate_state != AlertState.NORMAL
            and (now - self.last_alert_time) < self.cooldown_seconds
            and candidate_state == self.state
        ):
            return self.state

        # State transition or re-alert trigger after cooldown
        if candidate_state != self.state or (now - self.last_alert_time) >= self.cooldown_seconds:
            self.state = candidate_state
            if self.state != AlertState.NORMAL:
                self.last_alert_time = now

        return self.state

    def reset(self):
        self.state = AlertState.NORMAL
        self.last_alert_time = 0.0
        self.threshold_entry_times = {k: None for k in self.threshold_entry_times}
