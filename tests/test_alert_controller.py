import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from alert_controller import AlertController, AlertState


class TestAlertController(unittest.TestCase):
    def test_critical_state(self):
        controller = AlertController(
            watch_threshold=0.25,
            warning_threshold=0.40,
            critical_threshold=0.60
        )
        state = controller.get_state(0.70)
        self.assertEqual(state, AlertState.CRITICAL)

    def test_normal_alert(self):
        controller = AlertController(
            watch_threshold=0.25,
            warning_threshold=0.40,
            critical_threshold=0.60
        )
        state = controller.update(0.10, current_time=100.0)
        self.assertEqual(state, AlertState.NORMAL)

    def test_watch_alert_sustained(self):
        controller = AlertController(
            watch_threshold=0.25,
            warning_threshold=0.40,
            critical_threshold=0.60,
            watch_duration=3.0
        )

        t = 100.0
        # Not triggered immediately
        state = controller.update(0.30, current_time=t)
        self.assertEqual(state, AlertState.NORMAL)

        # After 2 seconds: still under 3 seconds
        state = controller.update(0.30, current_time=t + 2.0)
        self.assertEqual(state, AlertState.NORMAL)

        # After 3.5 seconds: sustained -> WATCH triggers
        state = controller.update(0.30, current_time=t + 3.5)
        self.assertEqual(state, AlertState.WATCH)

    def test_warning_alert_sustained(self):
        controller = AlertController(
            watch_threshold=0.25,
            warning_threshold=0.40,
            critical_threshold=0.60,
            warning_duration=5.0
        )

        t = 100.0
        state = controller.update(0.50, current_time=t)
        self.assertEqual(state, AlertState.NORMAL)

        # After 5.1s: triggers WARNING
        state = controller.update(0.50, current_time=t + 5.1)
        self.assertEqual(state, AlertState.WARNING)

    def test_critical_alert_sustained(self):
        controller = AlertController(
            watch_threshold=0.25,
            warning_threshold=0.40,
            critical_threshold=0.60,
            critical_duration=10.0
        )

        t = 100.0
        state = controller.update(0.70, current_time=t)
        self.assertEqual(state, AlertState.NORMAL)

        # After 10.5s: triggers CRITICAL
        state = controller.update(0.70, current_time=t + 10.5)
        self.assertEqual(state, AlertState.CRITICAL)

    def test_microsleep_emergency_fasttrack(self):
        controller = AlertController(
            watch_threshold=0.25,
            warning_threshold=0.40,
            critical_threshold=0.60,
            critical_duration=10.0
        )

        # Eyes closed continuously for 2.6 seconds -> immediate CRITICAL override
        state = controller.update(0.15, continuous_closed_seconds=2.6, current_time=100.0)
        self.assertEqual(state, AlertState.CRITICAL)

    def test_alert_cooldown(self):
        controller = AlertController(
            watch_threshold=0.25,
            warning_threshold=0.40,
            critical_threshold=0.60,
            watch_duration=0.0,
            cooldown_seconds=10.0
        )

        t = 100.0
        state1 = controller.update(0.30, current_time=t)
        self.assertEqual(state1, AlertState.WATCH)
        self.assertEqual(controller.last_alert_time, 100.0)

        # Within cooldown period (5s later)
        state2 = controller.update(0.30, current_time=t + 5.0)
        self.assertEqual(state2, AlertState.WATCH)
        self.assertEqual(controller.last_alert_time, 100.0)


def test_critical_state():
    """Pytest top-level test runner matching Section 17 example."""
    controller = AlertController(
        watch_threshold=0.25,
        warning_threshold=0.40,
        critical_threshold=0.60
    )
    state = controller.get_state(0.70)
    assert state == AlertState.CRITICAL


if __name__ == "__main__":
    unittest.main()
