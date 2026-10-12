import unittest
import sys
from pathlib import Path

# Ensure app package and repository root are available in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
app_dir = repo_root / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from app.fatigue_monitor import FatigueMonitor


class TestFatigueMonitor(unittest.TestCase):
    def test_short_yawn_does_not_trigger(self):
        monitor = FatigueMonitor()
        score, _ = monitor.update(
            mar=0.70, pitch=0.0, now=0.0
        )
        self.assertEqual(score, 0.0)
        score, _ = monitor.update(
            mar=0.30, pitch=0.0, now=0.5
        )
        self.assertEqual(score, 0.0)

    def test_sustained_yawn_is_detected(self):
        monitor = FatigueMonitor()
        monitor.update(mar=0.70, pitch=0.0, now=0.0)
        score, reasons = monitor.update(
            mar=0.70, pitch=0.0, now=1.6
        )
        self.assertAlmostEqual(score, 0.30)
        self.assertIn("sustained_yawn", reasons)

    def test_sustained_head_droop_is_detected(self):
        monitor = FatigueMonitor()
        monitor.update(mar=0.0, pitch=20.0, now=0.0)
        score, reasons = monitor.update(
            mar=0.0, pitch=20.0, now=2.1
        )
        self.assertAlmostEqual(score, 0.42)
        self.assertIn("sustained_head_droop", reasons)

    def test_face_unavailable_resets_state(self):
        monitor = FatigueMonitor()
        monitor.update(mar=0.70, pitch=20.0, now=0.0)
        score, reasons = monitor.update(
            mar=0.70, pitch=20.0, face_present=False, now=1.6
        )
        self.assertEqual(score, 0.0)
        self.assertIn("face_unavailable", reasons)
        self.assertIsNone(monitor._yawn_start)
        self.assertIsNone(monitor._droop_start)


if __name__ == "__main__":
    unittest.main()
