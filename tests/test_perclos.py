import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from perclos import PerclosTracker


class TestPerclosTracker(unittest.TestCase):
    def test_empty_perclos(self):
        tracker = PerclosTracker(window_seconds=30)
        self.assertEqual(tracker.calculate(), 0.0)
        self.assertEqual(tracker.get_sample_count(), 0)

    def test_all_open(self):
        tracker = PerclosTracker(window_seconds=30)
        base_time = 1000.0
        for i in range(10):
            tracker.update(eyes_closed=False, timestamp=base_time + i)

        self.assertEqual(tracker.calculate(), 0.0)
        self.assertEqual(tracker.get_sample_count(), 10)

    def test_all_closed(self):
        tracker = PerclosTracker(window_seconds=30)
        base_time = 1000.0
        for i in range(10):
            tracker.update(eyes_closed=True, timestamp=base_time + i)

        self.assertEqual(tracker.calculate(), 1.0)

    def test_fifty_percent_closed(self):
        tracker = PerclosTracker(window_seconds=30)
        base_time = 1000.0
        for i in range(10):
            tracker.update(eyes_closed=(i % 2 == 0), timestamp=base_time + i)

        self.assertEqual(tracker.calculate(), 0.5)

    def test_window_eviction(self):
        tracker = PerclosTracker(window_seconds=10)
        base_time = 100.0

        for _ in range(5):
            tracker.update(eyes_closed=True, timestamp=base_time)

        self.assertEqual(tracker.calculate(), 1.0)

        # Advance time by 15s with 5 open samples
        for _ in range(5):
            tracker.update(eyes_closed=False, timestamp=base_time + 15.0)

        self.assertEqual(tracker.get_sample_count(), 5)
        self.assertEqual(tracker.calculate(), 0.0)

    def test_continuous_closed_duration(self):
        tracker = PerclosTracker(window_seconds=30)
        t = 100.0

        tracker.update(eyes_closed=True, timestamp=t)
        self.assertEqual(tracker.get_current_closed_duration(current_time=t + 2.5), 2.5)

        tracker.update(eyes_closed=False, timestamp=t + 2.6)
        self.assertEqual(tracker.get_current_closed_duration(current_time=t + 3.0), 0.0)
        self.assertEqual(tracker.total_blinks, 1)
        self.assertAlmostEqual(tracker.last_blink_duration, 2.6, places=4)


if __name__ == "__main__":
    unittest.main()
