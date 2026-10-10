import unittest
import numpy as np
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from features import euclidean, calculate_ear, calculate_mar, landmark_to_pixel


class TestFeatures(unittest.TestCase):
    def test_euclidean(self):
        p1 = (0, 0)
        p2 = (3, 4)
        self.assertEqual(euclidean(p1, p2), 5.0)

        p3 = (1, 2, 3)
        p4 = (1, 2, 7)
        self.assertEqual(euclidean(p3, p4), 4.0)

    def test_calculate_ear_open_eye(self):
        # Synthetic coordinates representing an open eye
        # p1: outer corner (0, 0), p4: inner corner (10, 0) -> width = 10
        # p2: (3, 3), p6: (3, -3) -> vertical_1 = 6
        # p3: (7, 3), p5: (7, -3) -> vertical_2 = 6
        # Expected EAR = (6 + 6) / (2 * 10) = 12 / 20 = 0.60
        open_eye = [
            (0, 0),   # p1
            (3, 3),   # p2
            (7, 3),   # p3
            (10, 0),  # p4
            (7, -3),  # p5
            (3, -3)   # p6
        ]
        ear = calculate_ear(open_eye)
        self.assertAlmostEqual(ear, 0.60, places=4)

    def test_calculate_ear_closed_eye(self):
        # Eyelids touching / nearly closed
        # vertical distances = 1.0 each, width = 10
        # Expected EAR = (1 + 1) / 20 = 0.10
        closed_eye = [
            (0, 0),
            (3, 0.5),
            (7, 0.5),
            (10, 0),
            (7, -0.5),
            (3, -0.5)
        ]
        ear = calculate_ear(closed_eye)
        self.assertAlmostEqual(ear, 0.10, places=4)
        self.assertLess(ear, 0.21)

    def test_calculate_ear_zero_horizontal(self):
        degenerate = [(0, 0)] * 6
        ear = calculate_ear(degenerate)
        self.assertEqual(ear, 0.0)

    def test_calculate_mar(self):
        normal_mouth = [(0, 0), (20, 0), (10, 2), (10, -2)]
        mar_normal = calculate_mar(normal_mouth)
        self.assertAlmostEqual(mar_normal, 0.20, places=4)

        yawn_mouth = [(0, 0), (20, 0), (10, 9), (10, -9)]
        mar_yawn = calculate_mar(yawn_mouth)
        self.assertAlmostEqual(mar_yawn, 0.90, places=4)
        self.assertGreater(mar_yawn, 0.65)


if __name__ == "__main__":
    unittest.main()
