import unittest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_drowsiness.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_logging(self):
        db = Database(database_path=str(self.db_path), async_mode=False)

        # Log prediction
        db.log_prediction(
            perclos=0.45,
            ear=0.18,
            alert_state="warning",
            mar=0.35,
            head_pitch=12.0,
            head_yaw=4.0,
            model_version="rule_v1"
        )

        preds = db.get_recent_predictions(limit=5)
        self.assertEqual(len(preds), 1)
        self.assertAlmostEqual(preds[0][1], 0.45, places=4)
        self.assertAlmostEqual(preds[0][2], 0.18, places=4)
        self.assertEqual(preds[0][3], "warning")

        # Log alert
        db.log_alert(level="warning", channel="audio")
        alerts = db.get_recent_alerts(limit=5)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0][1], "warning")
        self.assertEqual(alerts[0][2], "audio")

        db.close()

    def test_database_async_flush(self):
        db = Database(database_path=str(self.db_path), async_mode=True)

        for i in range(15):
            db.log_prediction(perclos=0.1 * i, ear=0.2, alert_state="normal")

        # Graceful close flushes queue
        db.close()

        db_sync = Database(database_path=str(self.db_path), async_mode=False)
        preds = db_sync.get_recent_predictions(limit=50)
        self.assertEqual(len(preds), 15)
        db_sync.close()


if __name__ == "__main__":
    unittest.main()
