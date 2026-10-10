import sqlite3
import threading
import queue
from datetime import datetime, timezone
from pathlib import Path


class Database:
    """
    Non-blocking asynchronous SQLite event and metrics recorder.
    Ensures zero I/O jitter on real-time computer vision inference loops.
    """

    def __init__(self, database_path=None, async_mode=True):
        if database_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.database_path = project_root / "data" / "logs" / "drowsiness.db"
        elif not Path(database_path).is_absolute() and not Path(database_path).exists():
            project_root = Path(__file__).resolve().parent.parent
            alt_path = project_root / database_path
            self.database_path = alt_path if (alt_path.exists() or not Path(database_path).parent.exists()) else Path(database_path)
        else:
            self.database_path = Path(database_path)

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.async_mode = async_mode

        # Initialize schema synchronously first
        self._init_schema()

        if self.async_mode:
            self._queue = queue.Queue(maxsize=10000)
            self._stop_event = threading.Event()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="DatabaseWorker"
            )
            self._worker_thread.start()

    def _get_connection(self):
        return sqlite3.connect(str(self.database_path), check_same_thread=False)

    def _init_schema(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            perclos REAL,
            ear REAL,
            mar REAL,
            head_pitch REAL,
            head_yaw REAL,
            alert_state TEXT,
            model_version TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            channel TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_start TEXT NOT NULL,
            session_end TEXT,
            driver_id TEXT,
            notes TEXT
        )
        """)

        conn.commit()
        conn.close()

    def _worker_loop(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        batch_counter = 0

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                if batch_counter > 0:
                    conn.commit()
                    batch_counter = 0
                continue

            try:
                task_type, params = task
                if task_type == "prediction":
                    cursor.execute("""
                    INSERT INTO predictions (timestamp, perclos, ear, mar, head_pitch, head_yaw, alert_state, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, params)
                elif task_type == "alert":
                    cursor.execute("""
                    INSERT INTO alerts (timestamp, level, channel)
                    VALUES (?, ?, ?)
                    """, params)
                batch_counter += 1

                # Commit every 20 records or on idle
                if batch_counter >= 20:
                    conn.commit()
                    batch_counter = 0
            except Exception as e:
                print(f"[DatabaseWorker] Insert error: {e}")
            finally:
                self._queue.task_done()

        if batch_counter > 0:
            conn.commit()
        conn.close()

    def log_prediction(self, perclos, ear, alert_state, mar=0.0, head_pitch=0.0, head_yaw=0.0, model_version="rule_v1"):
        """Logs a single inference prediction cycle asynchronously."""
        ts = datetime.now(timezone.utc).isoformat()
        params = (ts, float(perclos), float(ear), float(mar), float(head_pitch), float(head_yaw), str(alert_state), model_version)

        if self.async_mode:
            try:
                self._queue.put_nowait(("prediction", params))
            except queue.Full:
                pass  # Avoid blocking pipeline if queue is saturated
        else:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO predictions (timestamp, perclos, ear, mar, head_pitch, head_yaw, alert_state, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, params)
            conn.commit()
            conn.close()

    def log_alert(self, level, channel="audio"):
        """Logs an alert escalation event."""
        ts = datetime.now(timezone.utc).isoformat()
        params = (ts, str(level), str(channel))

        if self.async_mode:
            try:
                self._queue.put_nowait(("alert", params))
            except queue.Full:
                pass
        else:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO alerts (timestamp, level, channel)
            VALUES (?, ?, ?)
            """, params)
            conn.commit()
            conn.close()

    def get_recent_predictions(self, limit=10):
        """Fetches the latest prediction rows for inspection."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, perclos, ear, alert_state FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_recent_alerts(self, limit=10):
        """Fetches the latest triggered alerts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, level, channel FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def close(self):
        """Flushes queue and safely halts background worker thread."""
        if self.async_mode:
            self._stop_event.set()
            if self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
