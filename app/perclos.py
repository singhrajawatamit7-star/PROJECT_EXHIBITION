from collections import deque
import time


class PerclosTracker:
    """
    Tracks PERCLOS (Percentage of Eye Closure over a rolling time window).
    Also records short-term blink durations and eye closure statistics.
    """

    def __init__(self, window_seconds=30):
        self.window_seconds = window_seconds
        self.samples = deque()  # (timestamp, eyes_closed)
        self.closed_start_time = None
        self.last_blink_duration = 0.0
        self.total_blinks = 0

    def update(self, eyes_closed: bool, timestamp=None):
        """
        Appends latest sample, purges expired records, and returns current PERCLOS (0.0 - 1.0).
        """
        current_time = time.time() if timestamp is None else timestamp

        # Track individual blink duration
        if eyes_closed:
            if self.closed_start_time is None:
                self.closed_start_time = current_time
        else:
            if self.closed_start_time is not None:
                self.last_blink_duration = current_time - self.closed_start_time
                self.closed_start_time = None
                self.total_blinks += 1

        self.samples.append((current_time, bool(eyes_closed)))

        # Evict samples older than window_seconds
        while self.samples and (current_time - self.samples[0][0]) > self.window_seconds:
            self.samples.popleft()

        return self.calculate()

    def calculate(self):
        """Calculates ratio of eye-closed samples in rolling window."""
        if not self.samples:
            return 0.0

        closed_count = sum(1 for _, closed in self.samples if closed)
        return float(closed_count / len(self.samples))

    def get_sample_count(self):
        return len(self.samples)

    def get_current_closed_duration(self, current_time=None):
        """Returns how long eyes have been continuously closed currently (in seconds)."""
        if self.closed_start_time is None:
            return 0.0
        now = time.time() if current_time is None else current_time
        return max(0.0, now - self.closed_start_time)

    def reset(self):
        self.samples.clear()
        self.closed_start_time = None
        self.last_blink_duration = 0.0
        self.total_blinks = 0
