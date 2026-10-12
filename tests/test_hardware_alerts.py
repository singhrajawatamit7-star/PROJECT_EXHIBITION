import unittest
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from hardware_alerts import HardwareAlerts


class TestHardwareAlerts(unittest.TestCase):
    def test_graceful_fallback(self):
        alerts = HardwareAlerts(led_pin=17, buzzer_pin=27)
        # Should execute cleanly regardless of whether real GPIO is available
        alerts.set_state("normal")
        alerts.set_state("watch")
        alerts.set_state("warning")
        alerts.set_state("critical")
        alerts.set_state("no_face")
        alerts.trigger("watch")
        alerts.close()

    def test_disabled_on_non_gpio_platform(self):
        alerts = HardwareAlerts(led_pin=17, buzzer_pin=27)
        # If running on non-Raspberry Pi, alerts.enabled is False
        if not alerts.enabled:
            self.assertIsNone(alerts.led)
            self.assertIsNone(alerts.buzzer)


if __name__ == "__main__":
    unittest.main()
