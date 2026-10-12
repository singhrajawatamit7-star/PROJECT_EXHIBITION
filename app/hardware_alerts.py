"""
Optional Raspberry Pi GPIO alerts module.
Controls an active buzzer and LED indicators via GPIO pins when running on compatible hardware.
Falls back safely if GPIO/gpiozero is unavailable.
"""


class HardwareAlerts:
    def __init__(self, led_pin=17, buzzer_pin=27):
        try:
            from gpiozero import LED, Buzzer
            self.led = LED(led_pin)
            self.buzzer = Buzzer(buzzer_pin)
            self.enabled = True
        except Exception:
            self.led = None
            self.buzzer = None
            self.enabled = False

    def set_state(self, state):
        # state: normal, watch, warning, critical, no_face
        if not self.enabled:
            return
        state = str(getattr(state, "value", state)).lower()
        if state == "normal":
            self.led.off()
            self.buzzer.off()
        elif state == "watch":
            self.led.on()
            self.buzzer.off()
        elif state in ("warning", "critical"):
            self.led.on()
            self.buzzer.on()
        else:
            # Monitoring unavailable: avoid implying
            # that the driver has been verified as alert.
            self.led.off()
            self.buzzer.off()

    def trigger(self, level):
        """Alias for set_state for backward compatibility."""
        self.set_state(level)

    def close(self):
        if not self.enabled:
            return
        try:
            self.buzzer.off()
            self.led.off()
            self.buzzer.close()
            self.led.close()
        except Exception:
            pass
