
# app/hardware_alerts.py
from gpiozero import LED, Buzzer


class HardwareAlerts:
    """
    Raspberry Pi GPIO output controller.

    Example wiring:
      GPIO17 -> LED resistor -> LED -> GND
      GPIO27 -> compatible active buzzer module -> GND

    Check your specific hardware ratings and wiring.
    Do not drive a high-current motor directly from a GPIO pin.
    """

    def __init__(self, led_pin=17, buzzer_pin=27):
        self.led = LED(led_pin)
        self.buzzer = Buzzer(buzzer_pin)

    def trigger(self, level):
        level = str(getattr(level, "value", level)).lower()

        if level == "normal":
            self.led.off()
            self.buzzer.off()

        elif level == "watch":
            self.led.on()
            self.buzzer.off()

        elif level in ("warning", "critical"):
            self.led.on()
            self.buzzer.on()

    def close(self):
        self.buzzer.off()
        self.led.off()
        self.led.close()
        self.buzzer.close()
