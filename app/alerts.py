import os
import wave
import struct
import math
from pathlib import Path

# Safe import of pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def ensure_alert_sound(sound_path):
    """
    Synthesizes a 1.0 second, 880Hz / 440Hz dual-tone warning WAV file
    if the sound file does not exist, ensuring zero-dependency startup.
    """
    path = Path(sound_path)
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    duration = 1.0  # seconds
    n_samples = int(sample_rate * duration)

    with wave.open(str(path), 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            # Pulsing alarm: 880 Hz alternating with 587 Hz
            freq = 880.0 if (int(t * 8) % 2 == 0) else 587.33
            val = int(32767.0 * 0.4 * math.sin(2.0 * math.pi * freq * t))
            wav.writeframes(struct.pack('<h', val))

    print(f"[AlertActuator] Synthesized fallback alert audio at: {path}")
    return str(path)


class AlertActuator:
    """
    Controls multi-modal physical and software actuation channels:
    - Watch: Visual indicator
    - Warning: Audible alert tone
    - Critical: Audible alarm + visual flashing + simulated haptic vibration
    """

    def __init__(self, sound_path=None):
        if sound_path is None:
            project_root = Path(__file__).resolve().parent.parent
            sound_path = project_root / "data" / "warning.wav"
        elif not Path(sound_path).is_absolute() and not Path(sound_path).exists():
            # Check relative to project root
            project_root = Path(__file__).resolve().parent.parent
            alt_path = project_root / sound_path
            if alt_path.exists() or not Path(sound_path).parent.exists():
                sound_path = alt_path

        self.sound_path = ensure_alert_sound(sound_path)
        self.mixer_initialized = False
        self.warning_sound = None

        if PYGAME_AVAILABLE:
            try:
                # Initialize mixer with low latency
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self.warning_sound = pygame.mixer.Sound(self.sound_path)
                self.mixer_initialized = True
            except Exception as e:
                print(f"[AlertActuator] Warning: Audio mixer unavailable ({e}). Running in silent fallback mode.")
        else:
            print("[AlertActuator] Warning: pygame not available. Running in console mode.")

    def trigger(self, alert_state):
        """Triggers appropriate physical & acoustic actuators based on alert level."""
        state_str = str(alert_state).lower()
        if hasattr(alert_state, "value"):
            state_str = alert_state.value.lower()

        if state_str == "watch":
            print("[ACTUATOR: WATCH] -> [YELLOW LED ON] Visual warning indicator active.")

        elif state_str == "warning":
            print("[ACTUATOR: WARNING] -> [ORANGE LED ON] [AUDIO TONE PLAYING] Driver fatigue alert!")
            self._play_sound()

        elif state_str == "critical":
            print("[ACTUATOR: CRITICAL] -> [RED LED FLASHING] [AUDIO ALARM MAXIMUM] [HAPTIC MOTOR PULSING] Critical microsleep/drowsiness!")
            self._play_sound()

    def _play_sound(self):
        if self.mixer_initialized and self.warning_sound is not None:
            try:
                self.warning_sound.play()
            except Exception as e:
                print(f"[AlertActuator] Playback error: {e}")

    def close(self):
        if self.mixer_initialized:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
