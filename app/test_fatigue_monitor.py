
# app/test_fatigue_monitor.py
from fatigue_monitor import FatigueMonitor


def test_short_mouth_opening_is_not_yawn():
    monitor = FatigueMonitor(
        mar_threshold=0.65,
        yawn_seconds=1.2,
    )

    result = monitor.update(0.8, 0.0, True, now=10.0)
    assert not result.yawning

    result = monitor.update(0.2, 0.0, True, now=10.5)
    assert not result.yawning


def test_sustained_mouth_opening_is_yawn():
    monitor = FatigueMonitor(
        mar_threshold=0.65,
        yawn_seconds=1.2,
    )

    monitor.update(0.8, 0.0, True, now=10.0)
    result = monitor.update(0.8, 0.0, True, now=11.3)

    assert result.yawning


def test_sustained_head_down_is_detected():
    monitor = FatigueMonitor(
        head_down_degrees=18.0,
        head_down_seconds=1.5,
    )

    monitor.update(0.2, 25.0, True, now=10.0)
    result = monitor.update(0.2, 25.0, True, now=11.6)

    assert result.head_drooping


def test_missing_face_is_reported():
    monitor = FatigueMonitor()

    result = monitor.update(0.0, 0.0, False, now=10.0)

    assert result.no_face
    assert not result.yawning
    assert not result.head_drooping
