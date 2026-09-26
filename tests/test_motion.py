import numpy as np
import pytest

from edge_surveillance.config import MotionConfig
from edge_surveillance.core.motion_detector import MotionDetector


def _black(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _detector(**kwargs):
    base = {"min_change_percent": 1.0, "threshold": 25, "cooldown_s": 0.0}
    base.update(kwargs)
    return MotionDetector(MotionConfig(**base))


def test_first_frame_arms_reference():
    m = _detector()
    r = m.detect(_black())
    assert r.is_motion is False
    assert r.change_percent == 0.0


def test_identical_frames_no_motion():
    m = _detector()
    frame = _black()
    m.detect(frame)
    r = m.detect(frame)
    assert r.is_motion is False
    assert r.change_percent == pytest.approx(0.0)


def test_large_square_appearing_triggers():
    m = _detector(min_area_percent=1.0)
    m.detect(_black())
    m.detect(_black())
    moved = _black()
    moved[100:300, 200:400] = 255
    r = m.detect(moved)
    assert r.is_motion is True
    assert r.change_percent > 1.0


def test_shifted_square_triggers():
    first = _black()
    first[100:300, 200:400] = 255
    m = _detector(min_area_percent=1.0)
    m.detect(first)
    moved = _black()
    moved[100:300, 205:405] = 255
    r = m.detect(moved)
    assert r.is_motion is True
    assert r.change_percent > 1.0


def test_tiny_speck_ignored_by_blob_gate():
    m = _detector(min_change_percent=0.01, min_area_percent=1.0)
    m.detect(_black())
    m.detect(_black())
    speck = _black()
    speck[10:13, 10:13] = 255
    r = m.detect(speck)
    assert r.is_motion is False


def test_high_percent_gate_suppresses():
    m = _detector(min_change_percent=50.0)
    m.detect(_black())
    m.detect(_black())
    moved = _black()
    moved[100:300, 200:400] = 255
    r = m.detect(moved)
    assert r.is_motion is False
    assert r.change_percent > 0


def test_cooldown_suppresses_repeat():
    m = _detector(min_area_percent=1.0, cooldown_s=10.0)
    m.detect(_black(), now=0.0)
    m.detect(_black(), now=0.0)
    moved = _black()
    moved[100:300, 200:400] = 255
    assert m.detect(moved, now=100.0).is_motion is True
    repeat = m.detect(moved, now=101.0)
    assert repeat.is_motion is False
    assert repeat.change_percent >= 0


def test_reset_rearms():
    m = _detector()
    m.detect(_black())
    moved = _black()
    moved[100:300, 200:400] = 255
    assert m.detect(moved).is_motion is True
    m.reset()
    r = m.detect(moved)
    assert r.is_motion is False
    assert r.change_percent == 0.0


def test_empty_frame_raises():
    m = _detector()
    with pytest.raises(ValueError):
        m.detect(np.zeros((0, 0, 3), dtype=np.uint8))
