import numpy as np
import pytest

from edge_surveillance.server.render import draw_overlay
from edge_surveillance.state import FaceReport, FrameState


def _frame(w=160, h=120):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _report(name="ann", score=0.9, x=10, y=20, w=40, h=50):
    return FaceReport(name=name, score=score, x=x, y=y, w=w, h=h)


def test_frame_state_frame_is_read_only():
    state = FrameState(captured_at=1.0, frame=_frame())
    with pytest.raises(ValueError):
        state.frame[0, 0] = 255


def test_draw_overlay_does_not_touch_the_source_frame():
    frame = _frame()
    state = FrameState(captured_at=1.0, frame=frame, faces=(_report(),))
    before = frame.copy()
    draw_overlay(state.frame, state.faces)
    assert np.array_equal(frame, before)


def test_draw_overlay_draws_on_a_copy():
    frame = _frame()
    state = FrameState(captured_at=1.0, frame=frame, faces=(_report(),))
    out = draw_overlay(state.frame, state.faces)
    assert out is not frame
    assert out.flags.writeable
    assert out.any()  # something was drawn


def test_draw_overlay_with_no_faces_returns_untouched_copy():
    frame = np.full((40, 40, 3), 7, dtype=np.uint8)
    out = draw_overlay(frame, ())
    assert np.array_equal(out, frame)
    assert out is not frame


def test_overlay_pixels_differ_between_known_and_unknown():
    known = draw_overlay(_frame(), (_report(name="ann"),))
    unknown = draw_overlay(_frame(), (_report(name="Unknown"),))
    assert not np.array_equal(known, unknown)


def test_draw_overlay_handles_box_at_top_edge():
    out = draw_overlay(_frame(), (_report(y=0),))
    assert out.any()