import threading
import time
from pathlib import Path
from typing import NamedTuple
from unittest.mock import create_autospec

import cv2
import numpy as np
import pytest

from edge_surveillance.camera.stream_reader import StreamReader
from edge_surveillance.config import AppConfig, CameraConfig
from edge_surveillance.core.face_detector import FaceDetection, YuNetDetector
from edge_surveillance.core.face_recognizer import MobileFaceNetRecognizer
from edge_surveillance.core.motion_detector import MotionDetector
from edge_surveillance.pipeline import SurveillancePipeline, claim_alert
from edge_surveillance.state import SharedState
from edge_surveillance.store.events import EventStore
from edge_surveillance.store.gallery import GalleryStore


class RecordingState(SharedState):
    """SharedState that keeps every published frame, for assertions over time."""

    def __init__(self):
        super().__init__()
        self.history: list = []

    def update(self, state) -> None:
        super().update(state)
        self.history.append(state)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeReader:
    """Minimal StreamReader interface for tests: no I/O, no threads.

    frame= repeats one frame forever (live source); frames= drains a script
    then reports EOF (file source)."""

    def __init__(self, frame=None, frames=None, is_live=True):
        self._frame = frame
        self._frames = list(frames) if frames is not None else None
        self.is_live = is_live
        self.stopped: list = []
        self.start_calls = 0

    def start(self):
        self.start_calls += 1
        return self

    def read(self):
        if self._frames is not None:
            if self._frames:
                return True, self._frames.pop(0)
            return False, None
        return True, self._frame

    def stop(self):
        self.stopped.append(True)


def _detector(faces):
    """Real YuNetDetector interface, no weights: spec'd mock validates calls."""
    detector = create_autospec(YuNetDetector, instance=True)
    detector.detect.return_value = list(faces)
    return detector


def _detector_seq(detections):
    """Detector returning a scripted list per motion frame; exhausts after
    len(detections) motion frames (StopIteration past that)."""
    detector = create_autospec(YuNetDetector, instance=True)
    detector.detect.side_effect = [list(d) for d in detections]
    return detector


def _recognizer(vec):
    """Same embedding on every call; pair with _recognizer_seq for per-call embeddings."""
    recognizer = create_autospec(MobileFaceNetRecognizer, instance=True)
    recognizer.extract.return_value = np.asarray(vec, dtype=np.float32)
    return recognizer


def _recognizer_seq(vecs):
    """Recognizer returning a different embedding per call, for multi-face frames.

    Exhausts after len(vecs) calls (StopIteration past that): single _process
    only, never a run() loop."""

    recognizer = create_autospec(MobileFaceNetRecognizer, instance=True)
    recognizer.extract.side_effect = [np.asarray(v, dtype=np.float32) for v in vecs]
    return recognizer


def _vec(*vals):
    """Unit-length embedding, so cosine matching in the gallery behaves."""
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _face(x=10, y=20, score=0.9):
    return FaceDetection(x=x, y=y, w=60, h=60, landmarks=np.zeros((5, 2)), score=score)


def _frame(w=160, h=120):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _lit_frame(w=160, h=120):
    return np.full((h, w, 3), 255, dtype=np.uint8)


N_FRAMES = 12


def _make_clip(path, n=N_FRAMES, w=160, h=120, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), f"VideoWriter failed to open {path}; missing mp4v codec?"
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if i >= n // 2:  # one motion transition, then static
            frame[20:100, 20:140] = 255
        writer.write(frame)
    writer.release()


def _config(events_dir, **event_overrides):
    config = AppConfig()
    config.camera = CameraConfig(source="0")
    # Sensitive enough that black->white reads as motion but white->white is static.
    config.motion.min_change_percent = 0.5
    config.motion.threshold = 25
    config.motion.min_area_percent = 0.5
    config.motion.cooldown_s = 0.0
    config.events.dir = str(events_dir)
    config.events.known_cooldown_s = 0.0
    config.events.unknown_cooldown_s = 0.0
    for key, value in event_overrides.items():
        setattr(config.events, key, value)
    return config


class Parts(NamedTuple):
    pipeline: SurveillancePipeline
    detector: YuNetDetector
    events: EventStore
    state: RecordingState
    reader: StreamReader | FakeReader


def _assemble(
    tmp_path,
    faces,
    enrolled=(("ann", (1.0, 0.0)),),
    detector=None,
    recognizer=None,
    query=None,
    reader=None,
    state=None,
    clock=None,
    report_ttl_s=None,
    **event_overrides,
) -> Parts:
    """Shared assembly: config + gallery + events + pipeline, no camera/video-codec I/O."""
    config = _config(tmp_path / "events", **event_overrides)
    if report_ttl_s is not None:
        config.detector.report_ttl_s = report_ttl_s
    gallery = GalleryStore(tmp_path / "gallery.pkl")
    for person_name, person_vec in enrolled:
        gallery.add(person_name, [_vec(*person_vec)])
    default_vec = enrolled[0][1]
    events = EventStore(tmp_path / "events.db")
    if detector is None:
        detector = _detector(faces)
    if recognizer is None:
        recognizer = _recognizer(_vec(*(query or default_vec)))
    state = state or RecordingState()
    reader = reader or FakeReader(frame=_frame(), is_live=True)
    pipeline = SurveillancePipeline(
        config,
        reader=reader,
        motion=MotionDetector(config.motion),
        detector=detector,
        recognizer=recognizer,
        gallery=gallery,
        events=events,
        state=state,
        **({"clock": clock} if clock is not None else {}),
    )
    return Parts(pipeline, detector, events, state, reader)


def _build_direct(tmp_path, faces, **kwargs):
    """Unit-style builder: no cv2 clip, FakeReader + optional FakeClock."""
    return _assemble(tmp_path, faces, **kwargs)


def _build_clip(tmp_path, faces, **kwargs):
    """Integration builder: real mp4 clip + real StreamReader + real MotionDetector."""
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    kwargs.setdefault("reader", StreamReader(CameraConfig(source=str(clip))))
    return _assemble(tmp_path, faces, **kwargs)


@pytest.fixture
def parts(tmp_path):
    """Default clip setup: 1 enrolled face, zero alert cooldowns; teardown
    stops the reader and closes the event DB."""
    built = _build_clip(tmp_path, [_face()], known_cooldown_s=0.0, unknown_cooldown_s=0.0)
    yield built
    try:
        built.reader.stop()
    finally:
        built.events.close()


def _frames_with_faces(state):
    return [s for s in state.history if s.faces]


def test_claim_alert_respects_cooldown():
    last: dict[str, float] = {}
    assert claim_alert(last, "ann", 100.0, 30.0) is True
    assert claim_alert(last, "ann", 110.0, 30.0) is False
    assert claim_alert(last, "ann", 131.0, 30.0) is True
    assert claim_alert(last, "bob", 131.0, 30.0) is True
    # Zero cooldown never suppresses, even at the same timestamp.
    assert claim_alert(last, "ann", 131.0, 0.0) is True
    assert claim_alert(last, "ann", 131.0, 0.0) is True


@pytest.mark.parametrize(
    ("report_ttl_s", "advance_s", "has_faces", "expect_faces"),
    [
        (2.0, 0.5, True, True),  # pause shorter than TTL -> box held
        (2.0, 2.1, True, False),  # pause longer than TTL -> box expires
        (0.0, 0.01, True, False),  # TTL of zero disables the hold
        (2.0, 0.5, False, False),  # never seen -> stays empty
    ],
    ids=["survive_pause", "expire_after_ttl", "ttl_zero_disables_hold", "never_seen"],
)
def test_face_report_hold(tmp_path, report_ttl_s, advance_s, has_faces, expect_faces):
    """A person who stops moving keeps their box for report_ttl_s, so the
    live overlay does not blink out every time they stand still."""
    clock = FakeClock()
    state = RecordingState()
    p = _build_direct(
        tmp_path,
        [_face()] if has_faces else [],
        reader=FakeReader(frame=_frame()),
        state=state,
        clock=clock,
        report_ttl_s=report_ttl_s,
    )
    try:
        p.pipeline._process(_frame())  # arms the motion reference
        p.pipeline._process(_lit_frame())  # motion -> reports published (if any faces)
        if has_faces:
            assert state.current.faces

        clock.advance(advance_s)
        p.pipeline._process(_lit_frame())  # unchanged -> no motion
        assert state.current.motion_percent == pytest.approx(0.0)
        if expect_faces:
            assert state.current.faces, "box vanished before the TTL expired"
        else:
            assert state.current.faces == ()
    finally:
        p.events.close()


def test_single_motion_miss_keeps_box(tmp_path):
    """One motion frame with no detections is usually the detector (profile
    view, occlusion), not a departure: the box holds."""
    clock = FakeClock()
    p = _build_direct(
        tmp_path,
        [_face()],
        detector=_detector_seq([[_face()], []]),
        clock=clock,
        report_ttl_s=2.0,
    )
    try:
        p.pipeline._process(_frame())
        clock.advance(0.1)
        p.pipeline._process(_lit_frame())  # motion -> hit
        assert p.state.current.faces

        clock.advance(0.1)
        p.pipeline._process(_frame())  # motion (white->black) but detector miss
        assert p.state.current.faces, "box blinked on a single miss"
    finally:
        p.events.close()


def test_repeated_motion_miss_clears_box_before_ttl(tmp_path):
    """Sustained motion with no detections means departure: boxes clear
    after the miss grace even though report_ttl_s has not lapsed."""
    from edge_surveillance.pipeline import _MAX_MISS_FRAMES

    clock = FakeClock()
    misses = _MAX_MISS_FRAMES + 1
    p = _build_direct(
        tmp_path,
        [_face()],
        detector=_detector_seq([[_face()]] + [[]] * misses),
        clock=clock,
        report_ttl_s=2.0,
    )
    try:
        p.pipeline._process(_frame())
        clock.advance(0.1)
        p.pipeline._process(_lit_frame())  # motion -> hit
        assert p.state.current.faces

        lit = True
        for _ in range(misses):
            clock.advance(0.1)
            lit = not lit
            p.pipeline._process(_lit_frame() if lit else _frame())
        # Total elapsed (~0.7s) is well inside the 2s TTL: only the miss
        # grace cleared the box.
        assert p.state.current.faces == ()
    finally:
        p.events.close()


def test_clock_injection_drives_recorded_timestamps(tmp_path):
    clock = FakeClock()
    p = _build_direct(tmp_path, [_face()], reader=FakeReader(frame=_frame()), clock=clock)
    try:
        p.pipeline._process(_frame())  # arms motion reference, no event
        clock.advance(5.0)
        p.pipeline._process(_lit_frame())  # motion -> event stamped at advanced time
        rows = p.events.recent()
        assert rows, "expected one event after motion"
        assert rows[0]["ts"] == pytest.approx(clock.now)
    finally:
        p.events.close()


def test_pipeline_records_known_face_and_publishes_state(parts):
    parts.pipeline.run()

    seen = _frames_with_faces(parts.state)
    assert seen, "no frame reported a face"
    report = seen[0].faces[0]
    assert report.name == "ann"
    assert (report.x, report.y, report.w, report.h) == (10, 20, 60, 60)
    assert parts.state.current is not None

    rows = parts.events.recent()
    assert rows, "expected at least one event"
    assert rows[0]["name"] == "ann" and rows[0]["type"] == "known"
    assert Path(rows[0]["path"]).is_file()


def test_pipeline_skips_face_models_when_no_motion(parts):
    parts.pipeline.run()
    # Only the black->white transition is motion, so detection must run on
    # strictly fewer frames than the clip contains.
    assert 0 < parts.detector.detect.call_count < N_FRAMES
    assert len(parts.state.history) == N_FRAMES


def test_every_frame_is_published_even_without_motion(parts):
    parts.pipeline.run()
    assert len(parts.state.history) == N_FRAMES
    for s in parts.state.history:
        assert s.frame is not None and s.motion_percent >= 0.0


def test_pipeline_records_unknown_face(tmp_path):
    p = _build_clip(tmp_path, [_face()], query=(0.0, 1.0))
    try:
        p.pipeline.run()
        rows = p.events.recent()
        assert rows, "expected at least one unknown event"
        assert rows[0]["type"] == "unknown"
        assert rows[0]["name"] is None
        assert Path(rows[0]["path"]).is_file()
    finally:
        p.events.close()


def test_unknown_cooldown_suppresses_repeat_alerts(tmp_path):
    """Unknowns share one cooldown slot: a second motion event inside the
    window is suppressed, one past it records again."""
    clock = FakeClock()
    p = _build_direct(tmp_path, [_face()], query=(0.0, 1.0), clock=clock, unknown_cooldown_s=60.0)
    # Steps derive from the configured cooldown so changing its value keeps
    # the inside/outside-window timing intact.
    cooldown = p.pipeline.config.events.unknown_cooldown_s
    inside_window = cooldown / 2
    try:
        p.pipeline._process(_frame())
        clock.advance(inside_window)
        p.pipeline._process(_lit_frame())  # motion -> first unknown event
        assert len(p.events.recent()) == 1

        clock.advance(inside_window)
        p.pipeline._process(_frame())  # motion (white->black) but inside cooldown
        assert len(p.events.recent()) == 1

        clock.advance(cooldown + 1)
        p.pipeline._process(_lit_frame())  # motion past the cooldown
        rows = p.events.recent()
        assert len(rows) == 2
        assert all(r["type"] == "unknown" and r["name"] is None for r in rows)
    finally:
        p.events.close()


def test_known_cooldown_suppresses_repeat_alerts(tmp_path):
    """Known faces share one cooldown slot per identity: a second motion
    event inside the window is suppressed, one past it records again."""
    clock = FakeClock()
    p = _build_direct(tmp_path, [_face()], clock=clock, known_cooldown_s=60.0)
    # Steps derive from the configured cooldown so changing its value keeps
    # the inside/outside-window timing intact.
    cooldown = p.pipeline.config.events.known_cooldown_s
    inside_window = cooldown / 2
    try:
        p.pipeline._process(_frame())
        clock.advance(inside_window)
        p.pipeline._process(_lit_frame())  # motion -> first known event
        assert len(p.events.recent()) == 1

        clock.advance(inside_window)
        p.pipeline._process(_frame())  # motion (white->black) but inside cooldown
        assert len(p.events.recent()) == 1

        clock.advance(cooldown + 1)
        p.pipeline._process(_lit_frame())  # motion past the cooldown
        rows = p.events.recent()
        assert len(rows) == 2
        assert all(r["type"] == "known" and r["name"] == "ann" for r in rows)
    finally:
        p.events.close()


def test_duplicate_same_identity_in_one_frame_alerts_once(tmp_path):
    """Two detections of one identity in a single frame stamp the same
    cooldown slot: the first records, the second is suppressed."""
    p = _build_clip(tmp_path, [_face(x=10), _face(x=80)], known_cooldown_s=60.0)
    try:
        p.pipeline.run()
        assert [e["name"] for e in p.events.recent()] == ["ann"]
    finally:
        p.events.close()


def test_two_different_identities_in_one_frame_both_alert(tmp_path):
    clock = FakeClock()
    recognizer = _recognizer_seq([_vec(1.0, 0.0), _vec(0.0, 1.0)])
    p = _build_direct(
        tmp_path,
        [_face(x=10), _face(x=80)],
        enrolled=[("ann", (1.0, 0.0)), ("bob", (0.0, 1.0))],
        recognizer=recognizer,
        reader=FakeReader(frame=_frame()),
        clock=clock,
        known_cooldown_s=60.0,
    )
    try:
        p.pipeline._process(_frame())
        p.pipeline._process(_lit_frame())  # motion -> both faces identified
        assert len(p.state.current.faces) == 2
        assert {r.name for r in p.state.current.faces} == {"ann", "bob"}
        rows = p.events.recent()
        assert len(rows) == 2
        assert {e["name"] for e in rows} == {"ann", "bob"}
    finally:
        p.events.close()


def test_unknown_twice_in_one_frame_alerts_once(tmp_path):
    """Unknowns collapse to the 'unknown' cooldown key, so two unknowns in
    one frame alert once."""
    clock = FakeClock()
    recognizer = _recognizer_seq([_vec(0.0, 1.0), _vec(0.0, 1.0)])
    p = _build_direct(
        tmp_path,
        [_face(x=10), _face(x=80)],
        recognizer=recognizer,
        reader=FakeReader(frame=_frame()),
        clock=clock,
        unknown_cooldown_s=60.0,
    )
    try:
        p.pipeline._process(_frame())
        p.pipeline._process(_lit_frame())
        rows = p.events.recent()
        assert len(rows) == 1
        assert rows[0]["type"] == "unknown"
    finally:
        p.events.close()


def test_snapshots_do_not_collide_within_one_second(tmp_path):
    # snapshot_name has ms resolution. Drive several motion events 5ms apart
    # (same wall-clock second) and require distinct paths.
    clock = FakeClock()
    p = _build_direct(tmp_path, [_face()], clock=clock, known_cooldown_s=0.0)
    try:
        p.pipeline._process(_frame())
        for i in range(5):
            clock.advance(0.005)
            p.pipeline._process(_lit_frame() if i % 2 == 0 else _frame())
        paths = [e["path"] for e in p.events.recent()]
        assert len(paths) > 1, "expected multiple motion events"
        assert len(paths) == len(set(paths))
    finally:
        p.events.close()


def test_process_does_not_mutate_input_frame(tmp_path):
    """The pipeline publishes the frame it was given (zero-copy) and must
    not draw on it: readers hold the reference, so in-place writes would
    corrupt already-published state."""
    p = _build_direct(tmp_path, [_face()])
    try:
        p.pipeline._process(_frame())
        frame = _lit_frame()
        before = frame.copy()
        p.pipeline._process(frame)
        assert np.array_equal(frame, before)
        assert p.state.current.frame is frame
    finally:
        p.events.close()


def test_run_terminates_at_end_of_file_and_is_replayable(parts):
    parts.pipeline.run()
    assert len(parts.state.history) == N_FRAMES
    parts.reader.stop()  # idempotent; must not raise
    parts.pipeline.run()  # file source reopens; second replay is safe
    assert len(parts.state.history) == 2 * N_FRAMES


def test_run_calls_reader_stop_on_eof(tmp_path):
    reader = FakeReader(frames=[_frame()], is_live=False)
    p = _build_direct(tmp_path, [_face()], reader=reader, state=RecordingState())
    try:
        p.pipeline.run()
        assert reader.start_calls == 1
        assert reader.stopped == [True]
        assert p.state.current is not None
    finally:
        p.events.close()


def test_run_rejects_concurrent_start(tmp_path):
    """A second run() while the loop is live fails loudly instead of interleaving."""
    reader = FakeReader(frame=_frame())
    state = RecordingState()
    p = _build_direct(tmp_path, [_face()], reader=reader, state=state)
    thread = threading.Thread(target=p.pipeline.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while state.current is None:
            assert time.monotonic() < deadline, "pipeline never started"
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="already running"):
            p.pipeline.run()
    finally:
        p.pipeline.stop()
        thread.join(timeout=5)
        p.events.close()
    assert not thread.is_alive()
    assert reader.stopped == [True]


def test_stop_breaks_out_of_an_endless_loop(tmp_path):
    """Endless live source: stop() from another thread must break run()
    and fire reader.stop() exactly once."""
    reader = FakeReader(frame=np.zeros((120, 160, 3), dtype=np.uint8))
    state = RecordingState()
    p = _build_direct(tmp_path, [], reader=reader, state=state)
    try:
        thread = threading.Thread(target=p.pipeline.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while state.current is None:
            assert time.monotonic() < deadline, "pipeline never started"
            time.sleep(0.01)
        p.pipeline.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert state.current is not None
        assert reader.stopped == [True]
    finally:
        p.events.close()


def test_telemetry_reports_fps_and_motion(monkeypatch, caplog, parts):
    import logging

    # Force a log line per frame so caplog sees telemetry deterministically.
    monkeypatch.setattr("edge_surveillance.pipeline._LOG_INTERVAL_S", 0.0)
    with caplog.at_level(logging.INFO, logger="edge_surveillance.pipeline"):
        parts.pipeline.run()
    assert "fps=" in caplog.text
    assert "people=1" in caplog.text
    assert any(s.fps > 0 for s in parts.state.history)
