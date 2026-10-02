import logging
import time
from collections.abc import Callable

import numpy as np

from edge_surveillance.camera.stream_reader import StreamReader
from edge_surveillance.config import AppConfig
from edge_surveillance.core.face_detector import FaceDetection, YuNetDetector
from edge_surveillance.core.face_recognizer import MobileFaceNetRecognizer
from edge_surveillance.core.motion_detector import MotionDetector
from edge_surveillance.state import FaceReport, FrameState, SharedState
from edge_surveillance.store.events import EventStore
from edge_surveillance.store.gallery import GalleryStore
from edge_surveillance.store.snapshots import save_snapshot, snapshot_path

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"
_LOG_INTERVAL_S = 10.0
# Tolerates single detector misses (profile, occlusion) before reports clear.
_MAX_MISS_FRAMES = 5


def claim_alert(last_alert: dict[str, float], key: str, now: float, cooldown: float) -> bool:
    """Reserve key's alert slot for now. False if still inside its cooldown.

    Stamps last_alert on success, so a second face with the same identity
    in the same frame is suppressed by the first one.
    """
    last = last_alert.get(key)
    if last is not None and now - last < cooldown:
        return False
    last_alert[key] = now
    return True


class SurveillancePipeline:
    """Camera -> motion gate -> face detect -> embed -> match -> record.

    Face models only run on frames the motion gate accepts. Publishes every
    frame to SharedState for the server; The frame is published raw and only
    encoded when requested.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        reader: StreamReader,
        motion: MotionDetector,
        detector: YuNetDetector,
        recognizer: MobileFaceNetRecognizer,
        gallery: GalleryStore,
        events: EventStore,
        state: SharedState,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._reader = reader
        self._motion = motion
        self._detector = detector
        self._recognizer = recognizer
        self._gallery = gallery
        self._events = events
        self._state = state
        self._clock = clock
        self._running = False
        self._last_alert: dict[str, float] = {}
        self._last_faces: tuple[FaceReport, ...] = ()
        self._last_faces_at = 0.0
        self._misses = 0
        self._period = 1.0 / config.camera.fps if config.camera.fps > 0 else 0.0
        self._frames = 0
        self._window_start = 0.0
        self._fps = 0.0

    def run(self) -> None:
        """Loop until stop() or end of a file source.

        Blocking; not re-entrant concurrently: call stop(), join the thread,
        then restart. Raises RuntimeError if a loop is already running.
        """
        if self._running:
            raise RuntimeError("pipeline already running")
        self._running = True
        self._reader.start()
        self._window_start = time.monotonic()
        try:
            while self._running:
                started = time.monotonic()
                ok, frame = self._reader.read()
                self._gallery.refresh()
                if ok and frame is not None:
                    self._process(frame)
                    self._count_frame()
                elif not self._reader.is_live:
                    logger.info("End of video source.")
                    break
                self._throttle(started)
        finally:
            self._running = False
            self._reader.stop()
            logger.info("Pipeline stopped.")

    def stop(self) -> None:
        """Signal run() to exit; thread-safe, may be called from any thread."""
        self._running = False

    def _count_frame(self) -> None:
        self._frames += 1
        elapsed = time.monotonic() - self._window_start
        if elapsed >= _LOG_INTERVAL_S:
            self._fps = self._frames / elapsed
            self._frames = 0
            self._window_start = time.monotonic()
            logger.info(
                "fps=%.1f | people=%d | last motion=%.2f%%",
                self._fps,
                len(self._gallery.list_people()),
                self._state.current.motion_percent if self._state.current else 0.0,
            )

    def _throttle(self, started: float) -> None:
        """Cap live processing rate; file sources replay as fast as they decode."""
        if not self._reader.is_live or self._period <= 0:
            return
        elapsed = time.monotonic() - started
        if elapsed < self._period:
            time.sleep(self._period - elapsed)

    def _process(self, frame: np.ndarray) -> None:
        """Motion-gated detect -> identify -> record; publishes every frame.

        The input frame is published as-is (zero-copy) and never mutated.
        Static frames hold the last reports for report_ttl_s; motion frames
        with zero detections tolerate up to _MAX_MISS_FRAMES misses, then
        clear so departures read truthfully.
        """
        now = self._clock()
        motion = self._motion.detect(frame)
        reports: list[FaceReport] = []
        if motion.is_motion:
            for det in self._detector.detect(frame):
                name, score = self._identify(frame, det)
                reports.append(
                    FaceReport(name=name, score=score, x=det.x, y=det.y, w=det.w, h=det.h)
                )
                self._record(name, score, frame, now)
        if reports:
            self._last_faces = tuple(reports)
            self._last_faces_at = now
            self._misses = 0
        elif motion.is_motion:
            self._misses += 1
        else:
            self._misses = 0
        self._state.update(
            FrameState(
                captured_at=now,
                frame=frame,
                faces=self._visible_faces(now, motion.is_motion),
                motion_percent=motion.change_percent,
                fps=self._fps,
            )
        )

    def _visible_faces(self, now: float, is_motion: bool) -> tuple[FaceReport, ...]:
        """Last reports while fresh: static frames hold for report_ttl_s,
        motion-misses hold within the TTL and _MAX_MISS_FRAMES grace.

        Group-level and position-stale by design (no per-face tracking).
        """
        if now - self._last_faces_at > self.config.detector.report_ttl_s:
            return ()
        if is_motion and self._misses > _MAX_MISS_FRAMES:
            return ()
        return self._last_faces

    def _identify(self, frame: np.ndarray, det: FaceDetection) -> tuple[str, float]:
        embedding = self._recognizer.extract(frame, det)
        return self._gallery.match(embedding, self.config.recognizer.similarity_threshold)

    def _record(self, name: str, score: float, frame: np.ndarray, now: float) -> None:
        """Write an event + snapshot unless the slot is cooling down.

        Unknown faces share one "unknown" slot regardless of which face;
        snapshot failure keeps the event row with a null path.
        """
        known = name != UNKNOWN
        key = name if known else "unknown"
        cooldown = (
            self.config.events.known_cooldown_s if known else self.config.events.unknown_cooldown_s
        )
        if not claim_alert(self._last_alert, key, now, cooldown):
            return
        path = snapshot_path(self.config.events.dir, name, now)
        event_id = self._events.add(
            ts=now,
            event_type="known" if known else "unknown",
            name=name if known else None,
            score=score,
            path=str(path),
        )
        try:
            save_snapshot(frame, path)
        except OSError as e:
            logger.warning("Snapshot failed for event %d: %s", event_id, e)
            self._events.set_path(event_id, None)
        logger.info("Event %d: %s (%.2f)", event_id, name, score)
