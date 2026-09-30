from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FaceReport:
    name: str
    score: float
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True, eq=False)
class FrameState:
    """One published frame plus what the pipeline saw in it.

    The frame is kept raw: the server encodes it to JPEG on demand, and
    structured face boxes let the UI draw its own overlay. The pipeline
    never writes into a published frame, so readers need no lock.
    """

    captured_at: float
    frame: np.ndarray = field(repr=False)
    faces: tuple[FaceReport, ...] = ()
    motion_percent: float = 0.0
    fps: float = 0.0


class SharedState:
    """Latest pipeline output, published by a single atomic rebind.

    Readers either see the previous FrameState or the new one, never a
    half-updated pair, so no lock is needed.
    """

    def __init__(self) -> None:
        self._current: FrameState | None = None

    @property
    def current(self) -> FrameState | None:
        return self._current

    def update(self, state: FrameState) -> None:
        self._current = state
