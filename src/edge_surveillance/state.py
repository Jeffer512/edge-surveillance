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

    frame is made read-only so a reader cannot mutate the snapshot the
    other readers are holding.
    """

    captured_at: float
    frame: np.ndarray = field(repr=False)
    faces: tuple[FaceReport, ...] = ()
    motion_percent: float = 0.0
    fps: float = 0.0

    def __post_init__(self) -> None:
        self.frame.setflags(write=False)


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
