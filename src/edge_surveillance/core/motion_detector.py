from dataclasses import dataclass

import cv2
import numpy as np

from edge_surveillance.config import MotionConfig


@dataclass
class MotionResult:
    is_motion: bool
    change_percent: float


class MotionDetector:
    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self._min_area_px = self.config.min_area_percent / 100.0 * self.config.probe_width * self.config.probe_height
        self._prev: np.ndarray | None = None
        self._last_fire: float = 0.0
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    def reset(self) -> None:
        self._prev = None
        self._last_fire = 0.0

    def detect(self, frame: np.ndarray, now: float | None = None) -> MotionResult:
        import time

        if frame is None or frame.size == 0:
            raise ValueError("empty frame")
        now = time.monotonic() if now is None else now

        small = cv2.resize(frame, (self.config.probe_width, self.config.probe_height))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev is None:
            self._prev = gray
            return MotionResult(False, 0.0)

        diff = cv2.absdiff(self._prev, gray)
        self._prev = gray

        _, mask = cv2.threshold(diff, self.config.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        mask = cv2.dilate(mask, self._dilate_kernel, iterations=1)

        change_percent = float(np.count_nonzero(mask)) / float(mask.size) * 100.0

        found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        has_large_blob = any(cv2.contourArea(c) >= self._min_area_px for c in contours)

        triggered = change_percent >= self.config.min_change_percent and has_large_blob
        if triggered and now - self._last_fire < self.config.cooldown_s:
            return MotionResult(False, change_percent)
        if triggered:
            self._last_fire = now
        return MotionResult(triggered, change_percent)
