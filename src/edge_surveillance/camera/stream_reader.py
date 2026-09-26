import logging
import threading
import time

import cv2
import numpy as np

from edge_surveillance.config import CameraConfig

logger = logging.getLogger(__name__)


class StreamReader:
    """Camera source with buffer-draining for live streams.

    File paths read sequentially (deterministic, no thread). Live sources
    (webcam index, rtsp/http URLs) drain via a background thread that keeps
    only the newest frame, so a slow pipeline never accumulates lag.
    """

    def __init__(self, config: CameraConfig | None = None, reconnect_delay: float = 2.0) -> None:
        self.config = config or CameraConfig()
        src = self.config.source
        self.source: int | str = int(src) if src.isdigit() else src
        self.reconnect_delay = reconnect_delay
        self.is_live = isinstance(self.source, int) or (
            isinstance(self.source, str)
            and self.source.startswith(("rtsp://", "http://", "https://"))
        )
        self._cap: cv2.VideoCapture | None = None
        self._latest: np.ndarray | None = None
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> "StreamReader":
        self._connect()
        self._running = True
        if self.is_live:
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
        return self

    def _connect(self) -> None:
        if self._cap is not None:
            self._cap.release()
        logger.info("Connecting to video source: %s", self.source)
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.is_live:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not self._cap.isOpened():
            logger.warning("Could not open source %s. Will retry...", self.source)

    def _capture_loop(self) -> None:
        assert self._cap is not None
        while self._running:
            if not self._cap.isOpened():
                time.sleep(self.reconnect_delay)
                self._connect()
                continue
            if not self._cap.grab():
                logger.warning("Stream dropped or frame empty. Reconnecting...")
                time.sleep(self.reconnect_delay)
                self._connect()
                continue
            ret, frame = self._cap.retrieve()
            if ret:
                with self._lock:
                    self._latest = frame
            else:
                time.sleep(0.01)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        if not self.is_live:
            ret, frame = self._cap.read()
            return (True, frame) if ret else (False, None)
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def get_dimensions(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest = None
        logger.info("Camera stream released.")
