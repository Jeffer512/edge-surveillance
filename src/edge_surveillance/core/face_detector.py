from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from edge_surveillance.config import DetectorConfig


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int
    landmarks: np.ndarray  # shape (5, 2): right eye, left eye, nose, right mouth, left mouth
    score: float


def parse_faces(raw: np.ndarray | None) -> list[FaceDetection]:
    """Convert YuNet output rows to FaceDetection objects."""
    if raw is None or len(raw) == 0:
        return []
    out: list[FaceDetection] = []
    for row in raw:
        x, y, w, h = row[:4].astype(int)
        out.append(
            FaceDetection(
                x=x,
                y=y,
                w=w,
                h=h,
                landmarks=np.array(row[4:14], dtype=np.float32).reshape(5, 2),
                score=float(row[14]),
            )
        )
    return out


class YuNetDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        model_path = Path(self.config.model)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YuNet weights not found: {model_path} (download to models/ first)"
            )
        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=self.config.input_size,
            score_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
            top_k=50,
        )
        self._frame_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if frame is None or frame.size == 0:
            raise ValueError("empty frame")
        h, w = frame.shape[:2]
        if (w, h) != self._frame_size:
            self._detector.setInputSize((w, h))
            self._frame_size = (w, h)
        _, faces = self._detector.detect(frame)
        return parse_faces(faces)
