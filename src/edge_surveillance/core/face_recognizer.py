from pathlib import Path

import cv2
import numpy as np

from edge_surveillance.config import RecognizerConfig
from edge_surveillance.core.face_detector import FaceDetection

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# Canonical 5-point template for 112x112 crops (InsightFace alignment reference):
# right eye, left eye, nose tip, right mouth corner, left mouth corner.
TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(frame: np.ndarray, detection: FaceDetection, size: int = 112) -> np.ndarray:
    """Warp the face region to a frontal size x size BGR crop using 5 landmarks."""
    transform, _ = cv2.estimateAffinePartial2D(detection.landmarks, TEMPLATE_112)
    if transform is None:
        x, y, w, h = detection.x, detection.y, detection.w, detection.h
        crop = frame[max(y, 0) : y + h, max(x, 0) : x + w]
        return cv2.resize(crop, (size, size))
    return cv2.warpAffine(frame, transform, (size, size))


def preprocess(face112: np.ndarray) -> np.ndarray:
    """112x112 BGR crop -> (1, 3, 112, 112) float32 tensor normalized to [-1, 1]."""
    rgb = cv2.cvtColor(face112, cv2.COLOR_BGR2RGB)
    return np.expand_dims(((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1), axis=0)


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    flat = embedding.reshape(-1).astype(np.float32)
    return flat / (np.linalg.norm(flat) + 1e-10)


def cosine_match(
    query: np.ndarray, gallery: np.ndarray, names: list[str], threshold: float
) -> tuple[str, float]:
    """Best cosine match of a normalized query against a row-normalized gallery."""
    if len(names) == 0 or gallery.size == 0:
        return "Unknown", 0.0
    query_flat = query.flatten()
    if gallery.shape != (len(names), query_flat.size):
        raise ValueError(f"gallery shape {gallery.shape} != ({len(names)}, {query_flat.size})")
    scores = gallery @ query_flat
    best = int(np.argmax(scores))
    score = float(scores[best])
    return (names[best], score) if score >= threshold else ("Unknown", score)


class MobileFaceNetRecognizer:
    def __init__(self, config: RecognizerConfig | None = None, backend: str = "auto") -> None:
        self.config = config or RecognizerConfig()
        if backend not in ("auto", "onnxruntime", "opencv"):
            raise ValueError(f"backend must be auto|onnxruntime|opencv, got {backend!r}")
        model_path = Path(self.config.model)
        if not model_path.is_file():
            raise FileNotFoundError(f"MobileFaceNet weights not found: {model_path}")
        use_ort = backend == "onnxruntime" or (backend == "auto" and _ORT_AVAILABLE)
        self.backend = "onnxruntime" if use_ort else "opencv"
        if self.backend == "onnxruntime":
            self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._net = None
        else:
            self._session = None
            self._net = cv2.dnn.readNetFromONNX(str(model_path))

    def extract(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Full path: align the detected face and return its L2-normalized embedding."""
        if frame is None or frame.size == 0:
            raise ValueError("empty frame")
        blob = preprocess(align_face(frame, detection))
        if self.backend == "onnxruntime":
            raw = self._session.run(None, {self._input_name: blob})[0]
            return l2_normalize(raw)
        self._net.setInput(blob)
        return l2_normalize(self._net.forward())
