import numpy as np
import pytest

from edge_surveillance.config import RecognizerConfig
from edge_surveillance.core.face_detector import FaceDetection
from edge_surveillance.core.face_recognizer import (
    TEMPLATE_112,
    MobileFaceNetRecognizer,
    align_face,
    cosine_match,
    l2_normalize,
    preprocess,
)


def _detection():
    return FaceDetection(x=10, y=20, w=100, h=120, landmarks=TEMPLATE_112.copy(), score=0.9)


def test_preprocess_range_and_shape():
    white = np.full((112, 112, 3), 255, dtype=np.uint8)
    blob = preprocess(white)
    assert blob.shape == (1, 3, 112, 112)
    assert blob.dtype == np.float32
    assert float(blob.max()) == pytest.approx((255 - 127.5) / 128.0)
    black = np.zeros((112, 112, 3), dtype=np.uint8)
    assert float(preprocess(black).min()) == pytest.approx(-127.5 / 128.0)


def test_l2_normalize_unit_length():
    v = l2_normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v.tolist() == pytest.approx([0.6, 0.8])


def test_align_identity_when_landmarks_match_template():
    frame = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    out = align_face(frame, _detection())
    assert out.shape == (112, 112, 3)
    assert np.allclose(out, frame, atol=2.0)


def test_cosine_match_known_and_unknown():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    name, score = cosine_match(np.array([1.0, 0.0]), gallery, ["ann", "bob"], 0.5)
    assert (name, score) == ("ann", pytest.approx(1.0))
    name, score = cosine_match(np.array([0.7, 0.7]), gallery, ["ann", "bob"], 0.99)
    assert name == "Unknown"
    assert cosine_match(np.array([1.0, 0.0]), np.zeros((0, 2)), [], 0.5) == ("Unknown", 0.0)


def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        MobileFaceNetRecognizer(RecognizerConfig(model="models/does-not-exist.onnx"))
    with pytest.raises(ValueError):
        MobileFaceNetRecognizer(backend="bogus")
