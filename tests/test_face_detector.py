import numpy as np
import pytest

from edge_surveillance.config import DetectorConfig
from edge_surveillance.core.face_detector import YuNetDetector, parse_faces


def _row(x=10, y=20, w=100, h=120, score=0.95):
    landmarks = [30, 40, 70, 40, 50, 60, 35, 90, 65, 90]  # 5x2
    return [x, y, w, h, *landmarks, score]


def test_parse_none_and_empty():
    assert parse_faces(None) == []
    assert parse_faces(np.zeros((0, 15), dtype=np.float32)) == []


def test_parse_row_fields():
    dets = parse_faces(np.array([_row()], dtype=np.float32))
    assert len(dets) == 1
    d = dets[0]
    assert (d.x, d.y, d.w, d.h) == (10, 20, 100, 120)
    assert d.score == pytest.approx(0.95)
    assert d.landmarks.shape == (5, 2)
    assert d.landmarks[0].tolist() == pytest.approx([30, 40])


def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        YuNetDetector(DetectorConfig(model="models/does-not-exist.onnx"))
