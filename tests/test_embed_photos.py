import cv2
import numpy as np
import pytest

from edge_surveillance.core.embed_photos import embed_photos
from edge_surveillance.core.face_detector import FaceDetection


class StubDetector:
    def __init__(self, faces):
        self._faces = faces

    def detect(self, _frame):
        return self._faces


class StubRecognizer:
    """Encodes the detection's score as the embedding, so the returned value
    reveals which face embed_photos selected."""

    def extract(self, _frame, detection):
        return np.array([detection.score, 0.0], dtype=np.float32)


def _detection(score=0.9):
    return FaceDetection(x=0, y=0, w=10, h=10, landmarks=np.zeros((5, 2)), score=score)


def _blank_photo(tmp_path, name="face.jpg"):
    path = tmp_path / name
    cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))
    return str(path)


def test_embed_photos_picks_best_scoring_face(tmp_path):
    out = embed_photos(
        [_blank_photo(tmp_path)],
        StubDetector([_detection(0.5), _detection(0.9)]),
        StubRecognizer(),
    )
    assert len(out) == 1
    assert out[0].tolist() == pytest.approx([0.9, 0.0])


def test_embed_photos_skips_faceless_and_unreadable(tmp_path):
    photos = [_blank_photo(tmp_path, "a.jpg"), str(tmp_path / "missing.jpg")]
    assert embed_photos(photos, StubDetector([]), StubRecognizer()) == []


def test_embed_photos_one_per_usable_photo(tmp_path):
    photos = [_blank_photo(tmp_path, "a.jpg"), _blank_photo(tmp_path, "b.jpg")]
    out = embed_photos(photos, StubDetector([_detection()]), StubRecognizer())
    assert len(out) == 2
