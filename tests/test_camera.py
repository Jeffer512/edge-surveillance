import time

import cv2
import numpy as np
import pytest

from edge_surveillance.camera.stream_reader import StreamReader
from edge_surveillance.config import CameraConfig


def _make_clip(path: str, n: int = 10, w: int = 64, h: int = 48) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    assert writer.isOpened(), "VideoWriter failed to open"
    for i in range(n):
        writer.write(np.full((h, w, 3), i * 20, dtype=np.uint8))
    writer.release()


def test_file_reads_in_order(tmp_path):
    clip = str(tmp_path / "clip.mp4")
    _make_clip(clip)
    reader = StreamReader(CameraConfig(source=clip)).start()
    try:
        for i in range(10):
            ret, frame = reader.read()
            assert ret is True
            assert frame is not None
            assert int(frame.mean()) == pytest.approx(i * 20, abs=6.0)
        assert reader.read() == (False, None)  # EOF
    finally:
        reader.stop()


def test_file_dimensions(tmp_path):
    clip = str(tmp_path / "clip.mp4")
    _make_clip(clip, w=64, h=48)
    reader = StreamReader(CameraConfig(source=clip)).start()
    try:
        assert reader.get_dimensions() == (64, 48)
    finally:
        reader.stop()


def test_missing_file_returns_false_without_hanging():
    reader = StreamReader(CameraConfig(source="/does/not/exist.mp4")).start()
    try:
        assert reader.read() == (False, None)
    finally:
        reader.stop()
        reader.stop()  # idempotent


def test_unreachable_stream_returns_false_without_hanging():
    reader = StreamReader(
        CameraConfig(source="http://127.0.0.1:9/nonexistent"), reconnect_delay=0.01
    ).start()
    try:
        time.sleep(0.05)
        ret, _ = reader.read()
        assert ret is False
    finally:
        reader.stop()
