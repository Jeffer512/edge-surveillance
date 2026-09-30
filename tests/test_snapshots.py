import re
from pathlib import Path

import numpy as np
import pytest

from edge_surveillance.store.snapshots import safe_name, save_snapshot, snapshot_name, snapshot_path


def test_safe_name_strips_separators_and_dots():
    assert safe_name("ann") == "ann"
    assert safe_name("../../etc/passwd") == "etc_passwd"
    assert "/" not in safe_name("a/b/c")
    assert ".." not in safe_name("a..b")


def test_safe_name_collapses_and_falls_back():
    assert safe_name("my person") == "my_person"
    assert safe_name("***") == "unknown"
    assert safe_name("") == "unknown"
    assert safe_name("   ") == "unknown"


def test_snapshot_name_has_timestamp_ms_and_name():
    name = snapshot_name("ann", 1759000000.456)
    assert re.fullmatch(r"\d{8}_\d{6}_\d{3}_ann\.jpg", name), name


def test_snapshot_name_is_unique_within_same_second():
    a = snapshot_name("ann", 1759000000.100)
    b = snapshot_name("ann", 1759000000.200)
    assert a != b


def test_snapshot_path_stays_inside_directory():
    path = snapshot_path("/tmp/events", "../../evil", 1759000000.0)
    assert path.parent == Path("/tmp/events")


def test_save_snapshot_writes_readable_image(tmp_path):
    frame = np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    path = save_snapshot(frame, snapshot_path(tmp_path / "ev", "ann", 1759000000.0))
    assert path.is_file()
    import cv2

    assert cv2.imread(str(path)) is not None


def test_save_snapshot_creates_missing_directories(tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    path = save_snapshot(frame, tmp_path / "deep" / "nested" / "a.jpg")
    assert path.is_file()


def test_save_snapshot_raises_oserror_when_parent_is_a_file(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(OSError):
        save_snapshot(frame, blocker / "sub" / "a.jpg")


def test_save_snapshot_raises_oserror_when_opencv_cannot_encode(tmp_path):
    """cv2.imwrite raises cv2.error rather than returning False; callers
    must see an OSError, not an exception type they do not handle."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(OSError):
        save_snapshot(frame, tmp_path / "bad\0name.jpg")
