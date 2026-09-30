import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_name(name: str) -> str:
    """Reduce a person label to characters that cannot escape a directory."""
    return _UNSAFE.sub("_", name).strip("_") or "unknown"


def snapshot_name(name: str, ts: float) -> str:
    # Local time: filenames should read the way the operator thinks about "when".
    moment = datetime.fromtimestamp(ts).astimezone()
    return f"{moment:%Y%m%d_%H%M%S}_{moment.microsecond // 1000:03d}_{safe_name(name)}.jpg"


def snapshot_path(directory: str | Path, name: str, ts: float) -> Path:
    """Build the snapshot path without touching the filesystem."""
    return Path(directory) / snapshot_name(name, ts)


def save_snapshot(frame: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(path), frame)
    except cv2.error as e:  # imwrite raises instead of returning False on some failures
        raise OSError(f"could not write snapshot: {path}") from e
    if not written:
        raise OSError(f"could not write snapshot: {path}")
    return path
