import cv2
import numpy as np

from edge_surveillance.state import FaceReport

UNKNOWN = "Unknown"
FONT = cv2.FONT_HERSHEY_SIMPLEX
KNOWN_COLOR = (74, 222, 128)  # operational green, BGR
UNKNOWN_COLOR = (38, 38, 220)  # incident red, BGR


def draw_overlay(frame: np.ndarray, faces: tuple[FaceReport, ...]) -> np.ndarray:
    """Return a copy of frame with boxes and labels drawn on it.

    A separate JSON channel cannot stay in sync with the browser's decode.
    """
    out = frame.copy()
    for face in faces:
        known = face.name != UNKNOWN
        color = KNOWN_COLOR if known else UNKNOWN_COLOR
        cv2.rectangle(out, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2)
        label = face.name if known else UNKNOWN
        cv2.putText(
            out,
            f"{label} {face.score:.2f}",
            (face.x, max(18, face.y - 8)),
            FONT,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out
