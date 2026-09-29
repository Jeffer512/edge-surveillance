import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def embed_photos(photos, detector, recognizer) -> list[np.ndarray]:
    """Embed the best face of each photo. Empty list when nothing usable."""
    embeddings = []
    for photo in photos:
        image = cv2.imread(photo)
        if image is None:
            logger.warning("Skipping unreadable image: %s", photo)
            continue
        faces = detector.detect(image)
        if not faces:
            logger.warning("No face in %s", photo)
            continue
        embeddings.append(recognizer.extract(image, max(faces, key=lambda d: d.score)))
    return embeddings
