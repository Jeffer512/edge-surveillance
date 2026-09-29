import argparse
import logging
import sys
from pathlib import Path

from edge_surveillance.config import DetectorConfig, RecognizerConfig
from edge_surveillance.core.embed_photos import embed_photos
from edge_surveillance.core.face_detector import YuNetDetector
from edge_surveillance.core.face_recognizer import MobileFaceNetRecognizer
from edge_surveillance.store.gallery import GalleryStore

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Enroll a person from photos.")
    parser.add_argument("name", help="person label, e.g. Alice")
    parser.add_argument("photos", nargs="+", help="one or more photo paths")
    parser.add_argument("--gallery", default="data/embeddings.pkl", help="gallery pickle path")
    parser.add_argument("--detector-model", default=DetectorConfig().model)
    parser.add_argument("--recognizer-model", default=RecognizerConfig().model)
    args = parser.parse_args(argv)

    gallery_path = Path(args.gallery)
    try:
        detector = YuNetDetector(DetectorConfig(model=args.detector_model))
        recognizer = MobileFaceNetRecognizer(RecognizerConfig(model=args.recognizer_model))
    except FileNotFoundError as e:
        print(f"Missing weights: {e}\nRun: python scripts/download_models.py", file=sys.stderr)
        return 1

    embeddings = embed_photos(args.photos, detector, recognizer)
    if not embeddings:
        return 1
    store = GalleryStore(gallery_path)
    store.add(args.name, embeddings)
    print(f"Enrolled {args.name!r} from {len(embeddings)} photo(s) -> {gallery_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
