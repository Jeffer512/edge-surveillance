import argparse
import logging
import sys

from edge_surveillance.camera.stream_reader import StreamReader
from edge_surveillance.config import DetectorConfig, RecognizerConfig, load_config
from edge_surveillance.core.face_detector import YuNetDetector
from edge_surveillance.core.face_recognizer import MobileFaceNetRecognizer
from edge_surveillance.core.motion_detector import MotionDetector
from edge_surveillance.pipeline import SurveillancePipeline
from edge_surveillance.state import SharedState
from edge_surveillance.store.events import EventStore
from edge_surveillance.store.gallery import GalleryStore

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run the surveillance pipeline.")
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument("--source", default=None, help="override the camera source")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"{e}\nCopy config/config.example.yaml to config/config.yaml to create one.", file=sys.stderr)
        return 1
    if args.source:
        config.camera.source = args.source

    try:
        reader = StreamReader(config.camera)
        motion = MotionDetector(config.motion)
        detector = YuNetDetector(DetectorConfig(**vars(config.detector)))
        recognizer = MobileFaceNetRecognizer(RecognizerConfig(**vars(config.recognizer)))
    except FileNotFoundError as e:
        print(f"Missing weights: {e}\nRun: python scripts/download_models.py", file=sys.stderr)
        return 1

    events = EventStore(config.store.events_db)
    pipeline = SurveillancePipeline(
        config,
        reader=reader,
        motion=motion,
        detector=detector,
        recognizer=recognizer,
        gallery=GalleryStore(config.store.gallery),
        events=events,
        state=SharedState(),
    )
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        events.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
