import argparse
import logging
import sys
import threading

from edge_surveillance.camera.stream_reader import StreamReader
from edge_surveillance.config import load_config
from edge_surveillance.core.face_detector import YuNetDetector
from edge_surveillance.core.face_recognizer import MobileFaceNetRecognizer
from edge_surveillance.core.motion_detector import MotionDetector
from edge_surveillance.pipeline import SurveillancePipeline
from edge_surveillance.state import SharedState
from edge_surveillance.store.events import EventStore
from edge_surveillance.store.gallery import GalleryStore

logger = logging.getLogger(__name__)


def build_pipeline(
    config, state: SharedState, gallery: GalleryStore
) -> tuple[SurveillancePipeline, EventStore]:
    events = EventStore(config.store.events_db)
    pipeline = SurveillancePipeline(
        config,
        reader=StreamReader(config.camera),
        motion=MotionDetector(config.motion),
        detector=YuNetDetector(config.detector),
        recognizer=MobileFaceNetRecognizer(config.recognizer),
        gallery=gallery,
        events=events,
        state=state,
    )
    return pipeline, events


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run the surveillance pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="path to a YAML config file")
    parser.add_argument("--source", default=None, help="override the camera source")
    parser.add_argument(
        "--headless", action="store_true", help="run the pipeline without the web server"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(
            f"{e}\nCopy config/config.example.yaml to config/config.yaml to create one.",
            file=sys.stderr,
        )
        return 1
    if args.source:
        config.camera.source = args.source

    state = SharedState()
    gallery = GalleryStore(config.store.gallery)
    try:
        pipeline, events = build_pipeline(config, state, gallery)
    except FileNotFoundError as e:
        print(f"Missing weights: {e}\nRun: python scripts/download_models.py", file=sys.stderr)
        return 1

    if args.headless:
        try:
            pipeline.run()
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            events.close()
        return 0


    import uvicorn

    from edge_surveillance.server.app import create_app

    # uvicorn must own the main thread: signal handlers can only be
    # installed there, and it needs them for graceful shutdown on Ctrl-C.
    thread = threading.Thread(target=pipeline.run, name="pipeline", daemon=True)
    thread.start()
    app = create_app(state, gallery, events, config.server)
    host = "localhost" if config.server.host == "0.0.0.0" else config.server.host
    logger.info("Dashboard on http://%s:%d", host, config.server.port)
    try:
        uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        thread.join(timeout=5)
        events.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
