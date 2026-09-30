from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def _reject_unknown(cls: type, data: dict[str, Any], section: str) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise TypeError(f"Unknown '{section}' config keys: {sorted(unknown)} (known: {sorted(known)})")
    return data


@dataclass
class CameraConfig:
    source: str = "0"
    width: int = 640
    height: int = 480
    fps: int = 15

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"camera dimensions must be > 0, got {self.width}x{self.height}")
        if self.fps <= 0:
            raise ValueError(f"camera fps must be > 0, got {self.fps}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        return cls(**_reject_unknown(cls, dict(data), "camera"))


@dataclass
class MotionConfig:
    min_change_percent: float = 1.5
    threshold: int = 25
    min_area_percent: float = 1.0
    probe_width: int = 320
    probe_height: int = 240
    cooldown_s: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.min_change_percent <= 100:
            raise ValueError(f"min_change_percent must be in (0, 100], got {self.min_change_percent}")
        if not 0 < self.threshold < 256:
            raise ValueError(f"threshold must be in (0, 256), got {self.threshold}")
        if not 0 < self.min_area_percent <= 100:
            raise ValueError(f"min_area_percent must be in (0, 100], got {self.min_area_percent}")
        if self.probe_width <= 0 or self.probe_height <= 0:
            raise ValueError(f"probe size must be > 0, got {self.probe_width}x{self.probe_height}")
        if self.cooldown_s < 0:
            raise ValueError(f"cooldown_s must be >= 0, got {self.cooldown_s}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotionConfig:
        return cls(**_reject_unknown(cls, dict(data), "motion"))


@dataclass
class DetectorConfig:
    model: str = "models/face_detection_yunet.onnx"
    score_threshold: float = 0.8
    nms_threshold: float = 0.3
    input_size: tuple[int, int] = (320, 320)
    report_ttl_s: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.score_threshold <= 1:
            raise ValueError(f"score_threshold must be in (0, 1], got {self.score_threshold}")
        if not 0 <= self.nms_threshold <= 1:
            raise ValueError(f"nms_threshold must be in [0, 1], got {self.nms_threshold}")
        w, h = self.input_size
        if w <= 0 or h <= 0:
            raise ValueError(f"input_size must be > 0, got {self.input_size}")
        if self.report_ttl_s < 0:
            raise ValueError(f"report_ttl_s must be >= 0, got {self.report_ttl_s}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectorConfig:
        data = _reject_unknown(cls, dict(data), "detector")
        if "input_size" in data and isinstance(data["input_size"], list):
            data["input_size"] = tuple(data["input_size"])
        return cls(**data)


@dataclass
class RecognizerConfig:
    model: str = "models/mobilefacenet.onnx"
    similarity_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.similarity_threshold < 1:
            raise ValueError(f"similarity_threshold must be in (0, 1), got {self.similarity_threshold}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecognizerConfig:
        return cls(**_reject_unknown(cls, dict(data), "recognizer"))


@dataclass
class EventsConfig:
    dir: str = "data/events"
    known_cooldown_s: float = 30.0
    unknown_cooldown_s: float = 60.0

    def __post_init__(self) -> None:
        if self.known_cooldown_s < 0 or self.unknown_cooldown_s < 0:
            raise ValueError("event cooldowns must be >= 0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventsConfig:
        return cls(**_reject_unknown(cls, dict(data), "events"))


@dataclass
class StoreConfig:
    gallery: str = "data/embeddings.pkl"
    events_db: str = "data/events.db"

    def __post_init__(self) -> None:
        if not self.gallery.strip() or not self.events_db.strip():
            raise ValueError("store paths must be non-empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoreConfig:
        return cls(**_reject_unknown(cls, dict(data), "store"))


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServerConfig:
        return cls(**_reject_unknown(cls, dict(data), "server"))


@dataclass
class AppConfig:
    camera: CameraConfig = None  # type: ignore[assignment]
    motion: MotionConfig = None  # type: ignore[assignment]
    detector: DetectorConfig = None  # type: ignore[assignment]
    recognizer: RecognizerConfig = None  # type: ignore[assignment]
    events: EventsConfig = None  # type: ignore[assignment]
    store: StoreConfig = None  # type: ignore[assignment]
    server: ServerConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Allows AppConfig() with zero args -> all defaults.
        if self.camera is None:
            self.camera = CameraConfig()
        if self.motion is None:
            self.motion = MotionConfig()
        if self.detector is None:
            self.detector = DetectorConfig()
        if self.recognizer is None:
            self.recognizer = RecognizerConfig()
        if self.events is None:
            self.events = EventsConfig()
        if self.store is None:
            self.store = StoreConfig()
        if self.server is None:
            self.server = ServerConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        data = dict(data or {})
        known = {"camera", "motion", "detector", "recognizer", "events", "store", "server"}
        unknown = set(data) - known
        if unknown:
            raise TypeError(f"Unknown top-level config sections: {sorted(unknown)}")
        return cls(
            camera=CameraConfig.from_dict(data.get("camera", {})),
            motion=MotionConfig.from_dict(data.get("motion", {})),
            detector=DetectorConfig.from_dict(data.get("detector", {})),
            recognizer=RecognizerConfig.from_dict(data.get("recognizer", {})),
            events=EventsConfig.from_dict(data.get("events", {})),
            store=StoreConfig.from_dict(data.get("store", {})),
            server=ServerConfig.from_dict(data.get("server", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Top-level YAML must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config, injecting defaults for anything unspecified.

    - ``None`` -> pure defaults (useful for tests / Termux without a file).
    - explicit path that is missing -> FileNotFoundError 
    """
    if path is None:
        return AppConfig()
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return AppConfig.from_yaml(cfg_path)
