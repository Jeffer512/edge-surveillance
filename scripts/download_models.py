"""Fetch pinned model weights and verify them by sha256. Stdlib only so it
runs before dependencies are installed."""

import argparse
import hashlib
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


@dataclass
class ModelEntry:
    dest: str  # final path under --dir
    urls: list[str]  # mirrors tried in order
    sha256: str
    kind: str = "file"  # or "zip:member_name"


MODELS = [
    ModelEntry(
        dest="face_detection_yunet.onnx",
        urls=[
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    ModelEntry(
        dest="mobilefacenet.onnx",
        urls=[
            "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/arcface/arcface_mobilefacenet/pretrained/2022-08-24/arcface_mobilefacenet.zip",
        ],
        sha256="1ef27ee20b9264f00f65d59b2be340001a012ffd686fe6b835ec0d702820f5c9",
        kind="zip:mbf.onnx",
    ),
    ModelEntry(
        dest="mobilefacenet_256.onnx",
        urls=[
            "https://github.com/facex-engine/facex/releases/download/facex-nano-1.0/facex_nano.onnx",
        ],
        sha256="bd0094143bb4c7d04a76dcdad4e332f5985847b921ec4d6678c3fd89aaea3889",
    ),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(entry: ModelEntry, model_dir: Path) -> bool:
    dest = model_dir / entry.dest
    if dest.is_file() and sha256_of(dest) == entry.sha256:
        print(f"{entry.dest}: already present")
        return True
    for url in entry.urls:
        tmp = model_dir / (entry.dest + ".download")
        try:
            print(f"{entry.dest}: downloading from {url}")
            urlretrieve(url, tmp)
            if entry.kind.startswith("zip:"):
                member = entry.kind.split(":", 1)[1]
                with zipfile.ZipFile(tmp) as zf, zf.open(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
                tmp.unlink()
            else:
                tmp.rename(dest)
            if sha256_of(dest) != entry.sha256:
                dest.unlink(missing_ok=True)
                raise OSError("sha256 mismatch (corrupt download or upstream changed)")
            print(f"{entry.dest}: ok")
            return True
        except Exception as e:  # noqa: BLE001 - try the next mirror on any failure
            print(f"  {url}: {e}", file=sys.stderr)
    print(f"{entry.dest}: FAILED", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned model weights.")
    parser.add_argument("--dir", default="models", help="destination directory")
    args = parser.parse_args()
    model_dir = Path(args.dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return 0 if all(ensure(m, model_dir) for m in MODELS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
