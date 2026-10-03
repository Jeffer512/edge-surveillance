# edge-surveillance

Local motion detection and face recognition (MobileFaceNet) for PC and Termux/Android. Classical motion gating with **YuNet** and **arcface MobileFaceNet** on **ONNX Runtime**, a **FastAPI** dashboard, and SQLite event logging.

![Python](https://img.shields.io/badge/python-3.10+-blue)

## Overview

Point it at a webcam, an RTSP/HTTP phone camera, or a video file. Frames with no motion cost one grayscale subtraction; only motion frames pay for face detection and embedding. Enrolled people are recognized by name, everyone else is logged as `Unknown`.

### Features

- Motion gating — face models run only on frames that actually changed
- MobileFaceNet 512-d embeddings via `onnxruntime`, falling back to `cv2.dnn`
- Sources: webcam index, `rtsp://`, `http://` (IP Webcam), or a video file
- Live sources keep only the newest frame, so a slow pipeline never falls behind
- Enrollment from photos; the gallery is a locked, atomically-written pickle
- SQLite event log (WAL) with per-identity cooldowns and timestamped snapshots
- Local dashboard: live MJPEG feed, event history, people management
- No build step — a single HTML file served out of the package
- Nothing phones home; weights are pinned and sha256-verified at download

## Architecture

```
camera ─▶ motion (frame diff + morphology) ─▶ YuNet ─▶ MobileFaceNet ─▶ match ─▶ event + snapshot
                        │                  └──── only on motion frames ────┘
                        └──▶ SharedState (atomic rebind, one latest frame) ──▶ FastAPI dashboard
```

`SharedState` holds the newest frame and structured face boxes. The pipeline publishes by rebinding a single attribute, so readers need no lock and never observe a half-written frame.

## Prerequisites

- Python 3.10+
- A source: webcam index, `rtsp://`/`http://` URL, or a video file

## Quick start (PC)

```bash
git clone <repo-url>
cd edge-surveillance

uv sync                                    # or: pip install -e . && pip install fastapi uvicorn
uv run python scripts/download_models.py   # pinned weights, sha256-verified
cp config/config.example.yaml config/config.yaml

uv run edge-enroll Alice photos/alice/*.jpg
uv run edge-run
```

Open `http://127.0.0.1:8000`.

## Quick start (Termux)

numpy, `opencv-python`, and `onnxruntime` publish no Android wheels, so on the phone they come from Termux's own repo rather than PyPI. The setup script installs those with `pkg`, then layers pip on top for everything else:

```bash
pkg install -y git
git clone <repo-url> && cd edge-surveillance

bash scripts/termux_setup.sh            # pipeline + dashboard
bash scripts/termux_setup.sh --headless # pipeline only, skips fastapi/uvicorn
bash scripts/termux_setup.sh --venv     # install pyyaml fastapi and uvicorn packages in .venv
```

Then the same as the PC:

```bash
python scripts/download_models.py
cp config/config.example.yaml config/config.yaml
edge-enroll Alice photos/alice/*.jpg
edge-run
```

The script installs the package with `--no-deps` so pip never re-resolves the `pkg`-managed packages.

## Usage

1. **Choose a source** — set `camera.source` in `config/config.yaml`: `0` for a webcam, `http://127.0.0.1:8080/video` for Android IP Webcam, `rtsp://…`, or `path/to/video.mp4`. `--source` overrides it for one run. Files replay as fast as they decode; live sources are paced to `camera.fps`.
2. **Enroll people** — `edge-enroll <name> <photo> [photo …]` embeds the best face from each photo and averages them into one row. A running app notices gallery changes without a restart.
3. **Watch the dashboard** — *Live* shows the feed with server-drawn name/score boxes, *Events* lists the log with thumbnails, *People* lists enrolled names and removes them.
4. **Skip the dashboard** — `edge-run --headless` runs the pipeline alone and never imports FastAPI.

## Dependency groups

| Group | Contents | Install |
|---|---|---|
| *(base)* | opencv-python, numpy, pyyaml, onnxruntime | always |
| `server` | fastapi, uvicorn | `uv sync --group server` |
| `dev` | pytest, ruff, httpx2, plus `server` | `uv sync` (default) |

`uv sync` installs `dev` by default. For a pipeline-only environment that never imports FastAPI, use `uv sync --no-group dev`. These are dependency *groups* rather than extras on purpose: groups install standalone, so a Termux-style install never re-resolves the `pkg`-managed packages from PyPI.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | dashboard |
| GET | `/video_feed` | multipart MJPEG stream |
| GET | `/api/state` | fps, motion %, current faces |
| GET | `/api/events?limit=N` | recent events |
| GET | `/api/events/{id}/image` | snapshot JPEG |
| GET | `/api/people` | enrolled names |
| DELETE | `/api/people/{name}` | unenroll |

Interactive docs at `/docs`. The server binds `127.0.0.1` by default and has no authentication — set `server.host` to expose it on a LAN only where that is acceptable.

## Project structure

```
edge-surveillance/
├── config/config.example.yaml  # every setting, with defaults
├── scripts/
│   ├── download_models.py      # pinned weights + sha256, mirror fallback
│   └── termux_setup.sh         # pkg + pip split install for Android
├── src/edge_surveillance/
│   ├── config.py               # typed dataclasses, rejects unknown keys
│   ├── pipeline.py             # motion gate → detect → embed → match → record
│   ├── state.py                # FrameState + SharedState atomic publish
│   ├── camera/stream_reader.py # live sources drain to newest frame
│   ├── core/
│   │   ├── motion_detector.py  # frame diff, morphology, cooldown
│   │   ├── face_detector.py    # YuNet
│   │   ├── face_recognizer.py  # MobileFaceNet, ORT/cv2.dnn, cosine match
│   │   └── embed_photos.py     # photos → embeddings, best face per photo
│   ├── store/
│   │   ├── gallery.py          # frozen Gallery + locked GalleryStore
│   │   ├── events.py           # SQLite log, WAL
│   │   └── snapshots.py        # sanitized, timestamped JPEGs
│   ├── server/
│   │   ├── app.py              # routes, MJPEG stream
│   │   └── render.py           # overlays drawn server-side
│   ├── static/index.html       # the whole dashboard
│   └── cli/                    # edge-run, edge-enroll
├── tests/                      # no model weights needed
└── data/                       # auto-created: gallery, events.db, snapshots
```

## Testing

```bash
uv run pytest -q
uv run ruff check src scripts tests
```

Face models are stubbed, so the suite runs without the download step.

## Configuration

`config/config.yaml` mirrors `config/config.example.yaml`; unknown keys are rejected rather than ignored. Every field is optional and falls back to its default, so a partial file is valid.

| Section | Key | Default | |
|---|---|---|---|
| camera | `source` | `0` | webcam index, URL, or file path |
| | `fps` | `15` | processing rate cap for live sources |
| motion | `min_change_percent` | `1.5` | frame-difference sensitivity |
| | `cooldown_s` | `0.5` | minimum gap between motion triggers, gates the face models |
| detector | `score_threshold` | `0.8` | YuNet detection confidence |
| | `report_ttl_s` | `2.0` | how long boxes linger after the last motion frame |
| recognizer | `similarity_threshold` | `0.5` | cosine floor for a name match |
| events | `known_cooldown_s` | `30.0` | per-person alert cooldown |
| | `unknown_cooldown_s` | `60.0` | cooldown for the shared unknown slot |
| server | `host` / `port` | `127.0.0.1` / `8000` | dashboard bind address |

