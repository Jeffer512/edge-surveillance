import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from edge_surveillance.config import ServerConfig
from edge_surveillance.server.render import draw_overlay
from edge_surveillance.state import SharedState
from edge_surveillance.store.events import EventStore
from edge_surveillance.store.gallery import GalleryStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 70]
FEED_IDLE_S = 0.02
STALE_AFTER_S = 3.0


@dataclass
class Deps:
    state: SharedState
    gallery: GalleryStore
    events: EventStore
    config: ServerConfig


def _get_deps(request: Request) -> Deps:
    return request.app.state.deps


DepsDep = Annotated[Deps, Depends(_get_deps)]


class FaceOut(BaseModel):
    name: str
    score: float
    x: int
    y: int
    w: int
    h: int


class StateOut(BaseModel):
    connected: bool
    fps: float
    motion_percent: float
    captured_at: float | None
    age_s: float | None
    faces: list[FaceOut]


class EventOut(BaseModel):
    id: int
    ts: float
    type: str
    name: str | None
    score: float
    path: str | None


router = APIRouter(tags=["edge-surveillance"])


@router.get("/")
def read_index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


def _frame_stream(deps: Deps):
    last_sent: float | None = None
    while True:
        state = deps.state.current
        if state is not None and state.captured_at != last_sent:
            last_sent = state.captured_at
            ok, buf = cv2.imencode(".jpg", draw_overlay(state.frame, state.faces), JPEG_PARAMS)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        else:
            time.sleep(FEED_IDLE_S)


@router.get("/video_feed")
def video_feed(deps: DepsDep) -> StreamingResponse:
    return StreamingResponse(
        _frame_stream(deps),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/state")
def read_state(deps: DepsDep) -> StateOut:
    state = deps.state.current
    if state is None:
        return StateOut(
            connected=False, fps=0.0, motion_percent=0.0, captured_at=None, age_s=None, faces=[]
        )
    age = time.time() - state.captured_at
    return StateOut(
        connected=age < STALE_AFTER_S,
        fps=state.fps,
        motion_percent=state.motion_percent,
        captured_at=state.captured_at,
        age_s=age,
        faces=[
            FaceOut(name=f.name, score=f.score, x=f.x, y=f.y, w=f.w, h=f.h) for f in state.faces
        ],
    )


@router.get("/api/events")
def read_events(deps: DepsDep, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[EventOut]:
    return [EventOut(**row) for row in deps.events.recent(limit=limit)]


@router.get("/api/events/{event_id}/image")
def read_event_image(deps: DepsDep, event_id: Annotated[int, PathParam(ge=1)]) -> FileResponse:
    row = deps.events.get(event_id)
    if row is None or not row["path"]:
        raise HTTPException(status_code=404, detail="event not found")
    path = Path(row["path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="snapshot missing")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/api/people")
def read_people(deps: DepsDep) -> list[str]:
    return deps.gallery.list_people()


@router.delete("/api/people/{name}", status_code=204)
def delete_person(deps: DepsDep, name: Annotated[str, PathParam(min_length=1)]) -> Response:
    try:
        deps.gallery.remove(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from e
    logger.info("Removed %r via API", name)
    return Response(status_code=204)


def create_app(
    state: SharedState, gallery: GalleryStore, events: EventStore, config: ServerConfig
) -> FastAPI:
    app = FastAPI(title="Edge Surveillance", version="0.1.0")
    app.state.deps = Deps(state=state, gallery=gallery, events=events, config=config)
    app.include_router(router)
    return app
