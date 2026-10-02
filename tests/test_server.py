import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge_surveillance.config import ServerConfig
from edge_surveillance.server.app import create_app
from edge_surveillance.state import FaceReport, FrameState, SharedState
from edge_surveillance.store.events import EventStore
from edge_surveillance.store.gallery import GalleryStore


def _vec(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def ctx(tmp_path):
    state = SharedState()
    gallery = GalleryStore(tmp_path / "gallery.pkl")
    gallery.add("ann", [_vec(1.0, 0.0)])
    gallery.add("bob", [_vec(0.0, 1.0)])
    events = EventStore(tmp_path / "events.db")
    app = create_app(state, gallery, events, ServerConfig())
    yield TestClient(app), state, gallery, events
    events.close()


def _publish(state, faces=(), captured_at=None):
    state.update(
        FrameState(
            captured_at=captured_at if captured_at is not None else time.time(),
            frame=np.zeros((48, 64, 3), dtype=np.uint8),
            faces=tuple(faces),
            motion_percent=2.5,
            fps=12.5,
        )
    )


def test_index_serves_the_dashboard(ctx):
    client, *_ = ctx
    r = client.get("/")
    assert r.status_code == 200
    assert "Edge Surveillance" in r.text
    assert "/video_feed" in r.text


def test_state_reports_disconnected_before_any_frame(ctx):
    client, *_ = ctx
    body = client.get("/api/state").json()
    assert body["connected"] is False
    assert body["captured_at"] is None
    assert body["faces"] == []


def test_state_reports_faces_and_metrics(ctx):
    client, state, *_ = ctx
    _publish(state, [FaceReport(name="ann", score=0.87, x=1, y=2, w=3, h=4)])
    body = client.get("/api/state").json()
    assert body["connected"] is True
    assert body["fps"] == pytest.approx(12.5)
    assert body["motion_percent"] == pytest.approx(2.5)
    assert body["faces"] == [{"name": "ann", "score": 0.87, "x": 1, "y": 2, "w": 3, "h": 4}]
    assert 0 <= body["age_s"] < 5


def test_state_goes_stale_when_the_pipeline_stops_publishing(ctx):
    client, state, *_ = ctx
    _publish(state, captured_at=time.time() - 60)
    assert client.get("/api/state").json()["connected"] is False


def test_people_listing(ctx):
    client, *_ = ctx
    assert sorted(client.get("/api/people").json()) == ["ann", "bob"]


def test_delete_person_removes_from_the_gallery(ctx):
    client, _, gallery, _ = ctx
    assert client.delete("/api/people/ann").status_code == 204
    assert gallery.list_people() == ["bob"]
    assert client.get("/api/people").json() == ["bob"]


def test_delete_unknown_person_is_404(ctx):
    client, _, gallery, _ = ctx
    r = client.delete("/api/people/nobody")
    assert r.status_code == 404
    assert sorted(gallery.list_people()) == ["ann", "bob"]


def test_events_listing_and_limit(ctx):
    client, _, _, events = ctx
    for i in range(5):
        events.add(ts=float(i), event_type="known", name="ann", score=0.9, path=None)
    body = client.get("/api/events", params={"limit": 2}).json()
    assert len(body) == 2
    assert [e["id"] for e in body] == [5, 4]


def test_events_rejects_out_of_range_limit(ctx):
    client, *_ = ctx
    assert client.get("/api/events", params={"limit": 0}).status_code == 422
    assert client.get("/api/events", params={"limit": 9999}).status_code == 422


def test_event_image_roundtrip(ctx, tmp_path):
    import cv2

    from edge_surveillance.store.snapshots import save_snapshot, snapshot_path

    client, _, _, events = ctx
    path = save_snapshot(
        np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8),
        snapshot_path(tmp_path / "ev", "ann", time.time()),
    )
    event_id = events.add(ts=time.time(), event_type="known", name="ann", score=0.9, path=str(path))
    r = client.get(f"/api/events/{event_id}/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) is not None


def test_event_image_404_when_missing_or_without_file(ctx, tmp_path):
    client, _, _, events = ctx
    without_file = events.add(
        ts=time.time(), event_type="known", name="ann", score=0.9, path=str(tmp_path / "gone.jpg")
    )
    no_path = events.add(ts=time.time(), event_type="known", name="ann", score=0.9, path=None)
    assert client.get(f"/api/events/{without_file}/image").status_code == 404
    assert client.get(f"/api/events/{no_path}/image").status_code == 404
    assert client.get("/api/events/99999/image").status_code == 404


def test_video_feed_route_is_registered(tmp_path):
    from edge_surveillance.server.app import router

    paths = {r.path for r in router.routes}
    assert {
        "/",
        "/video_feed",
        "/api/state",
        "/api/events",
        "/api/events/{event_id}/image",
        "/api/people",
        "/api/people/{name}",
    } <= paths

    app = create_app(
        SharedState(),
        GalleryStore(tmp_path / "gallery.pkl"),
        EventStore(tmp_path / "events.db"),
        ServerConfig(),
    )
    included = [getattr(r, "original_router", None) for r in app.routes]
    assert router in included


def _stream_deps(tmp_path, state):
    from edge_surveillance.server.app import Deps

    gallery = GalleryStore(tmp_path / "gallery.pkl")
    events = EventStore(tmp_path / "events.db")
    return Deps(state=state, gallery=gallery, events=events, config=ServerConfig())


def test_frame_stream_yields_one_frame_per_new_capture(tmp_path):
    """Re-encoding an unchanged frame would burn CPU for no new information."""
    import threading

    from edge_surveillance.server.app import _frame_stream

    state = SharedState()
    gen = _frame_stream(_stream_deps(tmp_path, state))
    out: list[bytes] = []

    def pump() -> None:
        while len(out) < 2:
            out.append(next(gen))

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()

    _publish(state, captured_at=1000.0)
    thread.join(timeout=2)
    assert len(out) == 1 and b"--frame" in out[0]

    # Same captured_at: nothing new, so the generator must keep waiting
    # instead of re-encoding the same image.
    _publish(state, captured_at=1000.0)
    time.sleep(0.3)
    assert len(out) == 1, "unchanged frame was re-sent"

    # A genuinely new frame is sent, and the same generator resumes.
    _publish(state, captured_at=2000.0)
    thread.join(timeout=2)
    assert len(out) == 2 and b"--frame" in out[1]


def test_frame_stream_waits_when_nothing_published(tmp_path):
    import threading

    from edge_surveillance.server.app import _frame_stream

    gen = _frame_stream(_stream_deps(tmp_path, SharedState()))
    yielded: list[bytes] = []
    waiter = threading.Thread(target=lambda: yielded.append(next(gen)), daemon=True)
    waiter.start()
    waiter.join(timeout=0.2)
    assert waiter.is_alive()
    assert yielded == []


def test_event_image_route_is_wired(ctx):
    client, *_ = ctx
    assert client.get("/api/events/1/image").status_code == 404


def test_unknown_api_path_is_404_not_the_dashboard(ctx):
    """Guards against an SPA fallback that would mask API typos with 200+HTML."""
    client, *_ = ctx
    r = client.get("/api/typo")
    assert r.status_code == 404
    assert "Edge Surveillance" not in r.text