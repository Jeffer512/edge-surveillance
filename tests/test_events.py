import cv2
import numpy as np
import pytest

from edge_surveillance.store.events import EventStore


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "events.db")
    yield s
    s.close()


def test_add_returns_increasing_ids_and_recent_is_newest_first(store):
    a = store.add(ts=100.0, event_type="known", name="ann", score=0.9, path="a.jpg")
    b = store.add(ts=200.0, event_type="unknown", name=None, score=0.2, path="b.jpg")
    assert (a, b) == (1, 2)
    recent = store.recent()
    assert [e["id"] for e in recent] == [2, 1]
    assert recent[0]["name"] is None
    assert recent[0]["type"] == "unknown"
    assert recent[1]["name"] == "ann"
    assert recent[1]["type"] == "known"


def test_recent_respects_limit_and_repeats_same_timestamp(store):
    for _ in range(5):
        store.add(ts=100.0, event_type="known", name="ann", score=0.9, path="a.jpg")
    assert len(store.recent(limit=2)) == 2
    assert [e["id"] for e in store.recent(limit=2)] == [5, 4]


def test_set_path_marks_missing_snapshot(store):
    eid = store.add(ts=1.0, event_type="known", name="ann", score=0.9, path="a.jpg")
    store.set_path(eid, None)
    assert store.recent()[0]["path"] is None


def test_count_since(store):
    store.add(ts=100.0, event_type="known", name="ann", score=0.9, path="a.jpg")
    store.add(ts=500.0, event_type="unknown", name=None, score=0.1, path="b.jpg")
    assert store.count_since(200.0) == 1
    assert store.count_since(0.0) == 2


def test_data_survives_reopen(tmp_path):
    path = tmp_path / "events.db"
    s = EventStore(path)
    s.add(ts=1.0, event_type="known", name="ann", score=0.9, path="a.jpg")
    s.close()
    assert EventStore(path).recent()[0]["name"] == "ann"


def test_concurrent_writes_do_not_lose_rows(tmp_path):
    """Without the store's lock, interleaved commits silently drop rows."""
    store = EventStore(tmp_path / "events.db")
    import threading

    def worker() -> None:
        for i in range(150):
            store.add(ts=float(i), event_type="known", name="ann", score=0.9, path="a.jpg")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count_since(0.0) == 600
    store.close()


def test_concurrent_read_during_writes(tmp_path):
    store = EventStore(tmp_path / "events.db")
    import threading

    stop = threading.Event()
    errors: list[Exception] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                store.recent(limit=10)
        except Exception as e:  # noqa: BLE001 - surfaced via assertion below
            errors.append(e)

    def writer() -> None:
        for i in range(150):
            store.add(ts=float(i), event_type="known", name="ann", score=0.9, path="a.jpg")

    r = threading.Thread(target=reader)
    w = threading.Thread(target=writer)
    r.start()
    w.start()
    w.join()
    stop.set()
    r.join()
    assert errors == []
    assert store.count_since(0.0) == 150
    store.close()


def test_snapshot_images_are_readable(tmp_path):
    """A stored path should be loadable by the UI that serves it."""
    from edge_surveillance.store.snapshots import save_snapshot, snapshot_path

    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    path = save_snapshot(frame, snapshot_path(tmp_path / "ev", "ann", 1759000000.5))
    assert cv2.imread(str(path)) is not None
