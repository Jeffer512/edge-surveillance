import pickle

import numpy as np
import pytest

from edge_surveillance.store.gallery import Gallery, GalleryStore


def _vec(*vals):
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    return GalleryStore(tmp_path / "gallery.pkl")


def test_empty_store_is_empty(store):
    assert store.list_people() == []
    assert store.current.embeddings.shape == (0, 0)
    assert store.match(_vec(1, 0), threshold=0.5) == ("Unknown", 0.0)


def test_add_persists_and_matches(store, tmp_path):
    store.add("ann", [_vec(1, 0)])
    assert store.list_people() == ["ann"]
    assert (tmp_path / "gallery.pkl").is_file()
    assert store.match(_vec(1, 0), threshold=0.5)[0] == "ann"
    assert store.match(_vec(0, 1), threshold=0.99)[0] == "Unknown"
    assert GalleryStore(tmp_path / "gallery.pkl").list_people() == ["ann"]


def test_multi_photo_average_is_unit_length(store):
    store.add("bob", [_vec(1, 0), _vec(0, 1)])
    assert store.current.embeddings.shape == (1, 2)
    assert float(np.linalg.norm(store.current.embeddings[0])) == pytest.approx(1.0)


def test_dim_mismatch_raises(store):
    store.add("ann", [_vec(1, 0)])
    with pytest.raises(ValueError):
        store.add("bob", [_vec(1, 0, 0)])


def test_remove_drops_person_and_persists(store, tmp_path):
    store.add("ann", [_vec(1, 0)])
    store.add("bob", [_vec(0, 1)])
    store.remove("ann")
    assert store.list_people() == ["bob"]
    assert GalleryStore(tmp_path / "gallery.pkl").list_people() == ["bob"]
    with pytest.raises(KeyError):
        store.remove("ann")


def test_embeddings_are_read_only(store):
    store.add("ann", [_vec(1, 0)])
    with pytest.raises(ValueError):
        store.current.embeddings[0, 0] = 9.0
    with pytest.raises(ValueError):
        store.current.embeddings[0][0] = 9.0


def test_save_leaves_no_tmp_file(store, tmp_path):
    store.add("ann", [_vec(1, 0)])
    assert not (tmp_path / "gallery.pkl.tmp").exists()


def test_corrupt_gallery_raises_on_load(tmp_path):
    bad = tmp_path / "gallery.pkl"
    bad.write_bytes(pickle.dumps({"names": ["a", "b"], "embeddings": np.zeros((1, 2))}))
    with pytest.raises(ValueError):
        GalleryStore(bad)


def test_refresh_picks_up_external_write(store, tmp_path):
    store.add("ann", [_vec(1, 0)])
    assert store.refresh() is False

    other = GalleryStore(tmp_path / "gallery.pkl")
    other.add("bob", [_vec(0, 1)])
    assert store.refresh() is True
    assert store.list_people() == ["ann", "bob"]
    assert store.refresh() is False


def test_concurrent_removes_do_not_lose_updates(tmp_path):
    """The server deletes people while the pipeline refreshes; without the
    store's lock an interleaved read-build-save drops one of the two."""
    import threading

    path = tmp_path / "gallery.pkl"
    store = GalleryStore(path)
    for i in range(20):
        store.add(f"p{i}", [_vec(float(i), 1.0)])

    def drop(offset: int) -> None:
        for i in range(offset, 20, 2):
            try:
                store.remove(f"p{i}")
            except KeyError:
                pass

    threads = [threading.Thread(target=drop, args=(o,)) for o in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.list_people() == []


def test_gallery_equality_is_identity(store):
    store.add("ann", [_vec(1, 0)])
    assert store.current == store.current
    assert Gallery(names=("ann",)) != Gallery(names=("bob",))
