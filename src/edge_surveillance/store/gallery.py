import os
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from edge_surveillance.core.face_recognizer import cosine_match, l2_normalize


@dataclass(frozen=True, eq=False)
class Gallery:
    """Enrolled identities: names[i] matches embeddings row i.

    eq=False because the default dataclass __eq__ raises ValueError when
    comparing two different galleries: ndarray equality returns an array,
    not a bool. embeddings is made read-only in __post_init__ so a
    reader cannot mutate the snapshot it is holding.
    """

    names: tuple[str, ...] = ()
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32),
        repr=False,
    )

    def __post_init__(self) -> None:
        self.embeddings.setflags(write=False)


def _save(gallery: Gallery, path: Path) -> None:
    """Atomic write (tmp + rename) so crashes never leave a half-pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"names": list(gallery.names), "embeddings": gallery.embeddings}, f)
    os.replace(tmp, path)


def _load(path: Path) -> Gallery:
    with open(path, "rb") as f:
        data = pickle.load(f)
    names = tuple(data["names"])
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    if len(names) != len(embeddings):
        raise ValueError(f"gallery corrupt: {len(names)} names vs {len(embeddings)} rows")
    if embeddings.ndim != 2:
        raise ValueError(f"gallery embeddings must be 2D, got shape {embeddings.shape}")
    return Gallery(names=names, embeddings=embeddings)


class GalleryStore:
    """Owner of the enrolled gallery for a long-lived process.

    Every mutation goes through _commit, which writes to disk before swapping
    the in-memory value, so a crash can never leave memory ahead of disk.
    Out-of-process writes (the enroll CLI) are picked up by refresh().

    Mutators take a lock because two threads can now write: the pipeline
    enrolling or expiring, and the web server deleting a person. Without it
    an interleaved read-build-save would silently drop one of the two.
    Reads stay lock-free: a single attribute read is already atomic.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._current = _load(self._path) if self._path.is_file() else Gallery()
        self._mtime = self._stat()

    @property
    def current(self) -> Gallery:
        return self._current

    def _stat(self) -> tuple:
        try:
            st = self._path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return (-1.0, -1)

    def _commit(self, new: Gallery) -> Gallery:
        _save(new, self._path)
        self._current = new
        self._mtime = self._stat()
        return new

    def refresh(self) -> bool:
        """Reload from disk if it changed outside this process."""
        with self._lock:
            mtime = self._stat()
            if mtime != self._mtime:
                self._current = _load(self._path) if self._path.is_file() else Gallery()
                self._mtime = mtime
                return True
            return False

    def add(self, name: str, embeddings: list[np.ndarray]) -> None:
        """Average embeddings into a single row for one person."""
        mean = l2_normalize(np.mean(np.stack(embeddings), axis=0))
        with self._lock:
            cur = self._current
            if cur.embeddings.size and mean.size != cur.embeddings.shape[1]:
                raise ValueError(
                    f"embedding dim {mean.size} != gallery dim {cur.embeddings.shape[1]}"
                )
            row = mean.reshape(1, -1).astype(np.float32)
            stacked = row if not cur.embeddings.size else np.vstack([cur.embeddings, row])
            self._commit(Gallery(names=(*cur.names, name), embeddings=stacked.astype(np.float32)))

    def remove(self, name: str) -> None:
        """Drop a person. KeyError when not enrolled."""
        with self._lock:
            cur = self._current
            if name not in cur.names:
                raise KeyError(f"unknown person: {name!r}")
            keep = [i for i, n in enumerate(cur.names) if n != name]
            self._commit(
                Gallery(
                    names=tuple(cur.names[i] for i in keep),
                    embeddings=cur.embeddings[keep].astype(np.float32),
                )
            )

    def match(self, query: np.ndarray, threshold: float) -> tuple[str, float]:
        return cosine_match(query, self._current.embeddings, list(self._current.names), threshold)

    def list_people(self) -> list[str]:
        return list(self._current.names)
