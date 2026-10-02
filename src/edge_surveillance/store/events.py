import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    name TEXT,
    score REAL,
    path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class EventStore:
    """SQLite event log, safe to write from the pipeline and read from the server.

    check_same_thread=False lifts the cross-thread guard but adds no serialization.
    The lock keeps each execute+commit atomic. WAL keeps a reader from blocking the writer.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add(self, ts: float, event_type: str, name: str | None, score: float, path: str) -> int:
        """Insert an event row and return its id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, type, name, score, path) VALUES (?, ?, ?, ?, ?)",
                (ts, event_type, name, score, path),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_path(self, event_id: int, path: str | None) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET path = ? WHERE id = ?", (path, event_id))
            self._conn.commit()

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, type, name, score, path FROM events"
                " ORDER BY ts DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("id", "ts", "type", "name", "score", "path")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def get(self, event_id: int) -> dict | None:
        keys = ("id", "ts", "type", "name", "score", "path")
        with self._lock:
            row = self._conn.execute(
                "SELECT id, ts, type, name, score, path FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return dict(zip(keys, row, strict=True)) if row else None

    def count_since(self, ts: float) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE ts >= ?", (ts,)
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
