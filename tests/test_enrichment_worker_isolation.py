"""The enrichment worker must not share the Store's sqlite Connection.

`check_same_thread=False` lets a background thread use a Connection, but it
does not make concurrent *writes* safe: sqlite3's legacy isolation mode keeps
the implicit BEGIN/COMMIT as Connection state, so DML issued from two threads
interleaves into::

    sqlite3.OperationalError: cannot start a transaction within a transaction

or its mirror image, "no transaction active". WAL mode and busy_timeout fix
contention *between* connections and do nothing for this.

The failure is load-dependent, which is why it surfaced as a benchmark crash
and as flaky "no transaction active" errors rather than a test failure. These
tests pin the fix: the worker opens its own connection.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import enrichment_worker  # noqa: E402
import server  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    try:
        yield s
    finally:
        try:
            s.db.close()
        except Exception:
            pass


def test_store_exposes_its_database_path(store, tmp_path):
    """The worker needs this to open a connection of its own."""
    assert store.db_path == tmp_path / "memory.db"


def test_worker_opens_its_own_connection(store):
    worker = enrichment_worker._WorkerThread(store)
    conn, owned = worker._open_db()
    try:
        assert owned is True
        assert conn is not store.db
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_worker_falls_back_to_the_shared_connection_for_a_store_without_a_path():
    """Test doubles and pre-v13 Stores must still start, not crash."""

    class _Legacy:
        db = object()

    conn, owned = enrichment_worker._WorkerThread(_Legacy())._open_db()
    assert owned is False
    assert conn is _Legacy.db


def test_concurrent_writes_from_two_threads_do_not_corrupt_transaction_state(store):
    """Reproduces the crash: main thread writing while the worker writes.

    Against the shared Connection this raises OperationalError within a few
    hundred iterations; with a private connection per thread it completes.
    """
    worker = enrichment_worker._WorkerThread(store)
    conn, owned = worker._open_db()
    errors: list[Exception] = []
    stop = threading.Event()

    def hammer(db, tag: str) -> None:
        i = 0
        try:
            while not stop.is_set() and i < 400:
                db.execute(
                    "INSERT OR IGNORE INTO sessions (id, started_at, project) VALUES (?, ?, ?)",
                    (f"{tag}-{i}", "2026-08-27T00:00:00Z", "concurrency"),
                )
                db.commit()
                i += 1
        except Exception as e:  # noqa: BLE001 — the whole point is to catch it
            errors.append(e)
            stop.set()

    try:
        threads = [
            threading.Thread(target=hammer, args=(store.db, "main")),
            threading.Thread(target=hammer, args=(conn, "worker")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive(), "writer thread hung"

        assert not errors, f"concurrent writes failed: {errors[0]!r}"
        written = store.db.execute(
            "SELECT COUNT(*) FROM sessions WHERE project='concurrency'"
        ).fetchone()[0]
        assert written == 800
    finally:
        stop.set()
        if owned:
            conn.close()


def test_worker_thread_starts_and_stops_cleanly(store, monkeypatch):
    monkeypatch.setenv("MEMORY_ASYNC_ENRICHMENT", "true")
    monkeypatch.setenv("MEMORY_ENRICH_TICK_SEC", "0.01")
    thread = enrichment_worker.start_worker(store)
    assert thread is not None
    try:
        deadline = time.time() + 5
        while not thread.is_alive() and time.time() < deadline:
            time.sleep(0.01)
        assert thread.is_alive()
    finally:
        thread.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
