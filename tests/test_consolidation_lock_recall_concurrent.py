"""Concurrency tests for the consolidation advisory lock.

We assert two properties:

(1) Lock is exclusive at the application level — while one thread is
    inside ``consolidate_project`` (holding the advisory TTL lock), a
    parallel call from another thread returns immediately with
    ``paused=True`` instead of blocking or duplicating work.

(2) Reads are unaffected — recall (``SELECT * FROM knowledge``) on the
    same DB succeeds throughout the consolidation. This is what makes
    the daemon production-safe: it never starves the hot path.

Strategy: SQLite WAL mode allows N concurrent readers + 1 writer. We
configure the daemon connection to WAL on open, so concurrent reads
from a separate connection never block.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from workers import consolidation_daemon  # noqa: E402

_MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def _bootstrap_db(path: Path) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            version TEXT PRIMARY KEY, description TEXT,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, type TEXT, content TEXT, context TEXT DEFAULT '',
            project TEXT DEFAULT 'general', tags TEXT DEFAULT '[]',
            status TEXT DEFAULT 'active', confidence REAL DEFAULT 1.0,
            created_at TEXT, last_recalled TEXT, last_confirmed TEXT,
            superseded_by INTEGER, source TEXT DEFAULT 'explicit'
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT,
            project TEXT, status TEXT, summary TEXT, log_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            knowledge_id INTEGER PRIMARY KEY,
            binary_vector BLOB, float32_vector BLOB,
            embed_model TEXT, embed_dim INTEGER, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY, type TEXT, name TEXT, content TEXT,
            properties TEXT, source TEXT, importance REAL, first_seen_at TEXT,
            last_seen_at TEXT, mention_count INTEGER, status TEXT
        );
        CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
            relation_type TEXT, weight REAL, context TEXT,
            created_at TEXT, last_reinforced_at TEXT, reinforcement_count INTEGER,
            UNIQUE(source_id, target_id, relation_type)
        );
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            knowledge_id INTEGER, node_id TEXT, role TEXT, strength REAL,
            PRIMARY KEY(knowledge_id, node_id)
        );
        """
    )
    conn.executescript((_MIGRATIONS / "025_consolidation_state.sql").read_text())
    conn.executescript((_MIGRATIONS / "023_episodes.sql").read_text())
    conn.commit()
    conn.close()


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@pytest.fixture
def db_path(tmp_path) -> Path:
    p = tmp_path / "concurrent.db"
    _bootstrap_db(p)
    # Seed some facts so phases have material.
    conn = _open(p)
    base = datetime.now(timezone.utc)
    for i in range(20):
        conn.execute(
            """INSERT INTO knowledge (session_id, type, content, project, created_at, status, confidence)
               VALUES (?, 'fact', ?, 'alpha', ?, 'active', 1.0)""",
            (f"sess_{i}", f"alpha content #{i}", (base - timedelta(minutes=60+i)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    conn.commit()
    conn.close()
    return p


def test_second_consolidate_returns_paused_when_lock_held(db_path):
    """While thread A is consolidating, thread B's call returns paused=True."""
    barrier_in = threading.Event()
    barrier_release = threading.Event()
    results: dict = {}

    def slow_pause():
        # Signal we entered the locked region, then block until released.
        barrier_in.set()
        barrier_release.wait(timeout=5.0)
        return False

    def thread_a():
        conn = _open(db_path)
        try:
            stats = consolidation_daemon.consolidate_project(
                conn, "alpha", budget_seconds=30, pause_check=slow_pause,
            )
            results["a"] = stats
        finally:
            conn.close()

    def thread_b():
        # Wait for A to enter its locked phase.
        assert barrier_in.wait(timeout=5.0), "thread A never entered"
        conn = _open(db_path)
        try:
            stats = consolidation_daemon.consolidate_project(
                conn, "alpha", budget_seconds=10,
            )
            results["b"] = stats
        finally:
            conn.close()
        # Now let A finish.
        barrier_release.set()

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join(timeout=15)
    tb.join(timeout=15)
    assert not ta.is_alive() and not tb.is_alive()

    assert "a" in results and "b" in results
    # B was rejected by the lock → paused
    assert results["b"].paused is True
    # A finished cleanly
    assert results["a"].error is None


def test_recall_reads_succeed_during_consolidation(db_path):
    """A reader on a separate connection sees rows throughout consolidation."""
    read_results: list[int] = []
    reader_done = threading.Event()
    daemon_started = threading.Event()
    daemon_can_finish = threading.Event()

    def slow_pause():
        daemon_started.set()
        # Wait briefly so the reader has a window to fire SELECTs.
        daemon_can_finish.wait(timeout=5.0)
        return False

    def daemon_thread():
        conn = _open(db_path)
        try:
            consolidation_daemon.consolidate_project(
                conn, "alpha", budget_seconds=30, pause_check=slow_pause,
            )
        finally:
            conn.close()

    def reader_thread():
        conn = _open(db_path)
        try:
            assert daemon_started.wait(timeout=5.0)
            # Hammer a SELECT for ~0.5s without ever blocking.
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                rows = conn.execute(
                    "SELECT id FROM knowledge WHERE project = 'alpha' LIMIT 100"
                ).fetchall()
                read_results.append(len(rows))
            reader_done.set()
        finally:
            conn.close()
        daemon_can_finish.set()

    td = threading.Thread(target=daemon_thread)
    tr = threading.Thread(target=reader_thread)
    td.start()
    tr.start()
    td.join(timeout=15)
    tr.join(timeout=15)
    assert reader_done.is_set(), "reader did not complete"
    assert all(n == 20 for n in read_results), f"reads saw inconsistent rowcounts: {read_results}"
    # Lots of reads happened — the reader was not blocked.
    assert len(read_results) > 5


def test_lock_released_after_failure(db_path):
    """A failure inside consolidate_project still clears the lock."""
    def boom():
        raise RuntimeError("synthetic failure")

    conn = _open(db_path)
    try:
        stats = consolidation_daemon.consolidate_project(
            conn, "alpha", budget_seconds=10, pause_check=boom,
        )
        assert stats.error is not None
        row = conn.execute(
            "SELECT locked_until, last_status FROM consolidation_state WHERE project = ?",
            ("alpha",),
        ).fetchone()
        assert row["locked_until"] is None
        assert row["last_status"] == "failed"
    finally:
        conn.close()


def test_stale_lock_is_reclaimable(db_path):
    """A lock with TTL in the past is treated as available."""
    conn = _open(db_path)
    try:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        conn.execute(
            """
            INSERT INTO consolidation_state (project, locked_until, last_status)
            VALUES (?, ?, 'in_progress')
            """,
            ("alpha", past),
        )
        conn.commit()
        stats = consolidation_daemon.consolidate_project(
            conn, "alpha", budget_seconds=5,
        )
        # Should have run normally (not paused) since prior lock was stale.
        assert stats.paused is False
        assert stats.error is None
    finally:
        conn.close()
