"""Chaos test: kill the consolidation daemon mid-run, restart, verify recovery.

Three properties under test:

(1) When SIGKILL'd while a consolidation is in progress, the DB is not
    corrupted (SQLite WAL transactions either commit or roll back as a
    whole — never half-applied).

(2) After kill, ``consolidation_state.locked_until`` may still hold a
    TTL in the future. That stale lock is automatically reclaimable
    once the TTL expires (or if it was already in the past).

(3) On restart, the next consolidation cleanly proceeds — no manual
    cleanup required, no leftover ``in_progress`` rows pinning the
    project forever.

Implementation note: subprocess + SIGKILL is the closest analogue to
"power was cut" we can reasonably write in unit tests. We do NOT use
SIGTERM — the daemon traps that for clean shutdown.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_MIGRATIONS = _REPO / "migrations"


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
    base = datetime.now(timezone.utc)
    for i in range(40):
        conn.execute(
            """INSERT INTO knowledge (session_id, type, content, project, created_at, status, confidence)
               VALUES (?, 'fact', ?, 'alpha', ?, 'active', 1.0)""",
            (f"sess_{i}", f"alpha content #{i}", (base - timedelta(minutes=120 + i)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
    # Seed a touch in project_activity so the daemon has work.
    conn.execute(
        """INSERT INTO project_activity (project, last_touched_at, touch_count_24h)
           VALUES ('alpha', ?, 1)""",
        ((base - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),),
    )
    conn.commit()
    conn.close()


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ─── helpers to spawn a consolidation in a subprocess ──────────────────


def _spawn_long_running_consolidation(db_path: Path) -> subprocess.Popen:
    """Spawn a Python child that calls consolidate_project with a slow pause.

    The child runs forever until killed: the pause_check sleeps in a
    busy-wait so the function holds the lock indefinitely.
    """
    code = textwrap.dedent(
        f"""
        import sys, time, sqlite3
        sys.path.insert(0, {str(_SRC)!r})
        from workers import consolidation_daemon
        conn = sqlite3.connect({str(db_path)!r}, isolation_level=None, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # Hold the lock by stalling inside pause_check (which is called
        # between phases). We never return True from pause_check so the
        # function tries to keep working — but we sleep a long time
        # between calls.
        def _slow():
            time.sleep(60)
            return False
        try:
            consolidation_daemon.consolidate_project(
                conn, "alpha", budget_seconds=300, pause_check=_slow,
            )
        finally:
            conn.close()
        """
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ─── tests ─────────────────────────────────────────────────────────────


def test_db_intact_after_sigkill_mid_run(tmp_path):
    db_path = tmp_path / "chaos.db"
    _bootstrap_db(db_path)

    proc = _spawn_long_running_consolidation(db_path)
    # Give it time to claim the lock and enter the slow pause.
    time.sleep(1.0)
    assert proc.poll() is None, "child exited prematurely"

    # SIGKILL — no clean shutdown, no signal handler runs.
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)

    # DB must still open and be query-able.
    conn = _open(db_path)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM knowledge WHERE project = 'alpha'").fetchone()
        assert rows[0] == 40
        # Lock state row exists (lock was taken before the kill).
        lock_row = conn.execute(
            "SELECT locked_until, last_status FROM consolidation_state WHERE project = 'alpha'"
        ).fetchone()
        assert lock_row is not None
    finally:
        conn.close()


def test_restart_resumes_after_kill_with_expired_lock(tmp_path):
    """Stale locks don't pin the project forever."""
    db_path = tmp_path / "chaos.db"
    _bootstrap_db(db_path)

    # Simulate a killed daemon: write a lock TTL in the past.
    conn = _open(db_path)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    conn.execute(
        """INSERT INTO consolidation_state (project, locked_until, last_status)
           VALUES ('alpha', ?, 'in_progress')""",
        (past,),
    )
    conn.commit()

    # New daemon process can claim the stale lock and consolidate.
    sys.path.insert(0, str(_SRC))
    from workers import consolidation_daemon  # noqa: WPS433

    stats = consolidation_daemon.consolidate_project(
        conn, "alpha", budget_seconds=5,
    )
    assert stats.paused is False
    assert stats.error is None
    # Lock cleared.
    row = conn.execute(
        "SELECT locked_until, last_status FROM consolidation_state WHERE project = 'alpha'"
    ).fetchone()
    assert row["locked_until"] is None
    assert row["last_status"] == "ok"
    conn.close()


def test_kill_then_immediate_restart_returns_paused_until_ttl_expires(tmp_path):
    """Right after kill, a fresh attempt sees the lock as still held.

    The lock is an *advisory TTL*, not a process lease. If the TTL is
    still in the future, a follower correctly treats the project as
    "in progress" until the TTL elapses. This is the conservative
    choice — the alternative (heartbeat-based liveness detection)
    would add complexity and a polling background fiber.
    """
    db_path = tmp_path / "chaos.db"
    _bootstrap_db(db_path)

    proc = _spawn_long_running_consolidation(db_path)
    time.sleep(1.0)
    assert proc.poll() is None
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)

    sys.path.insert(0, str(_SRC))
    from workers import consolidation_daemon  # noqa: WPS433

    conn = _open(db_path)
    try:
        # Lock TTL = budget(300s) + 30s, so a fresh attempt will be paused.
        stats = consolidation_daemon.consolidate_project(
            conn, "alpha", budget_seconds=5,
        )
        assert stats.paused is True
    finally:
        conn.close()


def test_atomic_rows_no_partial_inferred_facts(tmp_path):
    """SQLite transactions — no half-written rows after an abrupt kill."""
    db_path = tmp_path / "chaos.db"
    _bootstrap_db(db_path)

    proc = _spawn_long_running_consolidation(db_path)
    time.sleep(1.5)
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)

    conn = _open(db_path)
    try:
        # Every fact row should have a non-empty content and a project.
        bad = conn.execute(
            """SELECT COUNT(*) FROM knowledge
               WHERE content IS NULL OR content = '' OR project IS NULL"""
        ).fetchone()
        assert bad[0] == 0
        # No rows in episode_facts referencing missing episodes.
        orphans = conn.execute(
            """SELECT COUNT(*) FROM episode_facts ef
               LEFT JOIN episodes_v11 e ON e.id = ef.episode_id
               WHERE e.id IS NULL"""
        ).fetchone()
        assert orphans[0] == 0
    finally:
        conn.close()
