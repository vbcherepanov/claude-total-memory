"""Tests for the consolidation daemon's project selection logic.

We don't need a real DB schema for these — we just stub the picker
inputs (project_activity) and assert the daemon picks the OLDEST idle
project, skips the active one, respects the cooldown, and honors the
exclude list.
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

from workers import project_activity, consolidation_daemon  # noqa: E402

_MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def _open_test_db(path: Path) -> sqlite3.Connection:
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
    return conn


@pytest.fixture
def db(tmp_path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    conn = _open_test_db(db_path)
    yield conn
    conn.close()


def _now() -> datetime:
    return datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)


# ─── tests on selection logic ──────────────────────────────────────────


def test_picks_oldest_idle_project(db):
    base = _now()
    project_activity.touch(db, "alpha", when=base - timedelta(hours=2))
    project_activity.touch(db, "beta", when=base - timedelta(hours=5))
    project_activity.touch(db, "gamma", when=base - timedelta(hours=3))
    idle = project_activity.list_idle_projects(db, idle_seconds=1800, now=base)
    assert idle[0] == "beta"


def test_skips_active_project_via_exclude(db):
    base = _now()
    project_activity.touch(db, "alpha", when=base - timedelta(hours=10))
    project_activity.touch(db, "beta", when=base - timedelta(seconds=120))
    active = project_activity.get_active_project(db, now=base)
    assert active == "beta"
    idle = project_activity.list_idle_projects(
        db, idle_seconds=1800, exclude=[active], now=base,
    )
    assert "beta" not in idle
    assert "alpha" in idle


def test_respects_cooldown(db):
    base = _now()
    project_activity.touch(db, "alpha", when=base - timedelta(hours=10))
    db.execute(
        "INSERT INTO consolidation_state (project, last_consolidated_at, last_status) VALUES (?, ?, 'ok')",
        ("alpha", (base - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")),
    )
    db.commit()
    idle = project_activity.list_idle_projects(
        db, idle_seconds=1800, consolidate_cooldown_seconds=6 * 3600, now=base,
    )
    assert "alpha" not in idle


def test_exclude_list_drops_multiple(db):
    base = _now()
    for p in ("a", "b", "c"):
        project_activity.touch(db, p, when=base - timedelta(hours=3))
    idle = project_activity.list_idle_projects(
        db, idle_seconds=1800, exclude=["a", "c"], now=base,
    )
    assert idle == ["b"]


def test_consolidate_project_with_empty_db_returns_clean_stats(db):
    stats = consolidation_daemon.consolidate_project(
        db, "empty-project", budget_seconds=10,
    )
    assert stats.error is None
    assert stats.transitive_facts_added == 0
    assert stats.episodes_materialized == 0
    assert stats.duplicates_merged == 0
    # All five phases ran (none were paused) → all five recorded.
    assert "transitive" in stats.phases
    assert "decay" in stats.phases
    assert stats.paused is False


def test_consolidate_project_releases_lock_on_success(db):
    consolidation_daemon.consolidate_project(db, "alpha", budget_seconds=10)
    row = db.execute(
        "SELECT locked_until, last_status FROM consolidation_state WHERE project = ?",
        ("alpha",),
    ).fetchone()
    assert row is not None
    assert row["locked_until"] is None
    assert row["last_status"] == "ok"


def test_consolidate_project_pause_check_aborts_early(db):
    base = _now()
    # Write some real fact rows so phases have something to scan.
    for i in range(5):
        db.execute(
            """INSERT INTO knowledge (session_id, type, content, project, created_at, status, confidence)
               VALUES (?, 'fact', ?, 'alpha', ?, 'active', 1.0)""",
            (f"s{i}", f"content {i}", _iso_min_ago(base, 60 + i)),
        )
    db.commit()

    pause_called = {"n": 0}

    def always_pause():
        pause_called["n"] += 1
        return True

    stats = consolidation_daemon.consolidate_project(
        db, "alpha", budget_seconds=10, pause_check=always_pause,
    )
    assert stats.paused is True
    assert pause_called["n"] >= 1
    # Lock cleared, status reflects paused.
    row = db.execute(
        "SELECT locked_until, last_status FROM consolidation_state WHERE project = ?",
        ("alpha",),
    ).fetchone()
    assert row["locked_until"] is None
    assert row["last_status"] == "paused"


def test_run_daemon_loop_picks_idle_then_stops(db, tmp_path):
    """Smoke-test the loop: insert one idle project, run for a tick, stop."""
    real_now = datetime.now(timezone.utc)
    project_activity.touch(db, "alpha", when=real_now - timedelta(hours=2))
    db.commit()
    db.close()

    db_path = str(tmp_path / "test.db")
    stop = threading.Event()
    runs: list = []

    def on_run(stats):
        runs.append(stats)
        stop.set()

    t = threading.Thread(
        target=consolidation_daemon.run_daemon,
        kwargs={
            "db_path": db_path,
            "poll_interval_seconds": 1,
            "idle_seconds": 1800,
            "consolidate_cooldown_seconds": 21600,
            "budget_seconds": 5,
            "stop_event": stop,
            "on_run": on_run,
        },
        daemon=True,
    )
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "daemon thread did not exit"
    assert len(runs) >= 1
    assert runs[0].project == "alpha"


def test_run_daemon_skips_when_no_candidates(db, tmp_path):
    db.close()
    db_path = str(tmp_path / "test.db")
    stop = threading.Event()

    runs: list = []

    def on_run(stats):
        runs.append(stats)

    t = threading.Thread(
        target=consolidation_daemon.run_daemon,
        kwargs={
            "db_path": db_path,
            "poll_interval_seconds": 1,
            "idle_seconds": 1800,
            "stop_event": stop,
            "on_run": on_run,
        },
        daemon=True,
    )
    t.start()
    time.sleep(2.0)
    stop.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert runs == []


# ─── helpers ───────────────────────────────────────────────────────────


def _iso_min_ago(base: datetime, minutes: int) -> str:
    when = base - timedelta(minutes=minutes)
    return when.strftime("%Y-%m-%dT%H:%M:%S.000Z")
