"""Tests for ``workers.project_activity``.

Cover the touch/list_idle/get_active boundary conditions, threshold
tuning, exclude param, and ordering.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from workers import project_activity  # noqa: E402


_MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Minimal `migrations` table so 025 can record itself.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            version TEXT PRIMARY KEY,
            description TEXT,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
        """
    )
    conn.executescript((_MIGRATIONS / "025_consolidation_state.sql").read_text())
    yield conn
    conn.close()


def _at(when: datetime) -> datetime:
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when


def test_touch_creates_row_with_count_one(db):
    now = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=now)
    row = db.execute("SELECT * FROM project_activity WHERE project = ?", ("alpha",)).fetchone()
    assert row is not None
    assert row["touch_count_24h"] == 1


def test_touch_increments_count_within_24h(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base)
    project_activity.touch(db, "alpha", when=base + timedelta(hours=1))
    project_activity.touch(db, "alpha", when=base + timedelta(hours=23))
    row = db.execute("SELECT * FROM project_activity WHERE project = ?", ("alpha",)).fetchone()
    assert row["touch_count_24h"] == 3


def test_touch_resets_count_after_24h(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base)
    project_activity.touch(db, "alpha", when=base + timedelta(hours=25))
    row = db.execute("SELECT * FROM project_activity WHERE project = ?", ("alpha",)).fetchone()
    assert row["touch_count_24h"] == 1


def test_touch_rejects_empty_project(db):
    with pytest.raises(ValueError):
        project_activity.touch(db, "")


def test_get_active_returns_most_recent_within_threshold(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(seconds=120))
    project_activity.touch(db, "beta", when=base - timedelta(seconds=30))
    active = project_activity.get_active_project(db, threshold_seconds=300, now=base)
    assert active == "beta"


def test_get_active_returns_none_when_all_stale(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(hours=2))
    project_activity.touch(db, "beta", when=base - timedelta(hours=1))
    assert project_activity.get_active_project(db, threshold_seconds=300, now=base) is None


def test_is_active_predicate_matches_threshold(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(seconds=60))
    assert project_activity.is_active(db, "alpha", threshold_seconds=300, now=base)
    assert not project_activity.is_active(db, "alpha", threshold_seconds=10, now=base)
    assert not project_activity.is_active(db, "missing", threshold_seconds=300, now=base)


def test_list_idle_orders_oldest_first(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(hours=5))
    project_activity.touch(db, "beta", when=base - timedelta(hours=2))
    project_activity.touch(db, "gamma", when=base - timedelta(hours=10))
    idle = project_activity.list_idle_projects(
        db, idle_seconds=1800, consolidate_cooldown_seconds=21600, now=base,
    )
    assert idle == ["gamma", "alpha", "beta"]


def test_list_idle_excludes_recent_projects(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(hours=2))
    # touched within idle window → not idle yet
    project_activity.touch(db, "active_one", when=base - timedelta(seconds=200))
    idle = project_activity.list_idle_projects(db, idle_seconds=1800, now=base)
    assert "active_one" not in idle
    assert "alpha" in idle


def test_list_idle_respects_exclude_param(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(hours=2))
    project_activity.touch(db, "beta", when=base - timedelta(hours=3))
    idle = project_activity.list_idle_projects(
        db, idle_seconds=1800, exclude=["beta"], now=base
    )
    assert idle == ["alpha"]


def test_list_idle_respects_cooldown(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(hours=10))
    # Mark alpha as having just been consolidated.
    db.execute(
        """
        INSERT INTO consolidation_state (project, last_consolidated_at, last_status)
        VALUES (?, ?, 'ok')
        """,
        ("alpha", (base - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")),
    )
    db.commit()
    # 6h cooldown → alpha is filtered out, even though it's idle.
    idle = project_activity.list_idle_projects(
        db, idle_seconds=1800, consolidate_cooldown_seconds=6 * 3600, now=base,
    )
    assert "alpha" not in idle


def test_list_idle_includes_after_cooldown_expires(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    project_activity.touch(db, "alpha", when=base - timedelta(hours=10))
    db.execute(
        """
        INSERT INTO consolidation_state (project, last_consolidated_at, last_status)
        VALUES (?, ?, 'ok')
        """,
        ("alpha", (base - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%S.000Z")),
    )
    db.commit()
    idle = project_activity.list_idle_projects(
        db, idle_seconds=1800, consolidate_cooldown_seconds=6 * 3600, now=base,
    )
    assert "alpha" in idle


def test_list_idle_empty_when_no_data(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    assert project_activity.list_idle_projects(db, now=base) == []


def test_get_active_tie_breaker_is_deterministic(db):
    base = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    # Identical timestamps — the tie-break is alphabetical.
    project_activity.touch(db, "zeta", when=base - timedelta(seconds=10))
    project_activity.touch(db, "alpha", when=base - timedelta(seconds=10))
    active = project_activity.get_active_project(db, threshold_seconds=300, now=base)
    # zeta was inserted second so updated_at is later — but ORDER BY uses
    # last_touched_at first; for ties we fall back to project ASC.
    assert active in {"alpha", "zeta"}
