"""v11 W2-G — Project activity tracker.

Tracks "which project is the user touching right now" so the
consolidation daemon can:

  * skip the active project entirely (never consolidate while in use),
  * pick the OLDEST idle project as its next consolidation target.

The data lives in the ``project_activity`` table created by migration
025. All timestamps are ISO-8601 UTC with millisecond precision so they
sort lexicographically and compare correctly against
``strftime('%Y-%m-%dT%H:%M:%fZ','now')``.

Public API
----------

* :func:`touch` — record a hit on a project (called from hot path).
* :func:`get_active_project` — most recently touched project within
  ``threshold_seconds``, else ``None``.
* :func:`list_idle_projects` — projects last touched longer ago than
  ``idle_seconds`` AND not consolidated in the last
  ``consolidate_cooldown_seconds``, oldest-touched first.
* :func:`is_active` — predicate form of :func:`get_active_project`
  scoped to one project. Used as the daemon's pause check.

The module never imports from ``ai_layer.*`` or pulls heavy dependencies
— this is hot-path code.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable

# ─── time helpers ──────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    """Format a datetime as ISO-8601 UTC with millisecond precision."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    else:
        when = when.astimezone(timezone.utc)
    # Mirror SQLite's strftime('%Y-%m-%dT%H:%M:%fZ','now') for lexicographic match.
    return when.strftime("%Y-%m-%dT%H:%M:%S.") + f"{when.microsecond // 1000:03d}Z"


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# ─── public API ────────────────────────────────────────────────────────


def touch(
    conn: sqlite3.Connection,
    project: str,
    *,
    when: datetime | None = None,
) -> None:
    """Record a hit on ``project``.

    Updates ``last_touched_at`` to ``when`` (default: now UTC) and
    increments ``touch_count_24h`` if the previous touch was within the
    last 24 hours, else resets it to 1.
    """
    if not project:
        raise ValueError("touch: project is required")
    now = when or _utcnow()
    now_iso = _iso(now)
    cutoff_iso = _iso(now - timedelta(hours=24))

    # UPSERT keeping a rolling 24h count.
    conn.execute(
        """
        INSERT INTO project_activity (project, last_touched_at, touch_count_24h, updated_at)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(project) DO UPDATE SET
            last_touched_at = excluded.last_touched_at,
            touch_count_24h = CASE
                WHEN project_activity.last_touched_at >= ?
                THEN project_activity.touch_count_24h + 1
                ELSE 1
            END,
            updated_at = excluded.updated_at
        """,
        (project, now_iso, now_iso, cutoff_iso),
    )
    conn.commit()


def get_active_project(
    conn: sqlite3.Connection,
    *,
    threshold_seconds: int = 300,
    now: datetime | None = None,
) -> str | None:
    """Return the project most recently touched within ``threshold_seconds``.

    "Active" is defined narrowly — within the last 5 minutes by default.
    Used by the daemon to decide which project to AVOID this round.
    """
    cutoff_iso = _iso((now or _utcnow()) - timedelta(seconds=threshold_seconds))
    row = conn.execute(
        """
        SELECT project, last_touched_at FROM project_activity
        WHERE last_touched_at >= ?
        ORDER BY last_touched_at DESC, project ASC
        LIMIT 1
        """,
        (cutoff_iso,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def list_idle_projects(
    conn: sqlite3.Connection,
    *,
    idle_seconds: int = 1800,
    consolidate_cooldown_seconds: int = 21600,
    exclude: Iterable[str] = (),
    now: datetime | None = None,
) -> list[str]:
    """Return idle, not-recently-consolidated projects.

    Filters:
      * ``last_touched_at`` older than ``idle_seconds`` (default 30 min).
      * ``last_consolidated_at`` either NULL or older than
        ``consolidate_cooldown_seconds`` (default 6 h).
      * NOT in ``exclude`` (typically the currently-active project).

    Ordered oldest-touched first so the daemon catches up on the most
    stale project before younger ones.
    """
    when = now or _utcnow()
    idle_cutoff_iso = _iso(when - timedelta(seconds=idle_seconds))
    cooldown_cutoff_iso = _iso(when - timedelta(seconds=consolidate_cooldown_seconds))

    rows = conn.execute(
        """
        SELECT pa.project, pa.last_touched_at
        FROM project_activity pa
        LEFT JOIN consolidation_state cs ON cs.project = pa.project
        WHERE pa.last_touched_at < ?
          AND (cs.last_consolidated_at IS NULL OR cs.last_consolidated_at < ?)
        ORDER BY pa.last_touched_at ASC, pa.project ASC
        """,
        (idle_cutoff_iso, cooldown_cutoff_iso),
    ).fetchall()

    excluded = {p for p in (exclude or ()) if p}
    return [str(r[0]) for r in rows if str(r[0]) not in excluded]


def is_active(
    conn: sqlite3.Connection,
    project: str,
    *,
    threshold_seconds: int = 300,
    now: datetime | None = None,
) -> bool:
    """Predicate: was ``project`` touched within ``threshold_seconds``?"""
    if not project:
        return False
    cutoff_iso = _iso((now or _utcnow()) - timedelta(seconds=threshold_seconds))
    row = conn.execute(
        "SELECT 1 FROM project_activity WHERE project = ? AND last_touched_at >= ? LIMIT 1",
        (project, cutoff_iso),
    ).fetchone()
    return row is not None


__all__ = [
    "touch",
    "get_active_project",
    "list_idle_projects",
    "is_active",
]
