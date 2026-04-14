"""Tests for backfill_v6 utility."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def bf_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    root = Path(__file__).parent.parent
    conn.executescript((root / "migrations" / "001_v5_schema.sql").read_text())
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT, project TEXT DEFAULT 'general',
            status TEXT DEFAULT 'active', created_at TEXT
        );
        """
    )
    for m in ("002_multi_representation", "003_triple_extraction_queue",
              "004_deep_enrichment", "005_representations_queue"):
        conn.executescript((root / "migrations" / f"{m}.sql").read_text())
    yield conn
    conn.close()


def _add(db, content="x", project="demo", status="active"):
    return db.execute(
        "INSERT INTO knowledge (content, project, status, created_at) "
        "VALUES (?, ?, ?, '2026-04-14T00:00:00Z')",
        (content, project, status),
    ).lastrowid


def test_backfill_enqueues_all_active(bf_db):
    from tools.backfill_v6 import backfill

    k1 = _add(bf_db, project="a")
    k2 = _add(bf_db, project="a")
    k3 = _add(bf_db, project="b")

    result = backfill(bf_db)
    assert result["scanned"] == 3
    assert result["triples"] == 3
    assert result["enrichment"] == 3
    assert result["representations"] == 3


def test_backfill_filter_by_project(bf_db):
    from tools.backfill_v6 import backfill

    _add(bf_db, project="a")
    _add(bf_db, project="a")
    _add(bf_db, project="b")

    result = backfill(bf_db, project="a")
    assert result["scanned"] == 2


def test_backfill_skips_archived(bf_db):
    from tools.backfill_v6 import backfill

    _add(bf_db, status="active")
    _add(bf_db, status="archived")
    _add(bf_db, status="superseded")

    result = backfill(bf_db)
    assert result["scanned"] == 1


def test_backfill_idempotent(bf_db):
    from tools.backfill_v6 import backfill

    _add(bf_db)
    _add(bf_db)

    # First run enqueues everything
    r1 = backfill(bf_db)
    assert r1["representations"] == 2
    # Second run — already pending, nothing new added
    r2 = backfill(bf_db)
    assert r2["representations"] == 0
    assert r2["triples"] == 0
    assert r2["enrichment"] == 0
