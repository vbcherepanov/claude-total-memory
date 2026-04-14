"""Tests for backfill_orphan_edges utility."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def bo_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    root = Path(__file__).parent.parent
    conn.executescript((root / "migrations" / "001_v5_schema.sql").read_text())
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT, status TEXT DEFAULT 'active',
            superseded_by INTEGER, created_at TEXT
        );
        """
    )
    conn.executescript((root / "migrations" / "003_triple_extraction_queue.sql").read_text())
    yield conn
    conn.close()


def _add_node(db, name: str, type_: str = "concept", mentions: int = 1) -> str:
    import uuid
    nid = uuid.uuid4().hex
    db.execute(
        "INSERT INTO graph_nodes (id, type, name, mention_count, first_seen_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, '2026-04-14T00:00:00Z', '2026-04-14T00:00:00Z')",
        (nid, type_, name, mentions),
    )
    return nid


def _add_knowledge(db, content: str) -> int:
    return db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, '2026-04-14T00:00:00Z')",
        (content,),
    ).lastrowid


def _link(db, kid: int, node_id: str):
    db.execute(
        "INSERT OR REPLACE INTO knowledge_nodes (knowledge_id, node_id, role, strength) "
        "VALUES (?, ?, 'mentions', 1.0)",
        (kid, node_id),
    )


def _add_edge(db, src: str, tgt: str, rel: str = "uses"):
    import uuid
    db.execute(
        "INSERT INTO graph_edges (id, source_id, target_id, relation_type, weight, created_at) "
        "VALUES (?, ?, ?, ?, 1.0, '2026-04-14T00:00:00Z')",
        (uuid.uuid4().hex, src, tgt, rel),
    )


def test_find_orphans_excludes_connected_nodes(bo_db):
    from tools.backfill_orphan_edges import find_orphan_nodes

    a = _add_node(bo_db, "alpha", mentions=3)
    b = _add_node(bo_db, "beta", mentions=2)
    c = _add_node(bo_db, "gamma", mentions=5)
    _add_edge(bo_db, a, b)
    bo_db.commit()

    orphans = find_orphan_nodes(bo_db, min_mentions=1)
    names = {r["name"] for r in orphans}
    assert "gamma" in names
    assert "alpha" not in names
    assert "beta" not in names


def test_find_orphans_respects_min_mentions(bo_db):
    from tools.backfill_orphan_edges import find_orphan_nodes

    _add_node(bo_db, "rare", mentions=1)
    _add_node(bo_db, "common", mentions=10)
    bo_db.commit()

    assert {r["name"] for r in find_orphan_nodes(bo_db, min_mentions=1)} == {"rare", "common"}
    assert {r["name"] for r in find_orphan_nodes(bo_db, min_mentions=5)} == {"common"}


def test_find_orphans_orders_by_mentions_desc(bo_db):
    from tools.backfill_orphan_edges import find_orphan_nodes

    _add_node(bo_db, "low", mentions=2)
    _add_node(bo_db, "high", mentions=50)
    _add_node(bo_db, "mid", mentions=10)
    bo_db.commit()

    orphans = find_orphan_nodes(bo_db, min_mentions=1)
    assert [r["name"] for r in orphans] == ["high", "mid", "low"]


def test_knowledge_ids_linked_to_nodes_deduplicates(bo_db):
    from tools.backfill_orphan_edges import knowledge_ids_linked_to_nodes

    n1 = _add_node(bo_db, "n1")
    n2 = _add_node(bo_db, "n2")
    k1 = _add_knowledge(bo_db, "shared")
    k2 = _add_knowledge(bo_db, "other")
    _link(bo_db, k1, n1)
    _link(bo_db, k1, n2)   # same knowledge, two nodes
    _link(bo_db, k2, n2)
    bo_db.commit()

    kids = knowledge_ids_linked_to_nodes(bo_db, [n1, n2])
    assert set(kids) == {k1, k2}
    assert len(kids) == 2  # k1 only once despite two links


def test_knowledge_ids_skips_archived(bo_db):
    from tools.backfill_orphan_edges import knowledge_ids_linked_to_nodes

    n = _add_node(bo_db, "x")
    k_active = _add_knowledge(bo_db, "active")
    k_arch = _add_knowledge(bo_db, "archived")
    _link(bo_db, k_active, n)
    _link(bo_db, k_arch, n)
    bo_db.execute("UPDATE knowledge SET status='archived' WHERE id=?", (k_arch,))
    bo_db.commit()

    kids = knowledge_ids_linked_to_nodes(bo_db, [n])
    assert kids == [k_active]


def test_enqueue_is_idempotent(bo_db):
    from tools.backfill_orphan_edges import enqueue

    k1 = _add_knowledge(bo_db, "a")
    k2 = _add_knowledge(bo_db, "b")
    bo_db.commit()

    # First run enqueues 2
    assert enqueue(bo_db, [k1, k2]) == 2
    # Second run — already pending, adds nothing
    assert enqueue(bo_db, [k1, k2]) == 0

    cnt = bo_db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE status='pending'"
    ).fetchone()[0]
    assert cnt == 2
