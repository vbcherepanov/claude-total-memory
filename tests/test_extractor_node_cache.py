"""Graph node-name cache: correct after writes, without re-reading the table.

`_ensure_node` used to drop the whole cache whenever it created a node, and
`extract_and_link` dropped it again at the end. Since nearly every save creates
at least one node, the 60s TTL never applied and the next save re-read all of
`graph_nodes`.

Replacing that with an incremental insert is only safe if the cache still
answers correctly, so these tests pin the behaviour rather than the timing:
a node created through the cache must be found again, a node written behind
the cache's back must arrive within the TTL, and node identity must never be
duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def extractor(db):
    from ingestion.extractor import ConceptExtractor

    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id TEXT PRIMARY KEY, type TEXT, name TEXT, content TEXT,
            source TEXT DEFAULT 'auto', status TEXT DEFAULT 'active',
            mention_count INTEGER DEFAULT 1,
            first_seen_at TEXT, last_seen_at TEXT
        );
        CREATE TABLE IF NOT EXISTS graph_edges (
            id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
            relation_type TEXT, weight REAL DEFAULT 1.0,
            reinforcement_count INTEGER DEFAULT 0,
            created_at TEXT, last_reinforced_at TEXT
        );
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            knowledge_id INTEGER, node_id TEXT, role TEXT, strength REAL,
            PRIMARY KEY (knowledge_id, node_id)
        );
        """
    )
    db.commit()
    return ConceptExtractor(db)


def _reads(extractor, monkeypatch) -> list[int]:
    """Count full `graph_nodes` scans by wrapping the refresh."""
    calls: list[int] = []
    original = type(extractor)._get_node_names

    def counting(self):
        before = self._node_names_cache
        result = original(self)
        if before is not result or before is None:
            calls.append(1)
        return result

    monkeypatch.setattr(type(extractor), "_get_node_names", counting)
    return calls


def test_a_created_node_is_immediately_visible_to_the_cache(extractor):
    node_id = extractor._ensure_node("postgresql", type="technology")
    cache = extractor._get_node_names()
    assert cache["postgresql"]["id"] == node_id
    assert cache["postgresql"]["type"] == "technology"


def test_the_same_name_never_creates_a_second_node(extractor):
    first = extractor._ensure_node("rabbitmq", type="technology")
    second = extractor._ensure_node("RabbitMQ", type="technology")
    third = extractor._ensure_node("  rabbitmq  ", type="technology")
    assert first == second == third

    count = extractor.db.execute(
        "SELECT COUNT(*) FROM graph_nodes WHERE name='rabbitmq'"
    ).fetchone()[0]
    assert count == 1


def test_creating_nodes_does_not_force_a_full_reread_each_time(extractor, monkeypatch):
    """The regression: one scan to warm up, none per created node."""
    scans = _reads(extractor, monkeypatch)
    for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
        extractor._ensure_node(name, type="concept")
    assert len(scans) == 1, f"re-read graph_nodes {len(scans)} times for 5 nodes"


def test_mention_count_still_increments_on_a_cache_hit(extractor):
    node_id = extractor._ensure_node("redis", type="technology")
    for _ in range(3):
        extractor._ensure_node("redis", type="technology")
    mentions = extractor.db.execute(
        "SELECT mention_count FROM graph_nodes WHERE id = ?", (node_id,)
    ).fetchone()[0]
    assert mentions == 4


def test_a_node_written_behind_the_cache_is_picked_up_after_the_ttl(extractor):
    """Another writer's node must not stay invisible forever."""
    extractor._ensure_node("known", type="concept")
    extractor.db.execute(
        "INSERT INTO graph_nodes (id, type, name, source, first_seen_at, last_seen_at) "
        "VALUES ('outside-1', 'concept', 'written-elsewhere', 'auto', '', '')"
    )
    extractor.db.commit()

    assert "written-elsewhere" not in extractor._get_node_names()

    # Age the cache past its TTL rather than sleeping through it.
    extractor._cache_timestamp -= 120
    assert "written-elsewhere" in extractor._get_node_names()


def test_cache_put_is_a_no_op_before_the_cache_is_warm(extractor):
    """Nothing should be resurrected into a cache that was never loaded."""
    extractor._node_names_cache = None
    extractor._cache_put("ghost", "id-1", "concept")
    assert extractor._node_names_cache is None
