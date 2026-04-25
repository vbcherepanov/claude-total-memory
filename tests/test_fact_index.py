"""Unit tests for src/fact_index.py — the v9.0 L2 dual-index lane.

Each test builds a tiny synthetic SQLite DB compatible with the v5/v6
schema (graph_nodes, graph_edges, knowledge, knowledge_nodes). We never
touch the real memory.db so these run in <100ms with no side effects.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fact_index import FactHit, FactIndex, extract_candidates  # noqa: E402


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _schema(db: sqlite3.Connection) -> None:
    """Minimal schema subset used by FactIndex."""
    db.executescript(
        """
        CREATE TABLE graph_nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL
        );
        CREATE TABLE graph_edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            context TEXT,
            reinforcement_count INTEGER DEFAULT 0
        );
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT,
            content TEXT,
            type TEXT,
            tags TEXT,
            context TEXT
        );
        CREATE TABLE knowledge_nodes (
            knowledge_id INTEGER,
            node_id TEXT,
            role TEXT,
            strength REAL DEFAULT 1.0,
            PRIMARY KEY (knowledge_id, node_id)
        );
        """
    )


def _add_node(db: sqlite3.Connection, nid: str, name: str, typ: str = "person") -> None:
    db.execute("INSERT INTO graph_nodes(id,type,name) VALUES(?,?,?)", (nid, typ, name))


def _add_edge(
    db: sqlite3.Connection,
    eid: str,
    src: str,
    rel: str,
    tgt: str,
    weight: float = 1.0,
    context: str = "",
) -> None:
    db.execute(
        "INSERT INTO graph_edges(id,source_id,target_id,relation_type,weight,context) "
        "VALUES(?,?,?,?,?,?)",
        (eid, src, tgt, rel, weight, context),
    )


def _add_knowledge(
    db: sqlite3.Connection,
    kid: int,
    project: str,
    content: str,
    node_id: str | None = None,
) -> None:
    db.execute(
        "INSERT INTO knowledge(id,project,content,type) VALUES(?,?,?,?)",
        (kid, project, content, "fact"),
    )
    if node_id:
        db.execute(
            "INSERT INTO knowledge_nodes(knowledge_id,node_id,role,strength) VALUES(?,?,?,?)",
            (kid, node_id, "subject", 1.0),
        )


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _schema(conn)
    return conn


@pytest.fixture
def rich_db(db: sqlite3.Connection) -> sqlite3.Connection:
    """A small knowledge graph about Alice, Bob, Berlin, Tokyo."""
    # Entities
    _add_node(db, "n_alice", "Alice", "person")
    _add_node(db, "n_bob", "Bob", "person")
    _add_node(db, "n_berlin", "Berlin", "place")
    _add_node(db, "n_tokyo", "Tokyo", "place")
    _add_node(db, "n_google", "Google", "company")
    _add_node(db, "n_mit", "MIT", "org")
    # Facts
    _add_edge(db, "e1", "n_alice", "traveled_to", "n_berlin", weight=1.5)
    _add_edge(db, "e2", "n_alice", "traveled_to", "n_tokyo", weight=1.2)
    _add_edge(db, "e3", "n_alice", "works_at", "n_google", weight=2.0)
    _add_edge(db, "e4", "n_bob", "studied_at", "n_mit", weight=1.0)
    # Noise
    _add_edge(db, "e5", "n_alice", "mentioned_with", "n_bob", weight=0.5)
    _add_edge(db, "e6", "n_alice", "supersedes", "n_bob", weight=5.0)
    # Linked knowledge rows
    _add_knowledge(db, 10, "demo", "Alice took a trip to Berlin in May 2023.", "n_alice")
    _add_knowledge(db, 11, "demo", "Alice visited Tokyo last fall.", "n_alice")
    _add_knowledge(db, 12, "demo", "Alice is an engineer at Google.", "n_alice")
    _add_knowledge(db, 13, "other-proj", "Bob graduated from MIT in 2018.", "n_bob")
    db.commit()
    return db


# ──────────────────────────────────────────────
# extract_candidates
# ──────────────────────────────────────────────


def test_extract_candidates_simple_question():
    ents, attrs = extract_candidates("Where did Alice travel?")
    assert "Alice" in ents
    # "where" expands to travel/visit synonyms; "travel" root expands aliases
    assert any("travel" in a for a in attrs)
    assert "visited" in attrs or "went_to" in attrs


def test_extract_candidates_multi_word_entity():
    ents, _ = extract_candidates("When did New York host the event?")
    assert any(e.lower() == "new york" for e in ents)


def test_extract_candidates_no_entities_returns_empty():
    ents, attrs = extract_candidates("what about it?")
    # All stopwords → no capitalized entity; attrs may still be populated
    assert ents == []


# ──────────────────────────────────────────────
# FactIndex.lookup
# ──────────────────────────────────────────────


def test_lookup_by_entity_returns_all_real_edges(rich_db):
    idx = FactIndex(rich_db)
    hits = idx.lookup("Alice", limit=20)
    # Real edges only — mentioned_with / supersedes filtered.
    relations = {h.relation for h in hits}
    assert "traveled_to" in relations
    assert "works_at" in relations
    assert "mentioned_with" not in relations
    assert "supersedes" not in relations


def test_lookup_is_case_insensitive(rich_db):
    idx = FactIndex(rich_db)
    assert idx.lookup("alice") == idx.lookup("Alice") == idx.lookup("ALICE")


def test_lookup_with_attribute_substring(rich_db):
    idx = FactIndex(rich_db)
    hits = idx.lookup("Alice", "travel")
    assert len(hits) == 2
    assert {h.value for h in hits} == {"Berlin", "Tokyo"}


def test_lookup_respects_limit(rich_db):
    idx = FactIndex(rich_db)
    assert len(idx.lookup("Alice", limit=1)) == 1
    assert len(idx.lookup("Alice", limit=100)) == 3  # 3 real edges


def test_lookup_returns_knowledge_id(rich_db):
    idx = FactIndex(rich_db)
    hits = idx.lookup("Alice", "works")
    assert len(hits) == 1
    assert hits[0].value == "Google"
    assert hits[0].knowledge_id in {10, 11, 12}  # any Alice-linked row


def test_lookup_orders_by_weight(rich_db):
    idx = FactIndex(rich_db)
    hits = idx.lookup("Alice")
    # weights: works_at=2.0 > traveled_to(Berlin)=1.5 > traveled_to(Tokyo)=1.2
    assert hits[0].relation == "works_at"


def test_lookup_filters_by_project(rich_db):
    idx = FactIndex(rich_db)
    # Bob's knowledge is in "other-proj"; scoping to "demo" must exclude him.
    hits = idx.lookup("Bob", project="demo")
    assert hits == []
    hits_full = idx.lookup("Bob")
    assert len(hits_full) == 1 and hits_full[0].value == "MIT"


def test_lookup_empty_entity_returns_nothing(rich_db):
    idx = FactIndex(rich_db)
    assert idx.lookup("") == []
    assert idx.lookup("   ") == []


def test_lookup_unknown_entity_returns_empty(rich_db):
    idx = FactIndex(rich_db)
    assert idx.lookup("Zebedee") == []


def test_hit_to_dict_is_json_serializable(rich_db):
    import json

    idx = FactIndex(rich_db)
    hit = idx.lookup("Alice", limit=1)[0]
    d = hit.to_dict()
    assert json.dumps(d)  # no TypeError
    assert d["entity"] == "Alice"


# ──────────────────────────────────────────────
# FactIndex.lookup_query + knowledge_ids_for
# ──────────────────────────────────────────────


def test_lookup_query_extracts_entity_and_matches(rich_db):
    idx = FactIndex(rich_db)
    hits = idx.lookup_query("Where did Alice travel?")
    values = {h.value for h in hits}
    # Both cities should match via "travel" alias expansion
    assert "Berlin" in values and "Tokyo" in values


def test_lookup_query_falls_back_to_entity_only(rich_db):
    """When attribute hints don't match anything, still return edges for the entity."""
    idx = FactIndex(rich_db)
    hits = idx.lookup_query("Tell me about Alice.")
    assert hits, "should still surface Alice's real edges"
    rels = {h.relation for h in hits}
    assert "mentioned_with" not in rels  # noise filter still on


def test_lookup_query_no_entity_returns_empty(rich_db):
    idx = FactIndex(rich_db)
    assert idx.lookup_query("hey, what's up?") == []


def test_knowledge_ids_for_deduplicates(rich_db):
    idx = FactIndex(rich_db)
    ids = idx.knowledge_ids_for("Where did Alice travel? What did Alice do?")
    assert len(ids) == len(set(ids))


def test_knowledge_ids_for_respects_project(rich_db):
    idx = FactIndex(rich_db)
    ids_demo = idx.knowledge_ids_for("Bob", project="demo")
    assert ids_demo == []


# ──────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────


def test_stats_counts_nodes_and_real_edges(rich_db):
    idx = FactIndex(rich_db)
    s = idx.stats()
    assert s["nodes"] == 6
    assert s["real_edges"] == 4  # 6 total - 2 noise
    assert s["relation_types"] == 5  # traveled_to, works_at, studied_at, mentioned_with, supersedes
    assert "mentioned_with" in s["noisy_filtered"]


# ──────────────────────────────────────────────
# Thread safety smoke
# ──────────────────────────────────────────────


def test_concurrent_lookups_do_not_crash(tmp_path):
    import threading

    # Real file DB with check_same_thread=False — :memory: is per-thread.
    db_path = tmp_path / "fact_index.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _schema(conn)
    _add_node(conn, "n_alice", "Alice", "person")
    _add_node(conn, "n_berlin", "Berlin", "place")
    _add_edge(conn, "e1", "n_alice", "traveled_to", "n_berlin", weight=1.5)
    conn.commit()

    idx = FactIndex(conn)
    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(50):
                idx.lookup("Alice")
                idx.lookup_query("Where did Alice travel?")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conn.close()

    assert errors == [], f"thread-safety broke: {errors}"
