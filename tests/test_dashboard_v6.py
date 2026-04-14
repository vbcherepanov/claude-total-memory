"""Tests for dashboard_v6 API endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def dash_db():
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
              "004_deep_enrichment", "005_representations_queue",
              "006_filter_savings"):
        conn.executescript((root / "migrations" / f"{m}.sql").read_text())
    yield conn
    conn.close()


def test_savings_aggregates_empty(dash_db):
    from dashboard_v6 import api_v6_savings

    res = api_v6_savings(dash_db)
    assert res["applied_count"] == 0
    assert res["tokens_saved_estimate"] == 0
    assert res["by_filter"] == []


def test_savings_with_data(dash_db):
    from dashboard_v6 import api_v6_savings

    dash_db.executemany(
        "INSERT INTO filter_savings "
        "(knowledge_id, filter_name, input_chars, output_chars, reduction_pct, safety, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            (1, "pytest", 1000, 200, 80.0, "strict", "2026-04-14T00:00:00Z"),
            (2, "pytest", 2000, 500, 75.0, "strict", "2026-04-14T00:00:00Z"),
            (3, "cargo",  800,  300, 62.5, "strict", "2026-04-14T00:00:00Z"),
        ],
    )
    dash_db.commit()

    res = api_v6_savings(dash_db)
    assert res["applied_count"] == 3
    assert res["chars_saved"] == (1000+2000+800) - (200+500+300)
    assert res["tokens_saved_estimate"] == res["chars_saved"] // 4
    names = {b["name"] for b in res["by_filter"]}
    assert names == {"pytest", "cargo"}


def test_queues_all_empty_tables_return_zeros(dash_db):
    from dashboard_v6 import api_v6_queues

    res = api_v6_queues(dash_db)
    assert set(res.keys()) == {
        "triple_extraction_queue", "deep_enrichment_queue", "representations_queue"
    }
    for counts in res.values():
        assert counts.get("pending") == 0
        assert counts.get("done") == 0


def test_queues_report_status_breakdown(dash_db):
    from dashboard_v6 import api_v6_queues

    dash_db.executemany(
        "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at) VALUES (?,?,?)",
        [(1, "pending", "t"), (2, "pending", "t"), (3, "done", "t"), (4, "failed", "t")],
    )
    dash_db.commit()
    res = api_v6_queues(dash_db)
    assert res["triple_extraction_queue"]["pending"] == 2
    assert res["triple_extraction_queue"]["done"] == 1
    assert res["triple_extraction_queue"]["failed"] == 1


def test_coverage_no_knowledge(dash_db):
    from dashboard_v6 import api_v6_coverage

    res = api_v6_coverage(dash_db)
    assert res["active_knowledge"] == 0
    assert res["representations_pct"] == 0


def test_coverage_computes_percentages(dash_db):
    from dashboard_v6 import api_v6_coverage

    for i in range(4):
        dash_db.execute(
            "INSERT INTO knowledge (content, project, status, created_at) "
            "VALUES ('c','demo','active','2026-04-14T00:00:00Z')"
        )
    dash_db.executemany(
        "INSERT INTO knowledge_representations "
        "(knowledge_id, representation, content, binary_vector, float32_vector, embed_model, embed_dim, created_at) "
        "VALUES (?, 'raw', ?, ?, ?, 'fake', 4, ?)",
        [(1, "c", b"\0" * 4, b"\0" * 16, "t"), (2, "c", b"\0" * 4, b"\0" * 16, "t")],
    )
    dash_db.execute(
        "INSERT INTO knowledge_enrichment (knowledge_id, entities, intent, topics, updated_at) "
        "VALUES (1, '[]', 'fact', '[]', 't')"
    )
    dash_db.commit()

    res = api_v6_coverage(dash_db)
    assert res["active_knowledge"] == 4
    assert res["representations_pct"] == 50.0   # 2/4
    assert res["enrichment_pct"] == 25.0        # 1/4


def test_graph_delta_returns_recent_nodes_and_edges(dash_db):
    from dashboard_v6 import api_graph_delta

    dash_db.executemany(
        "INSERT INTO graph_nodes (id, type, name, first_seen_at, last_seen_at) "
        "VALUES (?, 'concept', ?, ?, ?)",
        [
            ("n1", "alpha", "2026-04-14T00:00:00Z", "2026-04-14T00:00:00Z"),
            ("n2", "beta",  "2026-04-14T01:00:00Z", "2026-04-14T01:00:00Z"),
        ],
    )
    dash_db.execute(
        "INSERT INTO graph_edges (id, source_id, target_id, relation_type, weight, created_at) "
        "VALUES ('e1', 'n1', 'n2', 'uses', 1.0, '2026-04-14T01:30:00Z')"
    )
    dash_db.commit()

    res = api_graph_delta(dash_db, since="2026-04-13T00:00:00Z")
    assert len(res["nodes"]) == 2
    assert len(res["edges"]) == 1
    assert res["max_ts"] >= "2026-04-14T01:30:00Z"

    res2 = api_graph_delta(dash_db, since="2026-04-14T00:30:00Z")
    names = {n["name"] for n in res2["nodes"]}
    assert "beta" in names
    # alpha still comes back as a filled endpoint so the edge is drawable
    assert "alpha" in names
    assert res2["stats"]["endpoint_nodes_filled"] >= 1


def test_graph_delta_fills_missing_edge_endpoints(dash_db):
    """Edges pointing to nodes older than `since` must have their endpoints added."""
    from dashboard_v6 import api_graph_delta

    # Three nodes: two old, one new. Edges bind old→old, old→new, new→old.
    dash_db.executemany(
        "INSERT INTO graph_nodes (id, type, name, first_seen_at, last_seen_at) "
        "VALUES (?, 'concept', ?, ?, ?)",
        [
            ("old1", "old_a", "2026-04-10T00:00:00Z", "2026-04-10T00:00:00Z"),
            ("old2", "old_b", "2026-04-10T00:00:00Z", "2026-04-10T00:00:00Z"),
            ("new1", "new_x", "2026-04-14T00:00:00Z", "2026-04-14T00:00:00Z"),
        ],
    )
    # A fresh edge that links old→old (both endpoints are older than `since`)
    dash_db.execute(
        "INSERT INTO graph_edges (id, source_id, target_id, relation_type, weight, created_at) "
        "VALUES ('e_old_old', 'old1', 'old2', 'ref', 1.0, '2026-04-14T00:30:00Z')"
    )
    dash_db.commit()

    res = api_graph_delta(dash_db, since="2026-04-13T00:00:00Z")
    node_ids = {n["id"] for n in res["nodes"]}
    # All endpoints present even though old1/old2 predate `since`
    assert {"old1", "old2", "new1"}.issubset(node_ids)
    assert res["stats"]["endpoint_nodes_filled"] >= 2
