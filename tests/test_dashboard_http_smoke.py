"""Black-box HTTP smoke test against an ephemeral dashboard instance.

Spins dashboard up on a random port, hits the new v6 endpoints and live page.
"""

from __future__ import annotations

import json
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dashboard_server():
    tmp = Path(tempfile.mkdtemp(prefix="dash_smoke_"))
    db_path = tmp / "memory.db"

    # Seed DB with the v6 schema + a sample graph node + filter saving
    root = Path(__file__).parent.parent
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript((root / "migrations" / "001_v5_schema.sql").read_text())
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, type TEXT,
            content TEXT, context TEXT DEFAULT '', project TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]', status TEXT DEFAULT 'active', superseded_by INTEGER,
            confidence REAL DEFAULT 1.0, source TEXT DEFAULT 'explicit',
            created_at TEXT, last_confirmed TEXT, recall_count INTEGER DEFAULT 0,
            last_recalled TEXT, branch TEXT DEFAULT ''
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            content, context, tags, content='knowledge', content_rowid='id'
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT,
            project TEXT DEFAULT 'general', status TEXT DEFAULT 'open',
            summary TEXT, log_count INTEGER DEFAULT 0, branch TEXT DEFAULT ''
        );
        """
    )
    for m in ("002_multi_representation", "003_triple_extraction_queue",
              "004_deep_enrichment", "005_representations_queue",
              "006_filter_savings"):
        conn.executescript((root / "migrations" / f"{m}.sql").read_text())

    # Seed a filter saving row so savings endpoint has something to return
    conn.execute(
        "INSERT INTO filter_savings "
        "(knowledge_id, filter_name, input_chars, output_chars, reduction_pct, safety, created_at) "
        "VALUES (1, 'pytest', 2000, 400, 80.0, 'strict', '2026-04-14T00:00:00Z')"
    )
    # Seed a graph node
    conn.execute(
        "INSERT INTO graph_nodes (id, type, name, first_seen_at, last_seen_at, mention_count) "
        "VALUES ('n-sample', 'concept', 'sample', '2026-04-14T00:00:00Z', '2026-04-14T00:00:00Z', 1)"
    )
    # Enable WAL on the RW connection so readonly consumers don't try to set it
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    conn.close()

    # Patch dashboard's DB_PATH
    sys.path.insert(0, str(root / "src"))
    import dashboard  # noqa: E402
    dashboard.DB_PATH = db_path

    class _ThreadedHTTP(HTTPServer):
        daemon_threads = True

    port = _free_port()
    server = _ThreadedHTTP(("127.0.0.1", port), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8") if e.fp else ""


def test_v6_savings_endpoint(dashboard_server):
    data = _get_json(f"{dashboard_server}/api/v6/savings")
    assert data["applied_count"] == 1
    assert data["chars_saved"] == 1600
    assert data["tokens_saved_estimate"] == 400
    assert data["by_filter"][0]["name"] == "pytest"


def test_v6_queues_endpoint(dashboard_server):
    data = _get_json(f"{dashboard_server}/api/v6/queues")
    assert "triple_extraction_queue" in data
    assert "deep_enrichment_queue" in data
    assert "representations_queue" in data


def test_v6_coverage_endpoint(dashboard_server):
    data = _get_json(f"{dashboard_server}/api/v6/coverage")
    assert "active_knowledge" in data
    assert "representations_pct" in data


def test_graph_delta_endpoint(dashboard_server):
    data = _get_json(f"{dashboard_server}/api/graph/delta")
    assert "nodes" in data
    assert "edges" in data
    assert any(n["id"] == "n-sample" for n in data["nodes"])


def test_main_page_includes_v6_panels(dashboard_server):
    status, body = _get(f"{dashboard_server}/")
    assert status == 200
    assert "Token savings" in body
    assert "v6 queues" in body
    assert "v6 coverage" in body
    # Marker must be replaced, not left raw
    assert "V6_PANELS_HERE" not in body


def test_live_graph_page_loads(dashboard_server):
    status, body = _get(f"{dashboard_server}/graph/live")
    assert status == 200
    assert "Graph Live" in body
    # WebGL 3D renderer is loaded
    assert "3d-force-graph" in body or "ForceGraph3D" in body
    # Back button + filter slider present
    assert "btnBack" in body
    assert "minMentions" in body
