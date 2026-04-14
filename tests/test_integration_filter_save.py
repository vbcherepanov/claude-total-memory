"""Integration: memory_save(filter="pytest") reduces content + logs filter_savings."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    yield s
    try:
        s.db.close()
    except Exception:
        pass


def test_save_with_pytest_filter_reduces_content(store):
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()

    noisy = """
============== test session starts ==============
platform darwin -- Python 3.13
collected 100 items
test_a.py::test_1 PASSED
test_a.py::test_2 PASSED
test_a.py::test_3 PASSED
test_b.py::test_x FAILED
test_b.py:42: AssertionError: expected 1 got 2
test_c.py::test_4 PASSED
test_c.py::test_5 PASSED
====== 1 failed, 99 passed in 2.34s ======
""" * 5

    original_len = len(noisy)
    rid, _, _ = store.save_knowledge(
        sid="s1", content=noisy, ktype="fact", project="demo",
        filter_name="pytest",
    )

    # Knowledge content is shorter than original
    saved = store.db.execute(
        "SELECT content FROM knowledge WHERE id=?", (rid,)
    ).fetchone()["content"]
    assert len(saved) < original_len
    # Critical info preserved
    assert "FAILED" in saved
    assert "AssertionError" in saved
    # Noise removed
    assert "collected 100 items" not in saved

    # filter_savings row exists
    row = store.db.execute(
        "SELECT filter_name, input_chars, output_chars, reduction_pct "
        "FROM filter_savings WHERE knowledge_id=?", (rid,)
    ).fetchone()
    assert row is not None
    assert row["filter_name"] == "pytest"
    assert row["input_chars"] == original_len
    assert row["output_chars"] < row["input_chars"]
    assert row["reduction_pct"] > 0


def test_save_without_filter_no_reduction(store):
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()

    content = "normal knowledge content"
    rid, _, _ = store.save_knowledge(
        sid="s1", content=content, ktype="fact", project="demo"
    )
    saved = store.db.execute(
        "SELECT content FROM knowledge WHERE id=?", (rid,)
    ).fetchone()["content"]
    assert saved == content

    # No filter_savings row
    count = store.db.execute(
        "SELECT COUNT(*) FROM filter_savings WHERE knowledge_id=?", (rid,)
    ).fetchone()[0]
    assert count == 0


def test_stats_reports_filter_savings(store):
    import server as _srv

    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()

    big_noise = ("DEBUG: something\n" * 500) + "ERROR: critical fail"
    store.save_knowledge(
        sid="s1", content=big_noise, ktype="fact", project="demo",
        filter_name="generic_logs",
    )

    stats = _srv.Recall(store).stats()
    fs = stats["v6_filter_savings"]
    assert fs["applied_count"] == 1
    assert fs["chars_saved"] > 0
    assert fs["tokens_saved_estimate"] > 0
    assert fs["total_reduction_pct"] > 0


def test_unknown_filter_name_falls_back_to_raw(store):
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()

    content = "raw content unchanged"
    rid, _, _ = store.save_knowledge(
        sid="s1", content=content, ktype="fact", project="demo",
        filter_name="bogus_filter_name",
    )
    # No crash, content unchanged
    saved = store.db.execute(
        "SELECT content FROM knowledge WHERE id=?", (rid,)
    ).fetchone()["content"]
    assert saved == content
