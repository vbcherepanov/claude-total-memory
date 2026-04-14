"""Tests for autofilter — content sniffer that picks the right TOML filter."""

from __future__ import annotations

import pytest


def test_pytest_output_detected():
    from autofilter import detect_filter
    text = """============== test session starts ==============
platform darwin -- Python 3.13
collected 10 items
test_a.py::test_one PASSED
test_b.py::test_x FAILED
test_b.py:42: AssertionError: expected 1 got 2
====== 1 failed, 9 passed in 0.05s ======"""
    assert detect_filter(text) == "pytest"


def test_cargo_output_detected():
    from autofilter import detect_filter
    text = """   Compiling foo v0.1.0
   Compiling bar v0.2.0
error[E0308]: mismatched types
  --> src/main.rs:10:5
   |
10 |     let x: i32 = "hello";
   Finished dev [unoptimized] target(s) in 2.34s"""
    assert detect_filter(text) == "cargo"


def test_git_status_detected():
    from autofilter import detect_filter
    text = """On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
    modified:   src/server.py
    modified:   src/reflection/agent.py"""
    assert detect_filter(text) == "git_status"


def test_docker_ps_detected():
    from autofilter import detect_filter
    text = """CONTAINER ID   IMAGE     COMMAND        CREATED        STATUS       PORTS     NAMES
abc123         nginx     "nginx"       2 hours ago    Up 2 hours   80/tcp    web
def456         redis     "redis-server" 1 day ago     Exited (0)             cache"""
    assert detect_filter(text) == "docker_ps"


def test_generic_logs_fallback():
    from autofilter import detect_filter
    text = """DEBUG: connection opened
INFO: processing request
WARN: slow query detected
ERROR: timeout
FATAL: crash in /var/log/x.log"""
    assert detect_filter(text) == "generic_logs"


def test_normal_text_returns_none():
    """Regular knowledge (decisions, facts) → no filter."""
    from autofilter import detect_filter
    assert detect_filter("User prefers Go for backend services.") is None
    assert detect_filter("Decision: use PostgreSQL 18 with UUID v7 primary keys.") is None
    assert detect_filter("") is None


def test_short_text_returns_none():
    """Don't bother filtering tiny content."""
    from autofilter import detect_filter
    # Even if it matches patterns, below min length = skip
    assert detect_filter("DEBUG ok") is None


def test_code_block_not_treated_as_logs():
    """Markdown code examples shouldn't trigger log filters."""
    from autofilter import detect_filter
    text = """Here is how to use the API:

```python
def save():
    return memory_save("content")
```

Note that `save()` always returns an id."""
    assert detect_filter(text) is None


# ──────────────────────────────────────────────
# New filter types
# ──────────────────────────────────────────────


def test_stack_trace_detected():
    from autofilter import detect_filter
    text = """Traceback (most recent call last):
  File "/Users/x/server.py", line 42, in handle
    return self.process(req)
  File "/Users/x/processor.py", line 15, in process
    raise ValueError("bad input")
ValueError: bad input — got None for required field"""
    assert detect_filter(text) == "stack_trace"


def test_npm_yarn_detected():
    from autofilter import detect_filter
    text = """yarn install v1.22.19
[1/4] Resolving packages...
[2/4] Fetching packages...
[3/4] Linking dependencies...
warning Pattern \"react@*\" is trying to unpack in the same destination
[4/4] Building fresh packages...
Done in 12.34s.
added 1234 packages in 12s
3 vulnerabilities (1 low, 2 high)"""
    assert detect_filter(text) == "npm_yarn"


def test_http_log_detected():
    from autofilter import detect_filter
    text = """127.0.0.1 - - [14/Apr/2026:10:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234
127.0.0.1 - - [14/Apr/2026:10:00:01 +0000] "POST /api/login HTTP/1.1" 401 87
127.0.0.1 - - [14/Apr/2026:10:00:02 +0000] "GET /api/orders/42 HTTP/1.1" 500 200
127.0.0.1 - - [14/Apr/2026:10:00:03 +0000] "DELETE /api/users/7 HTTP/1.1" 204 0"""
    assert detect_filter(text) == "http_log"


def test_sql_explain_detected():
    from autofilter import detect_filter
    text = """ Hash Join  (cost=12.34..56.78 rows=100 width=64) (actual time=1.23..4.56 rows=98 loops=1)
   Hash Cond: (orders.user_id = users.id)
   ->  Seq Scan on orders  (cost=0.00..30.00 rows=1000 width=32) (actual time=0.01..2.34 rows=1000 loops=1)
   ->  Hash  (cost=4.50..4.50 rows=200 width=32) (actual time=0.45..0.45 rows=200 loops=1)
         ->  Index Scan using users_pkey on users  (cost=0.28..4.50 rows=200 width=32)
 Planning Time: 0.234 ms
 Execution Time: 5.123 ms"""
    assert detect_filter(text) == "sql_explain"


def test_json_blob_detected():
    from autofilter import detect_filter
    text = """{
  "users": [
    {"id": 1, "name": "alice", "email": "a@example.com"},
    {"id": 2, "name": "bob", "email": "b@example.com"},
    {"id": 3, "name": "carol", "email": "c@example.com"}
  ],
  "total": 3,
  "page": 1
}"""
    assert detect_filter(text) == "json_blob"


def test_markdown_doc_detected_only_when_long():
    from autofilter import detect_filter
    short_md = "# Title\n\n- item 1\n- item 2\n\nSome text here."
    long_md = "# Title\n\n" + ("## Section\n\nParagraph with text.\n\n- item\n- item\n\n```code```\n\n" * 10)
    assert detect_filter(short_md) is None  # too short
    assert detect_filter(long_md) == "markdown_doc"


def test_decision_text_no_filter():
    """Real knowledge text should NOT trigger any filter."""
    from autofilter import detect_filter
    text = (
        "Decision: use PostgreSQL 18 for the orders database. "
        "Reason: UUID v7 primary keys give time-ordered IDs, JSONB with GIN "
        "index covers our flexible attribute storage, and CONCURRENTLY index "
        "creation lets us avoid locks on a 50M-row table."
    )
    assert detect_filter(text) is None
