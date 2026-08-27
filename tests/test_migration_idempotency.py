"""Migration runner must survive columns that already exist.

SQLite has no ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``. When a database
already carries a column a migration adds — a restored backup, a DB whose
`migrations` tracker was reset, a column added out-of-band — `executescript`
aborts on the first statement. Because it is all-or-nothing, the CREATE INDEX
statements after it never run and, since the migration is never recorded, the
same failure repeats on every single startup.

The runner therefore replays such a script statement by statement, skipping
only the redundant ALTERs, and then records the migration as applied.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import server  # noqa: E402


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A real Store on a throwaway memory dir.

    The base tables are created by `Store._create_tables`, not by a migration,
    so the migrator can only be exercised on top of a fully constructed Store.
    """
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        yield store
    finally:
        try:
            store.db.close()
        except Exception:
            pass


def _applied(runner) -> set[str]:
    return {
        r[0] for r in runner.db.execute("SELECT version FROM migrations").fetchall()
    }


def test_all_migrations_apply_on_a_fresh_database(runner):
    runner._apply_sql_migrations()
    versions = _applied(runner)
    on_disk = {p.stem.split("_", 1)[0] for p in (ROOT / "migrations").glob("*.sql")}
    assert versions == on_disk, f"unapplied: {sorted(on_disk - versions)}"


def test_rerunning_is_a_no_op(runner):
    runner._apply_sql_migrations()
    first = _applied(runner)
    runner._apply_sql_migrations()
    assert _applied(runner) == first


def test_preexisting_column_does_not_wedge_the_migration(runner):
    """The 028 lineage case: agent_id already there, tracker empty."""
    runner._apply_sql_migrations()
    runner.db.execute("DELETE FROM migrations WHERE version = '028'")
    runner.db.commit()
    assert "028" not in _applied(runner)

    runner._apply_sql_migrations()

    assert "028" in _applied(runner), "migration stayed unrecorded — will retry forever"
    cols = {r[1] for r in runner.db.execute("PRAGMA table_info(knowledge)").fetchall()}
    assert {"agent_id", "parent_agent_id"} <= cols
    indexes = {
        r[0]
        for r in runner.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    # The statements that follow the failing ALTER must still have run.
    assert {"idx_k_agent_id", "idx_k_parent_agent_id"} <= indexes


def test_a_genuinely_broken_migration_is_not_recorded(runner, tmp_path):
    """Only duplicate-column errors are tolerated; real errors still retry."""
    ok = runner._replay_migration_skipping_existing(
        "ALTER TABLE knowledge ADD COLUMN zzz TEXT;"
        "SELECT * FROM a_table_that_does_not_exist;",
        "999",
    )
    assert ok is False


def test_no_migration_fails_on_a_fresh_database(tmp_path, monkeypatch, capsys):
    """A clean install must apply every migration without a single failure.

    Regression for the ordering defect @juicetin reported in #12: `_migrate()`
    added the subagent-lineage columns before `_apply_sql_migrations()` ran
    `028_agent_lineage.sql`, so 028 hit "duplicate column name: agent_id" on
    every fresh database, aborted before its CREATE INDEX statements, and was
    never recorded — meaning it retried forever.
    """
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        logged = capsys.readouterr().err
        assert "failed" not in logged.lower(), f"a migration failed on a fresh DB:\n{logged}"

        applied = {
            r[0] for r in store.db.execute("SELECT version FROM migrations").fetchall()
        }
        on_disk = {p.stem.split("_", 1)[0] for p in (ROOT / "migrations").glob("*.sql")}
        assert applied == on_disk
    finally:
        store.db.close()


def test_lineage_columns_have_exactly_one_owner(tmp_path, monkeypatch):
    """028 owns agent_id/parent_agent_id; `_migrate()` must not also add them."""
    migrate_src = inspect.getsource(server.Store._migrate)
    assert "ADD COLUMN agent_id" not in migrate_src, (
        "_migrate() is adding a column that migration 028 also adds"
    )
    assert "ADD COLUMN parent_agent_id" not in migrate_src

    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    store = server.Store()
    try:
        cols = {r[1] for r in store.db.execute("PRAGMA table_info(knowledge)").fetchall()}
        assert {"agent_id", "parent_agent_id"} <= cols, "028 did not create them"
        indexes = {
            r[0]
            for r in store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert {"idx_k_agent_id", "idx_k_parent_agent_id"} <= indexes
    finally:
        store.db.close()


def test_no_column_is_added_by_both_python_and_sql_migrations():
    """Catch the next instance of the same class of bug."""
    migrate_src = inspect.getsource(server.Store._migrate)
    python_cols = set(re.findall(r"ADD COLUMN (\w+)", migrate_src))
    sql_cols: set[str] = set()
    for path in (ROOT / "migrations").glob("*.sql"):
        sql_cols |= set(re.findall(r"ADD COLUMN (\w+)", path.read_text()))

    overlap = python_cols & sql_cols
    assert not overlap, (
        f"{sorted(overlap)} added by both Store._migrate() and a SQL migration — "
        "whichever runs second will fail on a fresh database"
    )
