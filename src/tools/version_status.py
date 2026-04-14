"""Show the current version state of the install — code + DB schema.

Used by:
  - update.sh banner (before/after)
  - operator diagnostics
  - dashboard /api/v6/version

Reports:
  - code_version       — from src/version.py
  - applied_migrations — already in `migrations` table, in order
  - pending_migrations — files in migrations/*.sql with a version > last applied
  - db_path / db_size_mb
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def code_version() -> str:
    try:
        from version import VERSION
        return VERSION
    except Exception:
        return "unknown"


def code_release_date() -> str:
    try:
        from version import RELEASE_DATE
        return RELEASE_DATE
    except Exception:
        return ""


def discovered_migrations() -> list[tuple[str, str]]:
    """All SQL migration files on disk, sorted by version prefix."""
    out: list[tuple[str, str]] = []
    md = ROOT / "migrations"
    if not md.is_dir():
        return out
    for p in sorted(md.glob("*.sql")):
        stem = p.stem
        version = stem.split("_", 1)[0]
        desc = stem[len(version) + 1:].replace("_", " ") or stem
        out.append((version, desc))
    return out


def applied_migrations(db: sqlite3.Connection) -> list[dict]:
    try:
        rows = db.execute(
            "SELECT version, description, applied_at FROM migrations ORDER BY version"
        ).fetchall()
        return [dict(version=r[0], description=r[1], applied_at=r[2]) for r in rows]
    except sqlite3.Error:
        return []


def get_status(db_path: Path | None = None) -> dict[str, Any]:
    if db_path is None:
        memory_dir = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
        db_path = memory_dir / "memory.db"

    on_disk = discovered_migrations()
    on_disk_versions = [v for v, _ in on_disk]

    applied: list[dict] = []
    db_size_mb = 0.0

    if db_path.exists():
        db_size_mb = round(db_path.stat().st_size / 1048576, 2)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        applied = applied_migrations(conn)
        conn.close()

    applied_set = {a["version"] for a in applied}
    pending = [
        {"version": v, "description": d}
        for v, d in on_disk if v not in applied_set
    ]

    return {
        "code_version": code_version(),
        "release_date": code_release_date(),
        "db_path": str(db_path),
        "db_size_mb": db_size_mb,
        "schema": {
            "applied": applied,
            "applied_count": len(applied),
            "pending": pending,
            "pending_count": len(pending),
            "all_known": on_disk_versions,
        },
    }


def _print_human(s: dict) -> None:
    print(f"\n  claude-total-memory v{s['code_version']}  ({s['release_date']})")
    print(f"  DB: {s['db_path']}  ({s['db_size_mb']} MB)")
    print()
    a = s["schema"]["applied"]
    p = s["schema"]["pending"]
    if a:
        print(f"  Applied migrations ({len(a)}):")
        for r in a:
            print(f"    ✓ {r['version']} — {r['description']}  (at {r['applied_at']})")
    else:
        print("  No migrations applied yet (fresh install).")
    if p:
        print(f"\n  Pending migrations ({len(p)}):")
        for r in p:
            print(f"    ✗ {r['version']} — {r['description']}")
        print(f"\n  Run `bash update.sh` to apply.")
    else:
        print(f"\n  Schema up to date.")
    print()


def main() -> int:
    json_mode = "--json" in sys.argv
    s = get_status()
    if json_mode:
        print(json.dumps(s, indent=2))
    else:
        _print_human(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
