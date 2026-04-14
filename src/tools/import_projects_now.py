"""One-shot project importer — walks given paths, summarizes each subproject,
saves to memory.

For each first-level subdir of each input path, gathers:
  - README.md / Readme.md / readme.md (first 800 chars)
  - Tech stack indicators: package.json, pyproject.toml, composer.json,
    go.mod, Cargo.toml, build.gradle, *.sln, install.json (Bitrix)
  - CLAUDE.md if present (first 600 chars)
  - Top-level structure (first 20 entries)

Writes one knowledge record per subproject (type='fact', tags include
detected stack + 'project-import' + parent dir name).

Usage:
    ~/claude-memory-server/.venv/bin/python src/tools/import_projects_now.py \\
      ~/PROJECT ~/VBCHEREPANOV ~/BITRIX-PHP-SCRIPT [...] [--dry-run]
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))


def _log(msg: str) -> None:
    sys.stderr.write(f"[import-projects] {msg}\n")
    sys.stderr.flush()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────
# Tech-stack detection
# ────────────────────────────────────────────────

STACK_FILES = {
    "package.json":    ("javascript", "node"),
    "pyproject.toml":  ("python",),
    "requirements.txt":("python",),
    "composer.json":   ("php",),
    "go.mod":          ("go",),
    "Cargo.toml":      ("rust",),
    "Gemfile":         ("ruby",),
    "build.gradle":    ("java", "gradle"),
    "pom.xml":         ("java", "maven"),
    "Pipfile":         ("python",),
    "mix.exs":         ("elixir",),
    "Makefile":        ("makefile",),
    "Dockerfile":      ("docker",),
    "docker-compose.yml": ("docker",),
    "docker-compose.yaml": ("docker",),
    ".csproj":         ("dotnet", "csharp"),
    ".sln":            ("dotnet",),
    "install.json":    ("bitrix",),
    "version.php":     ("bitrix", "php"),
    "pubspec.yaml":    ("dart", "flutter"),
    "Podfile":         ("ios", "swift"),
}


def detect_stack(root: Path) -> set[str]:
    found: set[str] = set()
    try:
        names = {p.name for p in root.iterdir() if p.is_file()}
    except (OSError, PermissionError):
        return found
    for marker, tags in STACK_FILES.items():
        if marker in names:
            found.update(tags)
        elif marker.startswith(".") and any(n.endswith(marker) for n in names):
            found.update(tags)
    # Bitrix module-style detection: install.json + lib/
    if (root / "install").is_dir() and (root / "lib").is_dir():
        found.add("bitrix")
    return found


def read_first(path: Path, n: int = 800) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:n].strip()
    except Exception:
        return ""


def find_first(root: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = root / c
        if p.is_file():
            return p
    return None


def list_top(root: Path, limit: int = 20) -> list[str]:
    try:
        items = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except (OSError, PermissionError):
        return []
    out: list[str] = []
    for p in items[:limit]:
        if p.name.startswith("."):
            continue
        out.append(p.name + ("/" if p.is_dir() else ""))
    return out


# ────────────────────────────────────────────────
# Summary builder
# ────────────────────────────────────────────────


SKIP_DIRS = {
    "node_modules", ".git", ".venv", "venv", "__pycache__", ".idea",
    "vendor", "target", "build", "dist", ".pytest_cache", ".cache",
    "coverage", ".next", ".nuxt", ".gradle", "Pods", ".dart_tool",
}


def summarize_project(root: Path, parent_label: str) -> dict | None:
    """Return a dict ready for memory_save, or None if uninteresting."""
    name = root.name
    if name in SKIP_DIRS or name.startswith("."):
        return None

    stack = detect_stack(root)
    readme = find_first(root, ["README.md", "Readme.md", "readme.md", "README"])
    claude_md = root / "CLAUDE.md"
    structure = list_top(root)

    # Skip if no signal: no README, no manifest, no CLAUDE.md, almost empty
    has_signal = bool(readme) or bool(stack) or claude_md.is_file() or len(structure) >= 5
    if not has_signal:
        return None

    parts: list[str] = [f"# Project: {name}", ""]
    parts.append(f"Path: {root}")
    parts.append(f"Parent: {parent_label}")
    if stack:
        parts.append(f"Stack detected: {', '.join(sorted(stack))}")
    parts.append("")

    if readme:
        head = read_first(readme, 800)
        if head:
            parts.append("## README excerpt")
            parts.append(head)
            parts.append("")

    if claude_md.is_file():
        cm = read_first(claude_md, 600)
        if cm:
            parts.append("## CLAUDE.md")
            parts.append(cm)
            parts.append("")

    if structure:
        parts.append("## Top-level structure")
        for entry in structure:
            parts.append(f"- {entry}")

    content = "\n".join(parts).strip()
    tags = ["project-import", parent_label.lower().replace(" ", "_"), name.lower()]
    tags.extend(sorted(stack))

    return {
        "content": content[:6000],   # cap to keep DB row reasonable
        "type": "fact",
        "project": name,
        "tags": tags[:12],
    }


# ────────────────────────────────────────────────
# Bulk save (direct SQLite — no MCP roundtrip per record)
# ────────────────────────────────────────────────


def bulk_save(records: list[dict], session_id: str) -> int:
    """Insert directly. Mirrors server.Store.save_knowledge minus dedup so
    we can do hundreds in seconds. Returns number inserted."""
    memory_dir = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
    db_path = memory_dir / "memory.db"
    if not db_path.exists():
        _log(f"db not found at {db_path}")
        return 0

    db = sqlite3.connect(str(db_path), timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")

    # Ensure session row exists
    db.execute(
        "INSERT OR IGNORE INTO sessions (id, started_at, project, status) "
        "VALUES (?, ?, ?, 'open')",
        (session_id, _now(), "project-import"),
    )

    inserted = 0
    for r in records:
        try:
            cur = db.execute(
                """INSERT INTO knowledge
                     (session_id, type, content, context, project, tags, source,
                      confidence, created_at, last_confirmed)
                   VALUES (?, ?, ?, ?, ?, ?, 'project-import', 0.9, ?, ?)""",
                (
                    session_id, r["type"], r["content"], "",
                    r["project"], json.dumps(r["tags"]),
                    _now(), _now(),
                ),
            )
            kid = cur.lastrowid

            # Enqueue in v6 queues so reflection picks it up later
            for tbl in ("triple_extraction_queue", "deep_enrichment_queue",
                        "representations_queue"):
                try:
                    db.execute(
                        f"INSERT INTO {tbl} (knowledge_id, status, created_at) "
                        f"VALUES (?, 'pending', ?)",
                        (kid, _now()),
                    )
                except sqlite3.IntegrityError:
                    pass  # already pending

            inserted += 1
        except Exception as e:
            _log(f"insert failed for {r['project']}: {e}")

    db.commit()
    db.close()

    # Touch the reflection trigger so the queues drain ASAP
    try:
        (memory_dir / ".reflect-pending").touch()
    except Exception:
        pass

    return inserted


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────


def walk_paths(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for root in paths:
        if not root.is_dir():
            _log(f"skip (not a dir): {root}")
            continue
        parent_label = root.name
        _log(f"scanning {root}")
        n_added = 0
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            rec = summarize_project(child, parent_label)
            if rec:
                out.append(rec)
                n_added += 1
        _log(f"  {n_added} projects from {root.name}")
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    dry_run = "--dry-run" in args
    paths = [Path(a).expanduser() for a in args if not a.startswith("--")]

    t0 = time.time()
    records = walk_paths(paths)
    _log(f"collected {len(records)} project summaries in {time.time()-t0:.1f}s")

    if not records:
        _log("nothing to import")
        return 0

    # Show preview
    for r in records[:3]:
        _log(f"  preview: {r['project']} (stack: {[t for t in r['tags'] if t not in ('project-import',)][:5]})")
    if len(records) > 3:
        _log(f"  …and {len(records) - 3} more")

    if dry_run:
        _log("dry-run — not saving")
        return 0

    session_id = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _log(f"saving as session {session_id}")
    n = bulk_save(records, session_id)
    _log(f"inserted {n} records, enqueued for reflection drain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
