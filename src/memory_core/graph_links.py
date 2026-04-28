"""v11.0 Phase 3 — Deterministic graph link helper.

Wraps `graph.auto_link.auto_link_knowledge` with an explicit, narrow
contract — only deterministic relation types are emitted from the hot
path. The legacy auto_link funnels through `ConceptExtractor.extract_fast`
which is already LLM-free, but it can pull in `extract_deep` when
called incorrectly. This wrapper guarantees we never go down that road.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# Allowed relation types in the hot path. Anything else has to wait for
# the async enrichment worker.
_ALLOWED_RELATIONS = frozenset(
    {
        "parent",       # chunk → parent knowledge record
        "next",         # chunk_n → chunk_{n+1} (sequence inside same source)
        "from_file",    # knowledge → file_path node
        "defines",      # knowledge → symbol_name node
        "belongs_to",   # knowledge → project node
        "tagged",       # knowledge → tag node
        "mentions",     # knowledge → entity (fast match only)
        "provides",     # knowledge → concept (fast match only)
    }
)


def create_base_links(
    db_or_store: Any,
    knowledge_id: int,
    *,
    parent_id: Optional[int] = None,
    file_path: Optional[str] = None,
    symbol_name: Optional[str] = None,
    prev_id: Optional[int] = None,
    project: Optional[str] = None,
    tags: Optional[list[str]] = None,
    content: Optional[str] = None,
) -> list[str]:
    """Create the deterministic base set of graph links.

    Returns a list of created relation labels (strings drawn from
    `_ALLOWED_RELATIONS`). Failures are swallowed and logged via the
    underlying graph layer — this helper must never raise into save.
    """
    if knowledge_id is None:
        return []
    created: list[str] = []

    db = getattr(db_or_store, "db", db_or_store)

    # Lazy imports — keep memory_core import-cheap.
    try:
        from graph.store import GraphStore  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return created

    try:
        gs = GraphStore(db)
    except Exception:  # noqa: BLE001
        return created

    # 1. parent / sibling chain inside a single ingestion source.
    if parent_id:
        try:
            gs.link_knowledge(knowledge_id, _kid_to_node(gs, parent_id), role="parent")
            created.append("parent")
        except Exception:  # noqa: BLE001
            pass

    if prev_id:
        try:
            gs.link_knowledge(knowledge_id, _kid_to_node(gs, prev_id), role="next")
            created.append("next")
        except Exception:  # noqa: BLE001
            pass

    # 2. file_path / symbol_name — useful when ingesting source code.
    if file_path:
        try:
            node_id = gs.get_or_create(file_path, "file")
            gs.link_knowledge(knowledge_id, node_id, role="from_file")
            created.append("from_file")
        except Exception:  # noqa: BLE001
            pass

    if symbol_name:
        try:
            node_id = gs.get_or_create(symbol_name.lower(), "symbol")
            gs.link_knowledge(knowledge_id, node_id, role="defines")
            created.append("defines")
        except Exception:  # noqa: BLE001
            pass

    # 3. project / tags via the existing auto_link pipeline (fast extract only).
    if content is not None:
        try:
            from graph.auto_link import auto_link_knowledge  # noqa: WPS433

            auto_link_knowledge(
                db,
                knowledge_id=knowledge_id,
                content=content,
                project=project or "general",
                tags=tags or [],
            )
            if project and project != "general":
                created.append("belongs_to")
            if tags:
                created.append("tagged")
        except Exception:  # noqa: BLE001
            pass

    return [r for r in created if r in _ALLOWED_RELATIONS]


def _kid_to_node(gs: Any, knowledge_id: int) -> str:  # noqa: ANN401
    """Map a knowledge row to a stable graph node id ("knowledge:<id>")."""
    return gs.get_or_create(f"knowledge:{int(knowledge_id)}", "knowledge")


__all__ = ["create_base_links"]
