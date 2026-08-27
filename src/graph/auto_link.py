"""Auto-link knowledge records to the graph on save.

Called after every memory_save to link new knowledge to graph nodes.
Uses fast local extraction only -- must complete in <50ms.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any


def auto_link_knowledge(
    db: sqlite3.Connection,
    knowledge_id: int,
    content: str,
    project: str = "general",
    tags: list[str] | None = None,
) -> None:
    """Link a newly saved knowledge record to the graph.

    Extracts concepts via fast local matching against existing graph nodes,
    then links the knowledge record to those nodes plus project/tag nodes.

    Args:
        db: Active SQLite connection (with row_factory set).
        knowledge_id: The ID of the just-saved knowledge record.
        content: The knowledge content text.
        project: Project name (default "general").
        tags: List of tag strings.
    """
    try:
        from graph.store import GraphStore
        from ingestion.extractor import shared_extractor

        gs = GraphStore(db)
        # Reused, not constructed: a fresh extractor drops the node-name cache,
        # and this runs on every save.
        ex = shared_extractor(db)

        # Fast extraction only -- no Ollama call
        result = ex.extract_fast(content)

        # Link extracted concepts
        for concept in result.get("concepts", []):
            name = concept.get("name", "") if isinstance(concept, dict) else str(concept)
            if not name or len(name) < 2:
                continue
            strength = float(concept.get("strength", 0.8)) if isinstance(concept, dict) else 0.8
            node_id = concept.get("id") if isinstance(concept, dict) else None
            if not node_id:
                node_id = gs.get_or_create(name.lower(), "concept")
            gs.link_knowledge(knowledge_id, node_id, role="provides", strength=strength)

        # Link extracted entities
        for entity in result.get("entities", []):
            name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
            etype = entity.get("type", "concept") if isinstance(entity, dict) else "concept"
            if not name or len(name) < 2:
                continue
            node_id = entity.get("id") if isinstance(entity, dict) else None
            if not node_id:
                node_id = gs.get_or_create(name.lower(), etype)
            gs.link_knowledge(knowledge_id, node_id, role="mentions")

        # Link to project node
        if project and project != "general":
            proj_id = gs.get_or_create(project, "project")
            gs.link_knowledge(knowledge_id, proj_id, role="belongs_to")

        # Link to tag nodes
        if tags:
            for tag in tags:
                if isinstance(tag, str) and len(tag) > 2:
                    tag_id = gs.get_or_create(tag.lower().replace("-", "_"), "concept")
                    gs.link_knowledge(knowledge_id, tag_id, role="tagged")

        db.commit()

    except Exception as e:
        sys.stderr.write(f"[auto-link] Error linking knowledge {knowledge_id}: {e}\n")
