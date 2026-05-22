from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


SRC = str(Path(__file__).parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _text_json(result):
    return json.loads(result[0].text)


def test_v11_w3_dispatch_passes_call_args(monkeypatch):
    import server
    import v11_handlers

    class DummyRecall:
        def search(self, q, project, ktype, limit=10):
            return {
                "q": q,
                "project": project,
                "type": ktype,
                "limit": limit,
            }

    class DummyStore:
        db = object()

        def raw_append(self, *args, **kwargs):
            return None

        def embed(self, texts):
            return [[float(len(texts[0]))]]

    seen = {}

    def fake_recall_iterative(args, search_fn):
        seen["iterative"] = dict(args)
        return {"tool": "iterative", "search": search_fn("needle", "proj", "all", 3)}

    def fake_temporal(args):
        seen["temporal"] = dict(args)
        return {"tool": "temporal", "op": args["op"]}

    def fake_entity(args, conn, embed_fn):
        seen["entity"] = dict(args)
        return {
            "tool": "entity",
            "mention": args["mention"],
            "has_conn": conn is not None,
            "embedding": embed_fn("abc"),
        }

    def fake_consolidate_status(args, conn):
        seen["status"] = dict(args)
        return {"tool": "status", "project": args["project"], "has_conn": conn is not None}

    monkeypatch.setattr(server, "SID", "test-session", raising=False)
    monkeypatch.setattr(server, "store", DummyStore(), raising=False)
    monkeypatch.setattr(server, "recall", DummyRecall(), raising=False)
    monkeypatch.setattr(v11_handlers, "handle_recall_iterative", fake_recall_iterative)
    monkeypatch.setattr(v11_handlers, "handle_temporal_query", fake_temporal)
    monkeypatch.setattr(v11_handlers, "handle_entity_resolve", fake_entity)
    monkeypatch.setattr(v11_handlers, "handle_consolidate_status", fake_consolidate_status)

    async def run():
        return [
            _text_json(await server.call_tool("memory_recall_iterative", {"query": "needle"})),
            _text_json(await server.call_tool("memory_temporal_query", {"op": "normalize"})),
            _text_json(await server.call_tool("memory_entity_resolve", {"mention": "px"})),
            _text_json(await server.call_tool("memory_consolidate_status", {"project": "Dev"})),
        ]

    iterative, temporal, entity, status = asyncio.run(run())

    assert iterative["tool"] == "iterative"
    assert iterative["search"]["limit"] == 3
    assert temporal == {"tool": "temporal", "op": "normalize"}
    assert entity == {
        "tool": "entity",
        "mention": "px",
        "has_conn": True,
        "embedding": [3.0],
    }
    assert status == {"tool": "status", "project": "Dev", "has_conn": True}

    assert seen == {
        "iterative": {"query": "needle"},
        "temporal": {"op": "normalize"},
        "entity": {"mention": "px"},
        "status": {"project": "Dev"},
    }
