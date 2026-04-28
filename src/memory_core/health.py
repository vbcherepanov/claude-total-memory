"""v11.0 Phase 3 — Health probe for the deterministic core.

Returns a snapshot of the runtime that the dashboard / `memory_stats`
tool can render without hitting the network. Every check here is a cheap
local probe; nothing in this module imports `llm_provider`.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as _cfg  # noqa: E402

from memory_core.embedding_spaces import SUPPORTED_SPACES  # noqa: E402


def _probe_fastembed() -> bool:
    try:
        from embed_provider import FastEmbedProvider  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return False
    try:
        provider = FastEmbedProvider()
        return bool(provider.available())
    except Exception:  # noqa: BLE001
        return False


def _probe_fts5() -> bool:
    """Check whether SQLite was built with FTS5. Cheap one-shot query."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE _t USING fts5(c)")
            return True
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return False


def _detect_vector_backend() -> str:
    """Return the active vector backend name."""
    use_binary = (os.environ.get("MEMORY_USE_BINARY_SEARCH", "") or "").strip().lower()
    if use_binary in ("1", "true", "yes", "on"):
        return "sqlite-binary"
    try:
        import chromadb  # noqa: F401, WPS433
        return "chroma"
    except Exception:  # noqa: BLE001
        return "sqlite-binary"


def health() -> dict:
    """Snapshot the deterministic core's readiness."""
    mode = _cfg.get_memory_mode()
    return {
        "fastembed_available": _probe_fastembed(),
        "fts5_available": _probe_fts5(),
        "vector_backend": _detect_vector_backend(),
        "embedding_spaces_supported": list(SUPPORTED_SPACES),
        "mode": mode,
        "allow_ollama_in_hot_path": _cfg.allow_ollama_in_hot_path(),
        "use_llm_in_hot_path": _cfg.use_llm_in_hot_path(),
    }


__all__ = ["health"]
