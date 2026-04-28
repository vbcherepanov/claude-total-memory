"""v11.0 Phase 3 — Storage facade.

Re-exports `server.Store` under a stable, narrow import path so the rest
of memory_core/ never has to reach into `server`. This is intentionally a
re-export, not a wrapper: the legacy Store is a 1500-line god-object whose
extraction into a real Repository pattern is scheduled for Phase 5.

Importing this module is cheap — it lazily resolves `Store` on first
attribute access so `memory_core` can be imported without paying the cost
of loading server.py (chromadb, fastembed, etc.) until needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Make `src/` importable when the package is loaded via "memory_core."
_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _resolve_store_cls() -> type:
    """Lazy-import `server.Store` only when first needed."""
    from server import Store as _Store  # noqa: WPS433 — deliberate lazy import
    return _Store


class _LazyStoreProxy:
    """Module-level handle that defers `from server import Store` until use.

    This keeps `from memory_core.storage import Store` cheap at import
    time. The proxy supports `Store(...)` instantiation and `isinstance`
    checks against the real class.
    """

    __slots__ = ("_cls",)

    def __init__(self) -> None:
        self._cls: type | None = None

    def _ensure(self) -> type:
        if self._cls is None:
            self._cls = _resolve_store_cls()
        return self._cls

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: ANN401
        return self._ensure()(*args, **kwargs)

    def __instancecheck__(self, instance: Any) -> bool:  # noqa: ANN401
        return isinstance(instance, self._ensure())

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._ensure(), name)


# Public name — call sites: `from memory_core.storage import Store`.
Store = _LazyStoreProxy()


__all__ = ["Store"]
