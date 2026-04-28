"""v11.0 Phase 3 — Vector store abstraction.

Two adapters live here:

* :class:`ChromaVectorStore` — wraps an existing chromadb collection (the
  default v10.x backend). Chroma metadata fields include `embedding_space`
  so we can `where={"embedding_space": {"$in": [...]}}` filter.
* :class:`SQLiteBinaryVectorStore` — thin shim that delegates to the
  legacy `Store._binary_search` ladder. Phase 3 just keeps the interface
  uniform; full extraction is a Phase 5 job.

Both adapters honour the `embedding_space` filter so the multi-space
contract (§J) survives every search path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ─── records & protocol ──────────────────────────────────────────────


@dataclass
class VectorRecord:
    """One row that travels through the vector store."""

    id: str
    vector: list[float]
    content_type: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    embedding_space: str
    project: str
    ktype: str
    created_at: str
    language: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def chroma_metadata(self) -> dict[str, Any]:
        """Return a Chroma-compatible flat metadata dict (no nested objects)."""
        meta: dict[str, Any] = {
            "content_type": self.content_type,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": int(self.embedding_dimension),
            "embedding_space": self.embedding_space,
            "project": self.project,
            "ktype": self.ktype,
            "created_at": self.created_at,
        }
        if self.language:
            meta["language"] = self.language
        # Chroma rejects nested values; flatten by string-coercing leftovers.
        # `document` is passed separately to Chroma — keep it out of metadata.
        for k, v in self.metadata.items():
            if k == "document":
                continue
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            elif v is None:
                continue
            else:
                meta[k] = str(v)
        return meta


class VectorStore(Protocol):
    """Hot-path vector backend contract."""

    def add(self, records: list[VectorRecord]) -> None: ...

    def search(
        self,
        query_vec: list[float],
        *,
        top_k: int,
        embedding_space: str | Iterable[str] | None = None,
        project: str | None = None,
    ) -> list[tuple[str, float]]: ...

    def delete(self, ids: list[str]) -> None: ...


# ─── helpers ─────────────────────────────────────────────────────────


def _as_space_filter(space: str | Iterable[str] | None) -> list[str] | None:
    if space is None:
        return None
    if isinstance(space, str):
        space = [space]
    out = [s for s in (s.strip().lower() for s in space) if s]
    return out or None


# ─── ChromaDB adapter ────────────────────────────────────────────────


class ChromaVectorStore:
    """Adapter around an existing chromadb Collection.

    The collection is owned by the legacy `Store` and injected here so
    Phase 3 doesn't have to take ownership of chroma init. Phase 5 will
    push the init down to memory_core/.
    """

    backend = "chroma"

    def __init__(self, collection: Any) -> None:  # noqa: ANN401 — chromadb type
        if collection is None:
            raise ValueError("ChromaVectorStore requires a non-null chromadb collection")
        self._col = collection

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._col.upsert(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            metadatas=[r.chroma_metadata() for r in records],
            documents=[r.metadata.get("document", "") for r in records],
        )

    def search(
        self,
        query_vec: list[float],
        *,
        top_k: int,
        embedding_space: str | Iterable[str] | None = None,
        project: str | None = None,
    ) -> list[tuple[str, float]]:
        if not query_vec:
            return []
        where: dict[str, Any] = {}
        spaces = _as_space_filter(embedding_space)
        if spaces:
            where["embedding_space"] = {"$in": spaces}
        if project:
            where["project"] = project
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": int(top_k),
        }
        if where:
            kwargs["where"] = where
        res = self._col.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        # Chroma returns distances (lower = better); convert to similarity
        # so the rest of the pipeline can treat it like cosine score.
        dists = (res.get("distances") or [[0.0] * len(ids)])[0]
        return [(str(i), 1.0 - float(d)) for i, d in zip(ids, dists)]

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._col.delete(ids=list(ids))


# ─── SQLite binary adapter (Phase 3 shim) ────────────────────────────


class SQLiteBinaryVectorStore:
    """Phase 3 shim over `Store._binary_search` and the `embeddings` table.

    Search delegates to the injected store; the only logic this class
    owns is the `embedding_space` filter. We pre-resolve eligible
    knowledge ids via SQL, then ask the store for raw binary scores
    and intersect. Phase 5 will replace this with a native binary HNSW.
    """

    backend = "sqlite-binary"

    def __init__(self, store: Any) -> None:  # noqa: ANN401 — server.Store
        if store is None:
            raise ValueError("SQLiteBinaryVectorStore requires a Store instance")
        self._store = store

    # add()/delete() in v11 still go through Store._upsert_embedding because
    # binary blobs and the Chroma path are written together. This shim only
    # owns *reads* in Phase 3.

    def add(self, records: list[VectorRecord]) -> None:  # pragma: no cover — Phase 5
        raise NotImplementedError(
            "SQLiteBinaryVectorStore.add — Phase 5 will own writes; "
            "v11 still routes saves through Store._upsert_embedding."
        )

    def delete(self, ids: list[str]) -> None:  # pragma: no cover — Phase 5
        raise NotImplementedError(
            "SQLiteBinaryVectorStore.delete — Phase 5 will own writes."
        )

    def search(
        self,
        query_vec: list[float],
        *,
        top_k: int,
        embedding_space: str | Iterable[str] | None = None,
        project: str | None = None,
    ) -> list[tuple[str, float]]:
        if not query_vec:
            return []
        # Step 1 — ask the store for raw binary candidates.
        raw = self._store._binary_search(  # noqa: SLF001 — intentional shim
            query_vec,
            n_candidates=max(top_k * 5, 50),
            project=project,
            n_results=max(top_k * 2, top_k),
        )
        spaces = _as_space_filter(embedding_space)
        if not spaces:
            return [
                (str(row["id"]), float(row.get("score") or 0.0))
                for row in raw[: top_k]
            ]

        # Step 2 — intersect with embeddings.embedding_space filter.
        ids = [row["id"] for row in raw]
        if not ids:
            return []
        placeholder = ",".join(["?"] * len(ids))
        space_placeholder = ",".join(["?"] * len(spaces))
        rows = self._store.q(
            f"""
            SELECT knowledge_id FROM embeddings
            WHERE knowledge_id IN ({placeholder})
              AND embedding_space IN ({space_placeholder})
            """,
            (*ids, *spaces),
        )
        eligible = {r["knowledge_id"] for r in rows}
        out: list[tuple[str, float]] = []
        for row in raw:
            if row["id"] in eligible:
                out.append((str(row["id"]), float(row.get("score") or 0.0)))
            if len(out) >= top_k:
                break
        return out


__all__ = [
    "VectorRecord",
    "VectorStore",
    "ChromaVectorStore",
    "SQLiteBinaryVectorStore",
]
