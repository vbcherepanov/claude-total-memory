"""v9.0 B1 — Embedding backend selector (A/B facade).

Returns an :class:`embed_provider.EmbeddingProvider`-compatible object based
on ``V9_EMBED_BACKEND`` env flag:

+-------------+--------------------------------------------------------------+
| Backend     | Underlying model                                             |
+=============+==============================================================+
| fastembed   | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  |
| minilm      | same as fastembed (explicit alias)                           |
| e5-large    | intfloat/multilingual-e5-large (via fastembed)               |
| bge-m3      | BAAI/bge-m3 (via sentence-transformers; fastembed has no M3) |
+-------------+--------------------------------------------------------------+

Invalid value → falls back to ``fastembed`` default.

The facade leaves :mod:`embed_provider` untouched. BGE-M3 uses a thin
wrapper around :mod:`sentence_transformers` because fastembed's model
catalog does not include M3.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402
import embed_provider as _ep  # noqa: E402


# ──────────────────────────────────────────────
# Backend → model name table
# ──────────────────────────────────────────────

BACKEND_MODEL: dict[str, str] = {
    "fastembed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    # v9 D5 — locally fine-tuned MiniLM. The actual filesystem path is
    # resolved at provider build time from config.get_v9_locomo_tuned_path().
    "locomo-tuned-minilm": "<resolved-from-config>",
    # v9 D1 — OpenAI cloud embeddings. Resolved to OpenAIEmbedProvider with
    # MEMORY_EMBED_API_KEY / MEMORY_EMBED_API_BASE.
    "openai-3-small": "text-embedding-3-small",
    "openai-3-large": "text-embedding-3-large",
}

# Known dimensionality (for dim() without forcing a model load).
BACKEND_DIM: dict[str, int] = {
    "fastembed": 384,
    "minilm": 384,
    "e5-large": 1024,
    "bge-m3": 1024,
    "locomo-tuned-minilm": 384,  # MiniLM-L12-v2 base preserves 384 dims.
    "openai-3-small": 1536,
    "openai-3-large": 3072,
}

# Backends that sentence-transformers has to handle directly
# (fastembed catalog does not contain them).
_ST_ONLY = {"bge-m3", "locomo-tuned-minilm"}

# Backends that go through OpenAIEmbedProvider (HTTP), not local libs.
_OPENAI_BACKENDS = {"openai-3-small", "openai-3-large"}


# ──────────────────────────────────────────────
# sentence-transformers wrapper
# ──────────────────────────────────────────────


class SentenceTransformersProvider:
    """Minimal ST-backed provider for models fastembed does not ship.

    Matches the :class:`embed_provider.EmbeddingProvider` protocol exactly —
    plain lists-of-lists output, lazy model load, graceful ``available()``.
    """

    name = "sentence-transformers"

    def __init__(self, model: str, *, expected_dim: int = 0) -> None:
        self._model_name = model
        self._model: object | None = None  # ST model instance or False
        self._dim_cache: int = expected_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self):
        if self._model is False:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError:
            self._model = False
            return None
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[choose-embed] ST load failed for {self._model_name}: {exc}\n")
            self._model = False
            return None
        return self._model

    def available(self) -> bool:
        return self._ensure_model() is not None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        if model is None:
            raise RuntimeError(
                f"SentenceTransformersProvider: model {self._model_name!r} unavailable"
            )
        vecs = model.encode(  # type: ignore[attr-defined]
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        out: list[list[float]] = []
        for v in vecs:
            if hasattr(v, "tolist"):
                out.append(v.tolist())
            else:
                out.append([float(x) for x in v])
        if out and not self._dim_cache:
            self._dim_cache = len(out[0])
        return out

    def dim(self) -> int:
        return self._dim_cache


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def resolve_backend(override: str | None = None) -> str:
    """Return normalized backend name. Invalid → 'fastembed'."""
    if override is not None:
        raw = (override or "").strip().lower()
    else:
        raw = config.get_v9_embed_backend()
    if raw in BACKEND_MODEL:
        return raw
    return "fastembed"


def resolve_model_name(backend: str | None = None) -> str:
    """Return HF model id (or local dir) for the given backend."""
    b = resolve_backend(backend)
    if b == "locomo-tuned-minilm":
        return config.get_v9_locomo_tuned_path()
    return BACKEND_MODEL[b]


def get_provider(backend: str | None = None):
    """Build an EmbeddingProvider for the requested (or env-selected) backend.

    Routing:
      * openai-3-small / openai-3-large → embed_provider.OpenAIEmbedProvider
        (HTTP; reads MEMORY_EMBED_API_KEY).
      * bge-m3 / locomo-tuned-minilm    → sentence-transformers (local dir or
        HF id) because fastembed's model catalog can't load those.
      * everything else                  → embed_provider.FastEmbedProvider.
    """
    b = resolve_backend(backend)
    model = resolve_model_name(b)
    if b in _OPENAI_BACKENDS:
        return _ep.OpenAIEmbedProvider(
            api_key=config.get_embed_api_key("openai"),
            api_base=config.get_embed_api_base("openai"),
            model=model,
        )
    if b in _ST_ONLY:
        return SentenceTransformersProvider(model=model, expected_dim=BACKEND_DIM.get(b, 0))
    return _ep.FastEmbedProvider(model=model)


def backend_dim(backend: str | None = None) -> int:
    b = resolve_backend(backend)
    return BACKEND_DIM.get(b, 0)


__all__ = [
    "BACKEND_MODEL",
    "BACKEND_DIM",
    "SentenceTransformersProvider",
    "resolve_backend",
    "resolve_model_name",
    "get_provider",
    "backend_dim",
]
