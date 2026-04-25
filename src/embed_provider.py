"""Pluggable embedding provider abstraction.

Scaffolding only. The existing FastEmbed flow inside src/server.py stays
put — this module just exposes it behind a stable Protocol and adds
OpenAI/Cohere cloud backends. Wiring into server.py comes later.

Providers:
  - FastEmbedProvider   — local (fastembed lib), no HTTP.
  - OpenAIEmbedProvider — POST {base}/embeddings; supports any
    OpenAI-compatible embed endpoint (OpenRouter, LiteLLM, LM Studio).
  - CohereEmbedProvider — POST {base}/embed (v2 API).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Protocol, Sequence, runtime_checkable

import config

LOG = lambda msg: sys.stderr.write(f"[embed-provider] {msg}\n")


# ──────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""
        ...

    def dim(self) -> int:
        """Vector dimensionality. May return 0 if unknown until first call."""
        ...

    def available(self) -> bool:
        ...


# ──────────────────────────────────────────────
# Known model → dimension table
# ──────────────────────────────────────────────
#
# Used to report `dim()` without forcing an actual request. Conservative —
# callers should treat 0 as "unknown, run one embed first".

_OPENAI_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_COHERE_DIM = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}

_FASTEMBED_DIM = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


# ──────────────────────────────────────────────
# HTTP helper (shared shape with llm_provider)
# ──────────────────────────────────────────────


def _ssl_context():
    """Return an SSL context that works on Python.org macOS installs.

    Those installs ship without system CAs, so urllib's default verify
    always fails on TLS endpoints. Prefer certifi when available; fall
    back to the platform default (e.g. Linux distros where CAs exist).
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_post_json(
    url: str,
    body: dict,
    headers: dict[str, str],
    timeout: float,
    retries: int = 4,
) -> dict:
    """POST JSON with exponential backoff retry on timeout / 5xx / network errors."""
    import time
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    last_exc: Exception | None = None
    ctx = _ssl_context()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read()
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            # Retry only on rate limit / server errors; raise on 4xx auth/quota.
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < retries:
                wait = 2 ** attempt
                LOG(f"HTTP {e.code} on {url} — retry in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                last_exc = e
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries:
                wait = 2 ** attempt
                LOG(f"Network error on {url}: {e} — retry in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                last_exc = e
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


# ──────────────────────────────────────────────
# FastEmbed (local)
# ──────────────────────────────────────────────


class FastEmbedProvider:
    """Thin wrapper around the existing FastEmbed init in server.Store.

    Kept intentionally small: lazy-loads the model, converts generators to
    plain lists-of-lists so the output shape matches the HTTP providers.
    """

    name = "fastembed"

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or config.get_embed_model("fastembed")
        self._model: object | None = None  # TextEmbedding instance or False
        self._dim_cache: int = _FASTEMBED_DIM.get(self._model_name, 0)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> object | None:
        if self._model is False:
            return None
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError:
            self._model = False
            return None
        try:
            self._model = TextEmbedding(self._model_name)
        except Exception as exc:  # noqa: BLE001
            LOG(f"FastEmbed init failed: {exc}")
            self._model = False
            return None
        return self._model

    def available(self) -> bool:
        return self._ensure_model() is not None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        if model is None:
            raise RuntimeError("FastEmbedProvider: model unavailable")
        # fastembed yields generator of numpy arrays
        out: list[list[float]] = []
        for vec in model.embed(list(texts)):  # type: ignore[attr-defined]
            if hasattr(vec, "tolist"):
                out.append(vec.tolist())
            else:
                out.append([float(x) for x in vec])
        if out and not self._dim_cache:
            self._dim_cache = len(out[0])
        return out

    def dim(self) -> int:
        return self._dim_cache


# ──────────────────────────────────────────────
# OpenAI (and OpenAI-compatible)
# ──────────────────────────────────────────────


class OpenAIEmbedProvider:
    """OpenAI embeddings API.

    Body schema: `{"input": [texts...], "model": model}`. Response:
    `{"data": [{"embedding": [...]}, ...]}`. Supports `text-embedding-3-small`
    (1536) and `text-embedding-3-large` (3072). `api_base` override lets this
    target LiteLLM / OpenRouter / self-hosted proxies.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = (api_base or config.get_embed_api_base("openai")).rstrip("/")
        self._model = model or config.get_embed_model("openai")

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        return bool(self.api_key) and bool(self.api_base)

    def dim(self) -> int:
        return _OPENAI_DIM.get(self._model, 0)

    def embed(self, texts: Sequence[str], *, timeout: float = 60.0) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OpenAIEmbedProvider: missing api_key")
        if not texts:
            return []
        body = {"input": list(texts), "model": self._model}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = _http_post_json(
            f"{self.api_base}/embeddings",
            body=body,
            headers=headers,
            timeout=timeout,
        )
        try:
            items = resp["data"]
            # Preserve input order via `index` when provided.
            ordered: list[list[float]] = [[] for _ in items]
            for entry in items:
                idx = int(entry.get("index", 0))
                ordered[idx] = [float(x) for x in entry["embedding"]]
            return ordered
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"OpenAIEmbedProvider: malformed response: {exc}") from exc


# ──────────────────────────────────────────────
# Cohere
# ──────────────────────────────────────────────


class CohereEmbedProvider:
    """Cohere v2 embeddings.

    Body: `{"texts": [...], "model": model, "input_type": "search_document"}`.
    Response: `{"embeddings": {"float": [[...], ...]}}` (v2) or
    `{"embeddings": [[...], ...]}` (legacy). We accept both shapes.
    """

    name = "cohere"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        input_type: str = "search_document",
    ) -> None:
        self.api_key = api_key
        self.api_base = (api_base or config.get_embed_api_base("cohere")).rstrip("/")
        self._model = model or config.get_embed_model("cohere")
        self.input_type = input_type

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        return bool(self.api_key) and bool(self.api_base)

    def dim(self) -> int:
        return _COHERE_DIM.get(self._model, 0)

    def embed(self, texts: Sequence[str], *, timeout: float = 30.0) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("CohereEmbedProvider: missing api_key")
        if not texts:
            return []
        body = {
            "texts": list(texts),
            "model": self._model,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = _http_post_json(
            f"{self.api_base}/embed",
            body=body,
            headers=headers,
            timeout=timeout,
        )
        raw = resp.get("embeddings")
        try:
            if isinstance(raw, dict):
                # v2 shape: {"float": [[...]]}
                vectors = raw.get("float") or raw.get("embeddings") or []
            elif isinstance(raw, list):
                vectors = raw
            else:
                vectors = []
            return [[float(x) for x in v] for v in vectors]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"CohereEmbedProvider: malformed response: {exc}") from exc


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def make_embed_provider(name: str, **kwargs) -> EmbeddingProvider:
    """Build an embedding provider by name.

    kwargs (optional): api_key, api_base, model. Missing values fall back
    to config-driven defaults.
    """
    key = (name or "").strip().lower()
    if key == "auto":
        key = config.get_embed_provider()

    if key == "fastembed":
        return FastEmbedProvider(model=kwargs.get("model"))
    if key == "openai":
        return OpenAIEmbedProvider(
            api_key=kwargs.get("api_key") or config.get_embed_api_key("openai"),
            api_base=kwargs.get("api_base") or config.get_embed_api_base("openai"),
            model=kwargs.get("model"),
        )
    if key == "cohere":
        return CohereEmbedProvider(
            api_key=kwargs.get("api_key") or config.get_embed_api_key("cohere"),
            api_base=kwargs.get("api_base") or config.get_embed_api_base("cohere"),
            model=kwargs.get("model"),
        )
    raise ValueError(
        f"unknown embedding provider {name!r}; expected fastembed|openai|cohere|auto"
    )
