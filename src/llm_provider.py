"""Pluggable LLM provider abstraction.

Scaffolding only. Existing callers (deep_enricher, representations, ingestion
extractor, reflection agent) continue to call Ollama directly. Wiring them
to go through this module happens in a separate wave.

Why:
  - Support cloud LLMs (OpenAI-compatible + Anthropic) without forking call
    sites.
  - OpenAIProvider accepts `api_base`, so any OpenAI-compatible backend works
    unchanged: OpenRouter, Together, Groq, DeepSeek, LM Studio, llama.cpp.

Transport:
  - urllib only — no new runtime deps. This matches the rest of the
    codebase and keeps Dockerfile slim.

Design notes:
  - Each provider is a small class exposing the `LLMProvider` protocol below.
  - `make_provider(name, **overrides)` is a thin factory that reads env
    defaults from `config` and lets callers override per-call.
  - `complete()` errors are propagated; individual call-sites already wrap
    Ollama errors in try/except, so behavior parity is preserved once wired.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

import config

LOG = lambda msg: sys.stderr.write(f"[llm-provider] {msg}\n")


# ──────────────────────────────────────────────
# Availability probe cache
# ──────────────────────────────────────────────
# Short-TTL cache keyed by (provider_name, api_base, key_hash). Reuses the
# same TTL config.has_llm()/detect_ollama() use so env tuning is consistent.
_available_cache: dict[tuple[str, str, str], tuple[bool, float]] = {}


def _cache_key(provider: str, api_base: str, api_key: str | None) -> tuple[str, str, str]:
    key_hash = hashlib.sha1((api_key or "").encode("utf-8")).hexdigest()[:16]
    return (provider, api_base or "", key_hash)


def _cache_get_available(key: tuple[str, str, str]) -> bool | None:
    rec = _available_cache.get(key)
    if rec is None:
        return None
    value, expires_at = rec
    if time.time() >= expires_at:
        _available_cache.pop(key, None)
        return None
    return value


def _cache_set_available(key: tuple[str, str, str], value: bool) -> None:
    try:
        ttl = float(os.environ.get("MEMORY_LLM_PROBE_TTL_SEC", "60"))
    except ValueError:
        ttl = 60.0
    _available_cache[key] = (value, time.time() + ttl)


def _clear_available_cache() -> None:
    """Test helper — reset the availability cache."""
    _available_cache.clear()


# ──────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal contract every concrete provider implements."""

    name: str

    def available(self) -> bool:
        """Cheap check: is this provider usable right now (creds/model)."""
        ...

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> str:
        """Run a single completion. Returns the raw assistant text."""
        ...


# ──────────────────────────────────────────────
# HTTP helper
# ──────────────────────────────────────────────


def _http_post_json(
    url: str,
    body: dict,
    headers: dict[str, str],
    timeout: float,
) -> dict:
    """POST JSON, return parsed JSON. Raises urllib/JSON errors to caller.

    Uses certifi CA bundle when available (fixes SSL on Python 3.13 / macOS
    python.org builds that don't trust the system keychain).
    """
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    ctx = None
    if url.startswith("https://"):
        import ssl
        try:
            import certifi  # type: ignore
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
    return json.loads(raw)


# ──────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────


class OllamaProvider:
    """Wraps the existing `{OLLAMA_URL}/api/generate` flow.

    Preserves the option keys (`num_predict`, `temperature`) that the legacy
    call-sites use, so once we wire this in, behavior stays identical.
    """

    name = "ollama"

    def __init__(
        self,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_base = (api_base or config.get_ollama_url()).rstrip("/")
        # Default model deferred to call time so env changes are honored.
        self._default_model = model

    def available(self) -> bool:
        # Reuse the already-cached probe in config.has_llm() when possible.
        try:
            return config.has_llm()
        except Exception:  # noqa: BLE001
            return False

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> str:
        chosen_model = model or self._default_model or config.get_llm_model()
        body = {
            "model": chosen_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        resp = _http_post_json(
            f"{self.api_base}/api/generate",
            body=body,
            headers={},
            timeout=timeout,
        )
        return str(resp.get("response", "")).strip()


# ──────────────────────────────────────────────
# OpenAI-compatible (OpenAI, OpenRouter, Groq, Together, DeepSeek, LM Studio…)
# ──────────────────────────────────────────────


class OpenAIProvider:
    """OpenAI Chat Completions API.

    Works with any OpenAI-compatible endpoint by setting `api_base`:
      - OpenAI:      https://api.openai.com/v1
      - OpenRouter:  https://openrouter.ai/api/v1
      - Groq:        https://api.groq.com/openai/v1
      - Together:    https://api.together.xyz/v1
      - DeepSeek:    https://api.deepseek.com/v1
      - LM Studio:   http://localhost:1234/v1
      - llama.cpp:   http://localhost:8080/v1
    """

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = (api_base or config.get_llm_api_base("openai")).rstrip("/")
        self._default_model = model

    def available(self) -> bool:
        # Fast reject: no credentials at all.
        if not self.api_key or not self.api_base:
            return False
        key = _cache_key("openai", self.api_base, self.api_key)
        cached = _cache_get_available(key)
        if cached is not None:
            return cached
        # Probe `/models` with a short timeout. Network errors → False.
        url = f"{self.api_base}/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                # Fake responses in tests may not expose .status; treat any
                # non-raising response as a successful probe.
                status = getattr(resp, "status", None) or getattr(resp, "code", 200)
                ok = 200 <= int(status) < 300
        except urllib.error.HTTPError as exc:
            # 401 (bad key) / 403 / 404 all count as unreachable for our purpose.
            ok = 200 <= exc.code < 300
        except (urllib.error.URLError, OSError, TimeoutError):
            ok = False
        _cache_set_available(key, ok)
        return ok

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OpenAIProvider: missing api_key")
        chosen_model = model or self._default_model or config.get_llm_model_for_provider("openai")
        body = {
            "model": chosen_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = _http_post_json(
            f"{self.api_base}/chat/completions",
            body=body,
            headers=headers,
            timeout=timeout,
        )
        try:
            return str(resp["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"OpenAIProvider: malformed response: {exc}") from exc


# ──────────────────────────────────────────────
# Anthropic
# ──────────────────────────────────────────────


class AnthropicProvider:
    """Anthropic Messages API."""

    name = "anthropic"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = (api_base or config.get_llm_api_base("anthropic")).rstrip("/")
        self._default_model = model

    def available(self) -> bool:
        # Anthropic has no cheap probe endpoint (`/v1/messages` requires a
        # real body). Trust env: if creds are present, assume reachable and
        # let `complete()` surface genuine auth errors as runtime exceptions.
        if not self.api_key or not self.api_base:
            return False
        key = _cache_key("anthropic", self.api_base, self.api_key)
        cached = _cache_get_available(key)
        if cached is not None:
            return cached
        _cache_set_available(key, True)
        return True

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("AnthropicProvider: missing api_key")
        chosen_model = model or self._default_model or config.get_llm_model_for_provider("anthropic")
        body = {
            "model": chosen_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        resp = _http_post_json(
            f"{self.api_base}/messages",
            body=body,
            headers=headers,
            timeout=timeout,
        )
        try:
            content = resp["content"]
            # content is a list of blocks; pick first text block
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", "")).strip()
            # Fallback: old schema
            return str(content[0]["text"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"AnthropicProvider: malformed response: {exc}") from exc


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def make_provider(name: str, **kwargs) -> LLMProvider:
    """Build a provider by canonical name.

    Accepts optional kwargs: `api_key`, `api_base`, `model`. Missing values
    are filled from config/env, matching the rest of the module.

    `auto` resolves via the same logic as `config.get_llm_provider()`.
    """
    key = (name or "").strip().lower()
    if key == "auto":
        key = config.get_llm_provider()

    if key == "ollama":
        return OllamaProvider(
            api_base=kwargs.get("api_base"),
            model=kwargs.get("model"),
        )
    if key == "openai":
        return OpenAIProvider(
            api_key=kwargs.get("api_key") or config.get_llm_api_key("openai"),
            api_base=kwargs.get("api_base") or config.get_llm_api_base("openai"),
            model=kwargs.get("model"),
        )
    if key == "anthropic":
        return AnthropicProvider(
            api_key=kwargs.get("api_key") or config.get_llm_api_key("anthropic"),
            api_base=kwargs.get("api_base") or config.get_llm_api_base("anthropic"),
            model=kwargs.get("model"),
        )
    raise ValueError(
        f"unknown LLM provider {name!r}; expected ollama|openai|anthropic|auto"
    )
