"""v11.0 §J — Embedding-space resolver.

A single source of truth that maps a chunk's classifier output (its
content type and detected language) to one of the four supported
embedding spaces. Pure-Python, zero dependencies, never touches the
network — safe to call from any hot-path module.

Supported spaces:

    text     prose, markdown body, mixed/unknown
    code     program text in any language
    log      logs, stacktraces, OS error output
    config   sql / json / yaml / toml / ini / env / shell snippets

When a per-space embedding model is not configured (the env var is
empty), every space still records its own name on the vector row but the
caller is expected to fall back to the TEXT model for actual encoding.
This is the §J forward-compat contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Re-use config module (not a hot-path violation — config has no LLM).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as _cfg  # noqa: E402


SUPPORTED_SPACES: tuple[str, ...] = ("text", "code", "log", "config")
DEFAULT_SPACE: str = "text"


# ─── content_type → space ────────────────────────────────────────────


_TYPE_TO_SPACE: dict[str, str] = {
    "text": "text",
    "markdown": "text",
    "mixed": "text",
    "unknown": "text",
    "prose": "text",

    "code": "code",
    "source": "code",
    "patch": "code",
    "diff": "code",
    "git_patch": "code",

    "sql": "config",
    "json": "config",
    "yaml": "config",
    "toml": "config",
    "ini": "config",
    "env": "config",
    "shell": "config",
    "bash": "config",
    "config": "config",

    "log": "log",
    "logs": "log",
    "stacktrace": "log",
    "traceback": "log",
    "error": "log",
}


def resolve_space(content_type: str | None, language: str | None = None) -> str:
    """Pick the embedding space for a chunk.

    Resolution order:
      1. Explicit content_type lookup in `_TYPE_TO_SPACE`.
      2. If content_type is generic ("text"/"unknown") but `language` is
         set, treat it as code.
      3. Fall back to DEFAULT_SPACE.

    The function is total and never raises — it always returns one of
    SUPPORTED_SPACES so the caller can rely on a non-null column value.
    """
    ct = (content_type or "").strip().lower()
    space = _TYPE_TO_SPACE.get(ct)
    if space is not None:
        return space
    # Heuristic: a non-empty `language` ≈ code chunk even when classifier
    # could not pin the content_type tighter.
    if language and language.strip().lower() not in ("", "none", "text", "markdown"):
        return "code"
    return DEFAULT_SPACE


# ─── space → model selection ─────────────────────────────────────────


def model_for_space(space: str) -> str:
    """Return the embedding model configured for `space`.

    Empty per-space env vars mean "fall back to the TEXT model" — see the
    §J contract. We deliberately return the TEXT model in that case so
    every call-site gets something usable; the row metadata still records
    the requested space, so a future model swap is a one-flag change.
    """
    space = (space or DEFAULT_SPACE).strip().lower()
    text_model = _cfg.get_text_embed_model()
    if space == "text":
        return text_model
    if space == "code":
        return _cfg.get_code_embed_model() or text_model
    if space == "log":
        return _cfg.get_log_embed_model() or text_model
    if space == "config":
        return _cfg.get_config_embed_model() or text_model
    return text_model


def is_space_supported(space: str | None) -> bool:
    return (space or "").strip().lower() in SUPPORTED_SPACES


__all__ = [
    "SUPPORTED_SPACES",
    "DEFAULT_SPACE",
    "resolve_space",
    "model_for_space",
    "is_space_supported",
]
