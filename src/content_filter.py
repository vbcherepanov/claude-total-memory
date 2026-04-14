"""Content filter — declarative token-saving pipeline (rtk-style, safety-first).

Python port of rtk-ai/rtk's 8-stage TOML filter DSL, with a hard-wired
whitelist (URLs / absolute paths / inline `code` / `~/paths`) that survives
even the most aggressive filter rules when `safety="strict"` (default).

Stages (applied in order):
    1. strip_ansi     — remove \\x1b[...m color codes
    2. replace        — regex substitutions
    3. match_output   — early-exit if pattern found (useful for auto-detect)
    4. strip_lines    — drop lines matching any regex
    5. keep_lines     — keep only lines matching any regex
    6. truncate       — truncate each line to max_chars
    7. head / tail    — first / last N lines
    8. max_lines      — overall cap
    9. on_empty       — placeholder when everything filtered

Safety modes:
    - strict    (default) — whitelist always restored after filtering
    - semantic  — treat whitelist as preserved, allow paraphrase (caller guards)
    - off       — whitelist disabled (caller has accepted losses)
"""

from __future__ import annotations

import re
import sys
import tomllib  # Python 3.11+
from pathlib import Path
from typing import Any

LOG = lambda msg: sys.stderr.write(f"[content-filter] {msg}\n")


# ──────────────────────────────────────────────
# Regex toolbox
# ──────────────────────────────────────────────


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Whitelist patterns — these substrings MUST survive in strict mode.
_URL_RE = re.compile(r"https?://[^\s<>`'\"()]+")
_ABS_PATH_RE = re.compile(r"(?:^|\s)(/[A-Za-z0-9_.\-][A-Za-z0-9_./\-]*)")
_TILDE_PATH_RE = re.compile(r"(?:^|\s)(~/[A-Za-z0-9_./\-]+)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_CODE_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


# ──────────────────────────────────────────────
# Individual stages (pure, testable)
# ──────────────────────────────────────────────


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def apply_replace(text: str, rules: list[dict[str, str]]) -> str:
    for rule in rules or []:
        pattern = rule.get("pattern")
        replacement = rule.get("replacement", "")
        if not pattern:
            continue
        try:
            text = re.sub(pattern, replacement, text)
        except re.error as e:
            LOG(f"bad replace pattern {pattern!r}: {e}")
    return text


def keep_lines(text: str, patterns: list[str]) -> str:
    if not patterns:
        return text
    compiled = _compile_patterns(patterns)
    kept = [
        line for line in (text or "").splitlines()
        if any(p.search(line) for p in compiled)
    ]
    return "\n".join(kept)


def strip_lines(text: str, patterns: list[str]) -> str:
    if not patterns:
        return text
    compiled = _compile_patterns(patterns)
    kept = [
        line for line in (text or "").splitlines()
        if not any(p.search(line) for p in compiled)
    ]
    return "\n".join(kept)


def truncate_lines(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    out = []
    for line in (text or "").splitlines():
        if len(line) > max_chars:
            out.append(line[: max_chars - 3] + "...")
        else:
            out.append(line)
    return "\n".join(out)


def head_lines(text: str, n: int) -> str:
    if n <= 0:
        return ""
    return "\n".join((text or "").splitlines()[:n])


def tail_lines(text: str, n: int) -> str:
    if n <= 0:
        return ""
    return "\n".join((text or "").splitlines()[-n:])


# ──────────────────────────────────────────────
# Whitelist stashing (strict safety)
# ──────────────────────────────────────────────


def _extract_whitelist(text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Extract (urls, paths, tilde_paths, inline_codes) from text."""
    urls = list({m.group(0) for m in _URL_RE.finditer(text)})
    # Trim trailing punctuation from URLs
    urls = [u.rstrip(".,;:!?") for u in urls]
    paths = list({m.group(1) for m in _ABS_PATH_RE.finditer(text)})
    tildes = list({m.group(1) for m in _TILDE_PATH_RE.finditer(text)})
    inlines = list({m.group(0) for m in _INLINE_CODE_RE.finditer(text)})
    return urls, paths, tildes, inlines


def _append_preserved(filtered: str, preserved: list[str]) -> str:
    """Append whitelist items missing from `filtered` with a tag."""
    missing = [p for p in preserved if p and p not in filtered]
    if not missing:
        return filtered
    tag = " | preserved: " + ", ".join(missing)
    if filtered:
        return filtered + "\n" + tag.lstrip(" |")
    return tag.lstrip(" |")


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────


def run_pipeline(
    text: str,
    config: dict[str, Any],
    safety: str = "strict",
) -> str:
    """Apply stages from `config` in order, honoring `safety` mode."""
    if text is None:
        return ""

    # In strict mode, stash whitelist BEFORE filtering
    preserved: list[str] = []
    if safety == "strict":
        urls, paths, tildes, inlines = _extract_whitelist(text)
        preserved = urls + paths + tildes + inlines

    out = text

    if config.get("strip_ansi"):
        out = strip_ansi(out)
    if config.get("replace"):
        out = apply_replace(out, config["replace"])
    if config.get("strip_lines"):
        out = strip_lines(out, config["strip_lines"])
    if config.get("keep_lines"):
        out = keep_lines(out, config["keep_lines"])
    if config.get("truncate_chars"):
        out = truncate_lines(out, int(config["truncate_chars"]))
    if config.get("head") is not None:
        out = head_lines(out, int(config["head"]))
    if config.get("tail") is not None:
        out = tail_lines(out, int(config["tail"]))
    if config.get("max_lines") is not None:
        n = int(config["max_lines"])
        if n >= 0:
            out = "\n".join(out.splitlines()[:n])

    # On-empty fallback
    if not out.strip() and config.get("on_empty"):
        out = str(config["on_empty"])

    # Strict: guarantee whitelist survives
    if safety == "strict" and preserved:
        out = _append_preserved(out, preserved)

    return out


def filter_with_stats(
    text: str,
    config: dict[str, Any],
    safety: str = "strict",
) -> tuple[str, dict[str, Any]]:
    """Run the pipeline and return (filtered_text, stats)."""
    input_chars = len(text or "")
    input_lines = len((text or "").splitlines())

    out = run_pipeline(text, config, safety)

    output_chars = len(out)
    output_lines = len(out.splitlines())
    reduction_pct = 0.0
    if input_chars > 0:
        reduction_pct = max(0.0, (1 - output_chars / input_chars) * 100)

    return out, {
        "input_chars": input_chars,
        "input_lines": input_lines,
        "output_chars": output_chars,
        "output_lines": output_lines,
        "reduction_pct": round(reduction_pct, 1),
        "safety": safety,
    }


# ──────────────────────────────────────────────
# TOML config loading
# ──────────────────────────────────────────────


def load_filter_config(path: Path | str) -> dict[str, Any]:
    """Load a TOML filter config. Structure:

        name = "..."
        safety = "strict" | "semantic" | "off"

        [stages]
        strip_ansi = true
        replace = [{pattern = "...", replacement = "..."}, ...]
        strip_lines = ["pattern", ...]
        keep_lines = ["pattern", ...]
        truncate_chars = 200
        head = 100
        tail = 100
        max_lines = 500
        on_empty = "(empty)"
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    # Normalise: prefer nested [stages] block but accept flat
    if "stages" not in data:
        data["stages"] = {
            k: v for k, v in data.items()
            if k not in {"name", "description", "safety"}
        }
    return data


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    out = []
    for p in patterns:
        try:
            out.append(re.compile(p))
        except re.error as e:
            LOG(f"bad pattern {p!r}: {e}")
    return out
