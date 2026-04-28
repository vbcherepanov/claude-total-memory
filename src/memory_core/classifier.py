"""v11.0 Phase 3 — Deterministic content classifier.

Pure-Python regex/heuristic detector. Outputs the (content_type, language)
tuple that `embedding_spaces.resolve_space` expects.

Detection ladder:

1. File extension (when `file_path` is given) — strongest signal.
2. Markdown fences / heading density.
3. Stack-trace / log signatures.
4. Structured-config signatures (json / yaml / toml / sql / shell).
5. Generic code heuristics (function/class/import keywords across langs).
6. Fallback to "text".

NEVER calls an LLM and never reaches the network. The classifier is
deliberately tiny — it only has to be good enough to bucket a chunk into
one of the four embedding spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── result type ─────────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    type: str
    language: Optional[str] = None
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)


# ─── extension table ─────────────────────────────────────────────────


_EXT_TO_TYPE: dict[str, tuple[str, Optional[str]]] = {
    # Markdown / text
    ".md": ("markdown", None),
    ".markdown": ("markdown", None),
    ".rst": ("markdown", None),
    ".txt": ("text", None),
    # Code
    ".py": ("code", "python"),
    ".pyi": ("code", "python"),
    ".go": ("code", "go"),
    ".rs": ("code", "rust"),
    ".js": ("code", "javascript"),
    ".mjs": ("code", "javascript"),
    ".cjs": ("code", "javascript"),
    ".ts": ("code", "typescript"),
    ".tsx": ("code", "tsx"),
    ".jsx": ("code", "javascript"),
    ".vue": ("code", "vue"),
    ".php": ("code", "php"),
    ".rb": ("code", "ruby"),
    ".java": ("code", "java"),
    ".kt": ("code", "kotlin"),
    ".swift": ("code", "swift"),
    ".cs": ("code", "csharp"),
    ".cpp": ("code", "cpp"),
    ".cc": ("code", "cpp"),
    ".cxx": ("code", "cpp"),
    ".hpp": ("code", "cpp"),
    ".h": ("code", "c"),
    ".c": ("code", "c"),
    # Config / structured
    ".sql": ("sql", "sql"),
    ".json": ("json", None),
    ".jsonl": ("json", None),
    ".yaml": ("yaml", None),
    ".yml": ("yaml", None),
    ".toml": ("toml", None),
    ".ini": ("ini", None),
    ".cfg": ("ini", None),
    ".env": ("env", None),
    ".sh": ("shell", "bash"),
    ".bash": ("shell", "bash"),
    ".zsh": ("shell", "bash"),
    ".dockerfile": ("config", None),
    # Logs
    ".log": ("log", None),
}


# ─── regex signatures ────────────────────────────────────────────────


_RE_FENCE = re.compile(r"```")
_RE_MD_HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
_RE_MD_BULLET = re.compile(r"(?m)^[-*]\s+\S")

_RE_STACKTRACE_PY = re.compile(r"Traceback \(most recent call last\):")
_RE_STACKTRACE_JAVA = re.compile(r"^\s*at\s+[\w$.]+\([\w./:]+\)$", re.MULTILINE)
_RE_STACKTRACE_NODE = re.compile(r"^\s*at\s+\S+\s+\(.*:\d+:\d+\)$", re.MULTILINE)
_RE_STACKTRACE_GO = re.compile(r"^goroutine \d+ \[", re.MULTILINE)

_RE_LOG_LEVEL = re.compile(
    r"\b(DEBUG|INFO|WARN(ING)?|ERROR|FATAL|TRACE)\b\s*[:|\]]",
    re.IGNORECASE,
)
_RE_LOG_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
    re.MULTILINE,
)

_RE_JSON_OBJ = re.compile(r"^\s*[{\[]")
_RE_YAML_DOC = re.compile(r"^---\s*$", re.MULTILINE)
_RE_YAML_KV = re.compile(r"(?m)^[A-Za-z_][\w-]*:\s")
_RE_TOML_TABLE = re.compile(r"(?m)^\[\w[\w.-]*\]\s*$")
_RE_INI_SECTION = re.compile(r"(?m)^\[[^\]]+\]\s*$")
_RE_ENV_LINE = re.compile(r"(?m)^[A-Z_][A-Z0-9_]*\s*=")

_RE_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE\s+(TABLE|INDEX|VIEW)|DROP\s+(TABLE|INDEX)|ALTER\s+TABLE|JOIN|WHERE|GROUP\s+BY|ORDER\s+BY)\b",
    re.IGNORECASE,
)

_RE_SHELL_HEADER = re.compile(r"^#!.*?(bash|sh|zsh)\b")
_RE_SHELL_CMDS = re.compile(
    r"(?m)^\s*(if\s|for\s|while\s|case\s|export\s|function\s|\$\(|`|\.\/|sudo\s)"
)

_RE_CODE_HINTS = (
    re.compile(r"^\s*(import|from)\s+[\w.]+", re.MULTILINE),  # python/js
    re.compile(r"^\s*func\s+\w+", re.MULTILINE),  # go
    re.compile(r"^\s*fn\s+\w+", re.MULTILINE),  # rust
    re.compile(r"^\s*(public|private|protected)\s+\w+\s+\w+\s*\(", re.MULTILINE),
    re.compile(r"^\s*(class|interface|trait)\s+\w+", re.MULTILINE),
    re.compile(r"^\s*(def|async\s+def)\s+\w+\s*\(", re.MULTILINE),
    re.compile(r"^\s*(const|let|var)\s+\w+\s*=", re.MULTILINE),
    re.compile(r"^\s*<\?php\b", re.MULTILINE),
)

# Strong single-match signatures — even one of these is enough to call
# the chunk code (vs the multi-hint heuristic above which needs 2 hits).
_RE_CODE_STRONG = (
    # `def name(...) -> ...:` or `def name(...):`  (python, multi-line)
    (re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*(?:->\s*[\w\[\], .]+)?\s*:", re.MULTILINE), "python"),
    # `func name(...) ... {` (go) or `func (recv) name(...) ... {`
    (re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?\w+\s*\([^)]*\)[^{\n]*\{", re.MULTILINE), "go"),
    # `fn name(...) -> T {` (rust)
    (re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+\w+\s*[<(]", re.MULTILINE), "rust"),
    # `function name(...) {` or `name = (...) =>` (js/ts)
    (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w+\s*\(", re.MULTILINE), "javascript"),
    (re.compile(r"\b(?:const|let)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"), "javascript"),
    # `class Foo:` / `class Foo {` / `class Foo extends Bar {`
    (re.compile(r"^\s*class\s+\w+\s*[:({]", re.MULTILINE), None),
    # PHP open tag
    (re.compile(r"<\?php\b"), "php"),
    # C/C++/Java method signature with body opening brace on same line
    (re.compile(r"^\s*(?:public|private|protected|static)\s+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{", re.MULTILINE), None),
)


# ─── helpers ─────────────────────────────────────────────────────────


def _check_extension(file_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not file_path:
        return None, None
    suffix = Path(file_path).suffix.lower()
    if not suffix:
        # Dockerfile / Makefile and friends — name-based.
        name = Path(file_path).name.lower()
        if name in ("dockerfile", "containerfile"):
            return "config", None
        if name in ("makefile",):
            return "config", "make"
        if name == ".env" or name.startswith(".env."):
            return "env", None
        return None, None
    if suffix in _EXT_TO_TYPE:
        ct, lang = _EXT_TO_TYPE[suffix]
        return ct, lang
    return None, None


def _looks_like_json(content: str) -> bool:
    if not _RE_JSON_OBJ.match(content):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if stripped[0] not in "[{":
        return False
    # Cheap structural check — let json.loads do the heavy lifting only if
    # the document is small (we don't want to pay for megabyte logs).
    if len(stripped) <= 200_000:
        import json

        try:
            json.loads(stripped)
            return True
        except (ValueError, TypeError):
            return False
    # Big input — trust the prefix.
    return True


def _looks_like_yaml(content: str) -> bool:
    if _RE_YAML_DOC.search(content):
        return True
    kv = _RE_YAML_KV.findall(content)
    return len(kv) >= 3 and ":" in content and not _looks_like_json(content)


def _looks_like_toml(content: str) -> bool:
    return bool(_RE_TOML_TABLE.search(content))


def _looks_like_ini(content: str) -> bool:
    return (
        bool(_RE_INI_SECTION.search(content))
        and "=" in content
        and not _looks_like_toml(content)
    )


def _looks_like_env(content: str) -> bool:
    lines = [ln for ln in content.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        return False
    matches = sum(1 for ln in lines if _RE_ENV_LINE.match(ln))
    return matches >= max(2, len(lines) // 2)


def _looks_like_sql(content: str) -> bool:
    hits = len(_RE_SQL_KEYWORDS.findall(content))
    return hits >= 2


def _looks_like_shell(content: str) -> bool:
    if _RE_SHELL_HEADER.match(content):
        return True
    return len(_RE_SHELL_CMDS.findall(content)) >= 2


def _looks_like_log(content: str) -> bool:
    if _RE_LOG_TIMESTAMP.search(content) and _RE_LOG_LEVEL.search(content):
        return True
    # 3+ leveled lines without a timestamp also count (journalctl style).
    return len(_RE_LOG_LEVEL.findall(content)) >= 3


def _looks_like_stacktrace(content: str) -> bool:
    return any(
        rx.search(content)
        for rx in (
            _RE_STACKTRACE_PY,
            _RE_STACKTRACE_JAVA,
            _RE_STACKTRACE_NODE,
            _RE_STACKTRACE_GO,
        )
    )


def _looks_like_markdown(content: str) -> bool:
    if len(_RE_FENCE.findall(content)) >= 2:
        return True
    headings = len(_RE_MD_HEADING.findall(content))
    bullets = len(_RE_MD_BULLET.findall(content))
    return headings >= 2 or (headings >= 1 and bullets >= 2)


def _looks_like_code(content: str) -> tuple[bool, Optional[str]]:
    """Return (is_code, best-guess language).

    A single STRONG signature (full def/func/fn/function/class line with
    parentheses + body marker) is enough — short snippets like
    `def add(a, b): return a + b` should not be classified as text.
    Otherwise fall back to the multi-hint heuristic (needs ≥2 hits).
    """
    # Strong single-match — short code snippets land here.
    for rx, lang in _RE_CODE_STRONG:
        if rx.search(content):
            return True, lang
    # Multi-hint heuristic — covers prose-mixed code where no single
    # signature is unambiguous on its own.
    score = 0
    lang: Optional[str] = None
    for rx in _RE_CODE_HINTS:
        m = rx.search(content)
        if not m:
            continue
        score += 1
        token = m.group(0).strip().split()[0].lower()
        if token in {"import", "from"} and "::" not in content:
            lang = lang or "python"
        elif token == "func":
            lang = "go"
        elif token == "fn":
            lang = "rust"
        elif token in {"def", "async"}:
            lang = lang or "python"
        elif token in {"const", "let", "var"}:
            lang = lang or "javascript"
        elif token.startswith("<?php"):
            lang = "php"
    return score >= 2, lang


# ─── public API ──────────────────────────────────────────────────────


def classify(
    content: str,
    *,
    file_path: str | None = None,
) -> ClassificationResult:
    """Classify `content` into a content_type + optional language."""
    reasons: list[str] = []

    # 1. Extension wins (highest signal).
    ext_type, ext_lang = _check_extension(file_path)
    if ext_type is not None:
        reasons.append(f"extension match: {file_path}")
        return ClassificationResult(
            type=ext_type, language=ext_lang, confidence=0.95, reasons=reasons
        )

    if not content or not content.strip():
        return ClassificationResult(
            type="text", language=None, confidence=0.1, reasons=["empty input"]
        )

    sample = content[:8192]  # cap regex work for huge inputs

    # 2. Stacktrace before log — stacktrace is a stronger signal.
    if _looks_like_stacktrace(sample):
        reasons.append("stacktrace signature")
        return ClassificationResult(
            type="stacktrace", language=None, confidence=0.9, reasons=reasons
        )

    if _looks_like_log(sample):
        reasons.append("log level + timestamp")
        return ClassificationResult(
            type="log", language=None, confidence=0.85, reasons=reasons
        )

    # 3. Structured config family.
    if _looks_like_json(sample):
        reasons.append("parses as json")
        return ClassificationResult(
            type="json", language=None, confidence=0.9, reasons=reasons
        )
    if _looks_like_toml(sample):
        reasons.append("toml [section] header")
        return ClassificationResult(
            type="toml", language=None, confidence=0.85, reasons=reasons
        )
    if _looks_like_yaml(sample):
        reasons.append("yaml document / kv markers")
        return ClassificationResult(
            type="yaml", language=None, confidence=0.8, reasons=reasons
        )
    if _looks_like_ini(sample):
        reasons.append("ini section markers")
        return ClassificationResult(
            type="ini", language=None, confidence=0.75, reasons=reasons
        )
    if _looks_like_env(sample):
        reasons.append("dotenv KEY=VALUE pattern")
        return ClassificationResult(
            type="env", language=None, confidence=0.75, reasons=reasons
        )
    if _looks_like_sql(sample):
        reasons.append("sql keywords")
        return ClassificationResult(
            type="sql", language="sql", confidence=0.85, reasons=reasons
        )
    if _looks_like_shell(sample):
        reasons.append("shell shebang / control flow")
        return ClassificationResult(
            type="shell", language="bash", confidence=0.8, reasons=reasons
        )

    # 4. Code heuristics before markdown — fenced markdown is handled below.
    is_code, lang = _looks_like_code(sample)
    if is_code:
        reasons.append("code keywords across languages")
        return ClassificationResult(
            type="code", language=lang, confidence=0.8, reasons=reasons
        )

    # 5. Markdown.
    if _looks_like_markdown(sample):
        reasons.append("markdown headings / fences")
        return ClassificationResult(
            type="markdown", language=None, confidence=0.7, reasons=reasons
        )

    # 6. Fallback.
    reasons.append("no signal — fallback to text")
    return ClassificationResult(
        type="text", language=None, confidence=0.4, reasons=reasons
    )


__all__ = ["ClassificationResult", "classify"]
