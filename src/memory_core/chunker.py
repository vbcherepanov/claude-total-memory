"""v11.0 Phase 3 — Deterministic chunker.

Single entry point :func:`chunk` that dispatches on content_type:

* markdown — split by headings, each fenced block becomes its own
  code-typed sub-chunk so the embedding-space resolver can route it.
* text     — paragraph split with sentence-overlap (delegates to the
  existing `ingestion.chunker.SemanticChunker`).
* code     — prefer `ast_ingest` if importable; regex fallback by
  function/class boundaries; final fallback = char split.
* log / stacktrace — split per error block / frame group.
* default  — char-based split with paragraph-aware boundaries.

No LLM. No network. The Chunk dataclass carries everything downstream
needs (content hash for dedup, char counts for budgeting, etc).
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ─── data class ──────────────────────────────────────────────────────


@dataclass
class Chunk:
    text: str
    content_type: str
    position: int
    char_count: int
    token_count: int
    content_hash: str
    language: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _approx_tokens(text: str) -> int:
    # ~4 chars/token approximation, matches the legacy chunker.
    return max(1, len(text) // 4) if text else 0


def _make_chunk(
    text: str,
    *,
    content_type: str,
    language: Optional[str],
    position: int,
    metadata: Optional[dict] = None,
    parent_id: Optional[str] = None,
) -> Chunk:
    text = text.strip("\n")
    return Chunk(
        text=text,
        content_type=content_type,
        language=language,
        position=position,
        parent_id=parent_id,
        char_count=len(text),
        token_count=_approx_tokens(text),
        content_hash=_hash(text),
        metadata=dict(metadata or {}),
    )


# ─── markdown ────────────────────────────────────────────────────────


_RE_FENCE_BLOCK = re.compile(
    r"^```(?P<lang>[\w+\-./]*)\s*\n(?P<body>.*?)\n```\s*$",
    re.MULTILINE | re.DOTALL,
)
_RE_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


def _chunk_markdown(content: str, *, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    pos = 0

    # 1. Pull fenced code blocks first; replace with placeholders so the
    #    surrounding prose is split independently.
    code_blocks: list[tuple[str, Optional[str]]] = []
    placeholder_template = "<<FENCE_{}>>"

    def _stash(match: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append((match.group("body"), (match.group("lang") or None) or None))
        return placeholder_template.format(idx)

    skeleton = _RE_FENCE_BLOCK.sub(_stash, content)

    # 2. Split skeleton by headings.
    sections: list[str] = []
    last_end = 0
    for m in _RE_HEADING.finditer(skeleton):
        if m.start() > last_end:
            chunk_text = skeleton[last_end : m.start()].strip()
            if chunk_text:
                sections.append(chunk_text)
        last_end = m.start()
    tail = skeleton[last_end:].strip()
    if tail:
        sections.append(tail)
    if not sections:
        sections = [skeleton]

    # 3. Emit one chunk per section, splitting oversized sections by
    #    paragraph; restore code blocks in-place by emitting them separately.
    for section in sections:
        # Find placeholders and emit code blocks in their natural order.
        parts = re.split(r"(<<FENCE_\d+>>)", section)
        for part in parts:
            if not part.strip():
                continue
            ph = re.fullmatch(r"<<FENCE_(\d+)>>", part.strip())
            if ph:
                body, lang = code_blocks[int(ph.group(1))]
                chunks.append(
                    _make_chunk(
                        body,
                        content_type="code",
                        language=lang,
                        position=pos,
                        metadata={"from_markdown_fence": True},
                    )
                )
                pos += 1
                continue
            # Prose chunk; split if oversized.
            for sub in _split_by_paragraph(part, max_chars=max_chars):
                chunks.append(
                    _make_chunk(
                        sub,
                        content_type="markdown",
                        language=None,
                        position=pos,
                    )
                )
                pos += 1
    return chunks


def _split_by_paragraph(content: str, *, max_chars: int, overlap: int = 1) -> list[str]:
    """Split prose by paragraphs, with N-paragraph overlap when oversized."""
    if not content.strip():
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    if not paras:
        return [content.strip()]
    out: list[str] = []
    current: list[str] = []
    for p in paras:
        joined = "\n\n".join(current + [p])
        if len(joined) <= max_chars or not current:
            current.append(p)
            continue
        out.append("\n\n".join(current))
        # overlap the trailing N paragraphs of the just-emitted chunk
        if overlap > 0 and len(current) > overlap:
            current = current[-overlap:] + [p]
        else:
            current = [p]
    if current:
        out.append("\n\n".join(current))
    return out


# ─── code ────────────────────────────────────────────────────────────


_RE_CODE_BOUNDARY = re.compile(
    r"(?m)^(?:async\s+)?(?:def|class|func|fn|function|public|private|protected|static)\b"
)


def _chunk_code(
    content: str,
    *,
    language: Optional[str],
    max_chars: int,
) -> list[Chunk]:
    # Prefer ast_ingest when available — it produces semantically clean
    # function-level chunks. We import opportunistically; failures fall
    # through to the regex split so this stays hot-path safe.
    try:
        from ast_ingest.ingester import chunk_source as _ast_chunk_source  # type: ignore
    except Exception:  # noqa: BLE001
        _ast_chunk_source = None  # type: ignore[assignment]

    if _ast_chunk_source is not None and language:
        try:
            ast_chunks = _ast_chunk_source(content, language=language)
        except Exception:  # noqa: BLE001
            ast_chunks = None
        if ast_chunks:
            return [
                _make_chunk(
                    c.get("text") or c.get("content") or "",
                    content_type="code",
                    language=language,
                    position=i,
                    metadata={
                        k: v
                        for k, v in c.items()
                        if k in ("name", "kind", "start_line", "end_line", "signature")
                        and v is not None
                    },
                )
                for i, c in enumerate(ast_chunks)
                if (c.get("text") or c.get("content"))
            ]

    # Regex split by function/class boundaries.
    boundaries: list[int] = [0]
    for m in _RE_CODE_BOUNDARY.finditer(content):
        if m.start() not in boundaries:
            boundaries.append(m.start())
    boundaries.append(len(content))
    boundaries = sorted(set(boundaries))

    chunks: list[Chunk] = []
    pos = 0
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        block = content[start:end].strip("\n")
        if not block.strip():
            continue
        if len(block) <= max_chars:
            chunks.append(
                _make_chunk(
                    block,
                    content_type="code",
                    language=language,
                    position=pos,
                )
            )
            pos += 1
            continue
        # Oversized — char split, line-aligned.
        for piece in _split_by_chars_lines(block, max_chars=max_chars):
            chunks.append(
                _make_chunk(
                    piece,
                    content_type="code",
                    language=language,
                    position=pos,
                )
            )
            pos += 1

    if not chunks:
        # Single chunk fallback for very short/no-boundary inputs.
        for piece in _split_by_chars_lines(content, max_chars=max_chars):
            chunks.append(
                _make_chunk(
                    piece,
                    content_type="code",
                    language=language,
                    position=pos,
                )
            )
            pos += 1
    return chunks


def _split_by_chars_lines(content: str, *, max_chars: int) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    cur = 0
    for line in content.splitlines(keepends=True):
        if cur + len(line) > max_chars and buf:
            out.append("".join(buf).rstrip("\n"))
            buf = [line]
            cur = len(line)
        else:
            buf.append(line)
            cur += len(line)
    if buf:
        out.append("".join(buf).rstrip("\n"))
    return [o for o in out if o.strip()]


# ─── log / stacktrace ────────────────────────────────────────────────


_RE_TRACEBACK_HEADER = re.compile(
    r"(Traceback \(most recent call last\):|^goroutine \d+ \[|Exception in thread \"[^\"]+\")",
    re.MULTILINE,
)
_RE_LOG_LEVEL_LINE = re.compile(
    r"(?m)^.{0,40}\b(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|TRACE)\b"
)


def _chunk_log(content: str, *, max_chars: int) -> list[Chunk]:
    if not content.strip():
        return []

    # Split on traceback headers if any are present.
    if _RE_TRACEBACK_HEADER.search(content):
        positions = [m.start() for m in _RE_TRACEBACK_HEADER.finditer(content)]
        positions = [0] + positions + [len(content)]
        positions = sorted(set(positions))
        blocks = [
            content[positions[i] : positions[i + 1]].strip("\n")
            for i in range(len(positions) - 1)
        ]
    else:
        # Split by leveled groups: every level line starts a new block.
        blocks = []
        cur: list[str] = []
        for line in content.splitlines():
            if cur and _RE_LOG_LEVEL_LINE.match(line):
                blocks.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            blocks.append("\n".join(cur))

    chunks: list[Chunk] = []
    pos = 0
    for block in blocks:
        block = block.strip("\n")
        if not block.strip():
            continue
        if len(block) <= max_chars:
            chunks.append(
                _make_chunk(
                    block,
                    content_type="log",
                    language=None,
                    position=pos,
                )
            )
            pos += 1
            continue
        for piece in _split_by_chars_lines(block, max_chars=max_chars):
            chunks.append(
                _make_chunk(
                    piece,
                    content_type="log",
                    language=None,
                    position=pos,
                )
            )
            pos += 1
    return chunks


# ─── text fallback (paragraph + sentence overlap via legacy) ─────────


def _chunk_text(content: str, *, max_chars: int) -> list[Chunk]:
    # Reuse the existing semantic chunker for prose — it has battle-tested
    # paragraph merging and sentence overlap. Fallback to char split if it
    # fails to import for any reason.
    try:
        from ingestion.chunker import SemanticChunker  # noqa: WPS433

        sc = SemanticChunker()
        # The legacy class budgets in tokens; convert max_chars roughly.
        sc.MAX_CHUNK_TOKENS = max(64, max_chars // 4)
        raw = sc.chunk(content, content_type="text")
        if raw:
            return [
                _make_chunk(
                    item["content"],
                    content_type="text",
                    language=None,
                    position=i,
                )
                for i, item in enumerate(raw)
                if item.get("content")
            ]
    except Exception:  # noqa: BLE001
        pass

    pieces = _split_by_paragraph(content, max_chars=max_chars, overlap=1)
    return [
        _make_chunk(p, content_type="text", language=None, position=i)
        for i, p in enumerate(pieces)
    ]


# ─── public API ──────────────────────────────────────────────────────


_TYPE_TO_HANDLER: dict[str, str] = {
    "markdown": "markdown",
    "text": "text",
    "mixed": "text",
    "unknown": "text",
    "code": "code",
    "source": "code",
    "patch": "code",
    "diff": "code",
    "log": "log",
    "logs": "log",
    "stacktrace": "log",
    "traceback": "log",
}


def chunk(
    content: str,
    *,
    content_type: str,
    language: Optional[str] = None,
    max_chars: int = 1500,
) -> list[Chunk]:
    if not content or not content.strip():
        return []
    handler = _TYPE_TO_HANDLER.get((content_type or "").strip().lower(), "text")
    if handler == "markdown":
        return _chunk_markdown(content, max_chars=max_chars)
    if handler == "code":
        return _chunk_code(content, language=language, max_chars=max_chars)
    if handler == "log":
        return _chunk_log(content, max_chars=max_chars)
    return _chunk_text(content, max_chars=max_chars)


__all__ = ["Chunk", "chunk"]
