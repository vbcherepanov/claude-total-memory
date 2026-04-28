"""v11 W1-A — Episode extractor.

Walks a session's facts in chronological order and segments them into
coherent episodes. A new episode boundary is opened whenever ANY of the
following holds between consecutive facts:

  1. ``ended.created_at`` and ``next.created_at`` differ by more than
     :data:`TIME_GAP_SECONDS` (default 1 hour).
  2. The cosine similarity between their summary embeddings drops by
     more than :data:`COSINE_DROP_THRESHOLD` (default 0.30) — i.e. the
     topic shifted.
  3. Their participant sets (extracted from each fact's ``tags``
     column) are disjoint AND non-empty on at least one side. A change
     in *who* is involved always opens a new episode.

The extractor is deterministic. When ``llm_summarizer`` is supplied it
is called once per segment with the joined fact contents; otherwise the
summary falls back to ``"<first-fact-title> (+N more)"`` using the head
of the first fact (truncated at 80 chars) and the segment's fact count.

Idempotency: the migration guarantees a UNIQUE index on
``(project, IFNULL(session_id,''), started_at)``, so a second run with
the same session is a no-op for already-stored episodes. The function
returns the records that were *newly* persisted; rows that were skipped
because they already existed are NOT returned, matching the contract
"insert-or-skip".
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

# memory_core/episodes/extractor.py → memory_core → src
_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from memory_core.episodes.schema import EpisodeFact, EpisodeRecord  # noqa: E402


# ─── tunables ───────────────────────────────────────────────────────────

TIME_GAP_SECONDS: float = 60 * 60        # 1 hour, per W1-A spec
COSINE_DROP_THRESHOLD: float = 0.30       # 30 percentage-point drop
SUMMARY_FALLBACK_HEAD: int = 80           # chars of first fact in fallback


# ─── public API ─────────────────────────────────────────────────────────


def extract_episodes_from_session(
    conn: sqlite3.Connection,
    project: str,
    session_id: str,
    *,
    llm_summarizer: Callable[[str], str] | None = None,
    embed_fn: Callable[[str], Sequence[float]] | None = None,
) -> list[EpisodeRecord]:
    """Segment a session's facts into episodes and persist them.

    Args:
        conn: open SQLite connection. Must already have migrations 001
            and 023 applied.
        project: project name to scope the fact pull.
        session_id: target session. Required — the W1-A layer always
            extracts within a single session.
        llm_summarizer: optional summary generator. If absent the
            deterministic fallback runs. Pure function: ``str -> str``.
        embed_fn: optional embedding callable. If absent the function
            uses :class:`memory_core.embeddings.EmbeddingProvider`.

    Returns:
        Newly created :class:`EpisodeRecord` instances, ``id`` populated.
        Already-existing episodes for the same anchor are skipped silently
        (idempotency contract).
    """
    if not project:
        raise ValueError("extract_episodes_from_session: project is required")
    if not session_id:
        raise ValueError("extract_episodes_from_session: session_id is required")

    facts = _load_session_facts(conn, project, session_id)
    if not facts:
        return []

    embed = embed_fn or _default_embed_fn()
    summarize = llm_summarizer or _fallback_summarizer

    # Pre-compute embeddings once — segmentation needs adjacent pairs and
    # summary embedding writes need the same vector for the segment head.
    fact_vectors: list[Sequence[float]] = [embed(f.content) for f in facts]

    segments = _segment_facts(facts, fact_vectors)
    if not segments:
        return []

    inserted: list[EpisodeRecord] = []
    # One transaction across all segments — partial extraction must never
    # leave dangling episode rows without their fact links.
    with _transaction(conn):
        for seg_idx_range in segments:
            lo, hi = seg_idx_range  # inclusive lo, exclusive hi
            seg_facts = facts[lo:hi]
            seg_vecs = fact_vectors[lo:hi]
            record = _build_record(
                project=project,
                session_id=session_id,
                seg_facts=seg_facts,
                seg_vecs=seg_vecs,
                summarize=summarize,
                embed=embed,
            )
            new_id = _insert_episode(conn, record)
            if new_id is None:
                # idempotent skip — anchor already present
                continue
            record.id = new_id
            _insert_episode_facts(conn, new_id, [f.knowledge_id for f in seg_facts])
            _insert_episode_fts(conn, new_id, record)
            inserted.append(record)

    return inserted


# ─── fact loading ───────────────────────────────────────────────────────


def _load_session_facts(
    conn: sqlite3.Connection,
    project: str,
    session_id: str,
) -> list[EpisodeFact]:
    """Pull active facts for the session ordered by created_at ascending.

    The flat `knowledge` table is the source. We only consider rows with
    `status='active'` to avoid pulling superseded duplicates into an
    episode.
    """
    cur = conn.execute(
        """
        SELECT id, content, tags, created_at
        FROM knowledge
        WHERE project = ?
          AND session_id = ?
          AND COALESCE(status, 'active') = 'active'
          AND COALESCE(content, '') <> ''
        ORDER BY datetime(created_at) ASC, id ASC
        """,
        (project, session_id),
    )
    out: list[EpisodeFact] = []
    for row in cur.fetchall():
        # Row may be sqlite3.Row or plain tuple depending on connection setup.
        kid = int(row[0])
        content = str(row[1] or "")
        tags_raw = row[2]
        created_at = str(row[3] or "")
        out.append(
            EpisodeFact(
                knowledge_id=kid,
                created_at=created_at,
                content=content,
                tags=_parse_tags(tags_raw),
            )
        )
    return out


def _parse_tags(raw: object) -> tuple[str, ...]:
    """Robust JSON-array tag parser — empty/missing/garbage all map to ()."""
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(t).strip().lower() for t in raw if str(t).strip())
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ()
        try:
            data = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            # Comma-separated fallback for legacy rows.
            parts = [p.strip().lower() for p in s.split(",") if p.strip()]
            return tuple(parts)
        if isinstance(data, list):
            return tuple(str(t).strip().lower() for t in data if str(t).strip())
        return ()
    return ()


# ─── segmentation ───────────────────────────────────────────────────────


def _segment_facts(
    facts: list[EpisodeFact],
    vectors: list[Sequence[float]],
) -> list[tuple[int, int]]:
    """Return half-open index ranges [(lo, hi), ...] one per episode."""
    if not facts:
        return []
    ranges: list[tuple[int, int]] = []
    seg_start = 0
    for i in range(1, len(facts)):
        if _is_boundary(facts[i - 1], facts[i], vectors[i - 1], vectors[i]):
            ranges.append((seg_start, i))
            seg_start = i
    ranges.append((seg_start, len(facts)))
    return ranges


def _is_boundary(
    prev: EpisodeFact,
    curr: EpisodeFact,
    prev_vec: Sequence[float],
    curr_vec: Sequence[float],
) -> bool:
    # 1. Time gap rule.
    gap = _time_gap_seconds(prev.created_at, curr.created_at)
    if gap is not None and gap > TIME_GAP_SECONDS:
        return True
    # 2. Topic-shift rule via embedding cosine drop.
    sim = _cosine(prev_vec, curr_vec)
    if sim is not None and (1.0 - sim) > COSINE_DROP_THRESHOLD:
        return True
    # 3. Participant change. Only fires when both sides have tags AND
    # the intersection is empty — otherwise tag-poor sessions would
    # explode into one-fact episodes.
    if prev.tags and curr.tags:
        if not (set(prev.tags) & set(curr.tags)):
            return True
    return False


def _time_gap_seconds(a_iso: str, b_iso: str) -> float | None:
    a = _parse_iso(a_iso)
    b = _parse_iso(b_iso)
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    s = value.strip()
    # SQLite often emits "...Z" suffix; Python <3.11 fromisoformat needs +00:00
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return None
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ─── record building ────────────────────────────────────────────────────


def _build_record(
    *,
    project: str,
    session_id: str,
    seg_facts: list[EpisodeFact],
    seg_vecs: list[Sequence[float]],
    summarize: Callable[[str], str],
    embed: Callable[[str], Sequence[float]],
) -> EpisodeRecord:
    started_at = seg_facts[0].created_at
    ended_at = seg_facts[-1].created_at

    # Participants = union of canonical tags across the segment, sorted
    # for determinism. We keep all tags rather than a hard-coded subset
    # because v11 has no fixed entity registry yet.
    participants_set: set[str] = set()
    for f in seg_facts:
        participants_set.update(f.tags)
    participants = tuple(sorted(participants_set))

    joined = "\n\n".join(f.content for f in seg_facts)
    summary_text = summarize(joined) or _fallback_summarizer(joined)
    summary_text = summary_text.strip() or _fallback_summarizer(joined)

    summary_vec = embed(summary_text)

    return EpisodeRecord(
        project=project,
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        summary=summary_text,
        participants=participants,
        location=None,
        outcome=None,
        fact_ids=tuple(f.knowledge_id for f in seg_facts),
        embedding=tuple(float(v) for v in summary_vec) if summary_vec else None,
    )


def _fallback_summarizer(text: str) -> str:
    """Deterministic summary used when no LLM is wired in."""
    if not text:
        return "(empty episode)"
    head = text.strip().splitlines()[0]
    head = head.strip()
    if len(head) > SUMMARY_FALLBACK_HEAD:
        head = head[: SUMMARY_FALLBACK_HEAD - 1].rstrip() + "…"
    extras = max(0, text.count("\n\n"))
    return head if extras == 0 else f"{head} (+{extras} more)"


# ─── persistence ────────────────────────────────────────────────────────


def _insert_episode(conn: sqlite3.Connection, record: EpisodeRecord) -> int | None:
    """INSERT OR IGNORE — returns the new id, or None if skipped."""
    blob = _vec_to_blob(record.embedding) if record.embedding else None
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO episodes_v11 (
            project, session_id, started_at, ended_at,
            participants, location, summary, outcome, embedding_blob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.project,
            record.session_id,
            record.started_at,
            record.ended_at,
            _json_tuple(record.participants),
            record.location,
            record.summary,
            record.outcome,
            blob,
        ),
    )
    if cur.rowcount == 0:
        return None
    return int(cur.lastrowid)


def _insert_episode_facts(
    conn: sqlite3.Connection,
    episode_id: int,
    knowledge_ids: Iterable[int],
) -> None:
    rows = [(episode_id, int(kid)) for kid in knowledge_ids]
    if not rows:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO episode_facts (episode_id, knowledge_id) VALUES (?, ?)",
        rows,
    )


def _insert_episode_fts(
    conn: sqlite3.Connection,
    episode_id: int,
    record: EpisodeRecord,
) -> None:
    if not _fts_available(conn):
        return
    conn.execute(
        """
        INSERT INTO episodes_v11_fts (rowid, summary, participants, outcome)
        VALUES (?, ?, ?, ?)
        """,
        (
            episode_id,
            record.summary,
            " ".join(record.participants),
            record.outcome or "",
        ),
    )


def _fts_available(conn: sqlite3.Connection) -> bool:
    """Detect whether episodes_v11_fts exists in this DB.

    Some test setups apply migrations selectively; we don't want a missing
    FTS table to break extraction. Episode rows are still queryable by the
    cosine path even without FTS, so failing soft here is correct.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = 'episodes_v11_fts'"
    )
    return cur.fetchone() is not None


# ─── helpers ────────────────────────────────────────────────────────────


def _vec_to_blob(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *(float(x) for x in vec))


def _json_tuple(values: tuple[str, ...]) -> str | None:
    if not values:
        return None
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


class _transaction:
    """Context manager that commits on clean exit, rolls back on error.

    Python's ``sqlite3`` module starts an implicit transaction at the
    first DML statement under the default isolation level — issuing an
    explicit ``BEGIN`` here would raise "cannot start a transaction
    within a transaction". We rely on that implicit BEGIN and only own
    the commit/rollback decision so partial extraction never leaves
    dangling rows without their fact links.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> "sqlite3.Connection":
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        return False


def _default_embed_fn() -> Callable[[str], Sequence[float]]:
    """Late-bound default that resolves the production embedder.

    The import is deferred so test environments without fastembed can
    inject ``embed_fn`` directly without paying the import cost.
    """
    from memory_core.embeddings import EmbeddingProvider

    provider = EmbeddingProvider()

    def _embed(text: str) -> Sequence[float]:
        if not text:
            return ()
        return tuple(provider.embed_query(text))

    return _embed


__all__ = [
    "extract_episodes_from_session",
    "TIME_GAP_SECONDS",
    "COSINE_DROP_THRESHOLD",
]
