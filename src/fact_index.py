"""Structured fact lookup — (entity, attribute) → [values + knowledge_ids].

LoCoMo gain lane L2. Complements the semantic/FTS pipeline with a
high-precision short-circuit for questions of the form
"what/where/when did <entity> <action>?".

Data sources (all already populated by the existing v8 stack):

  * `graph_nodes`   — entities/concepts extracted from knowledge records.
  * `graph_edges`   — (subject_id, relation_type, target_id, weight).
  * `knowledge_nodes` — link from knowledge row → graph node (role = subject /
    object / mention).
  * `knowledge`     — raw text + metadata. Optionally, `type='synthesized_fact'`
    rows give already-distilled sentences that we parse opportunistically.

Lookup contract (sync, SQLite-backed):

    FactIndex(conn).lookup(entity="Alice", attribute="traveled", limit=10)
    → [{"value": "Berlin", "knowledge_id": 124, "weight": 1.8,
         "relation": "traveled_to", "evidence": "..."} , ...]

Query parsing is intentionally boring: lowercased substring match against
known entity names + a tiny stoplist. It runs in microseconds and feeds the
real retrieval path — it is NOT a replacement for semantic search, just a
pre-filter that lifts single-hop recall where the retriever currently
returns the raw turn instead of the fact.

Not wired into server.py yet (A2 lane owns server.py edits in Phase 1).
Consumed by ``benchmarks/locomo_bench_llm.py`` behind ``--fact-index`` and
exported for future server integration.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Iterable


# ──────────────────────────────────────────────
# Query parsing
# ──────────────────────────────────────────────
#
# We keep this deliberately small — no LLM, no spacy. The goal is ~μs
# overhead so the index can run on every recall call without regression.

_STOPWORDS = frozenset(
    # fmt: off
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by",
        "can", "did", "do", "does", "for", "from", "had", "has", "have", "he",
        "her", "him", "his", "how", "i", "if", "in", "is", "it", "its", "me",
        "my", "no", "not", "of", "on", "or", "our", "she", "so", "some", "than",
        "that", "the", "their", "them", "there", "these", "they", "this",
        "those", "to", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "whose", "why", "will", "with", "would", "you",
        "your", "about", "after", "again", "all", "also", "because", "before",
        "between", "could", "during", "each", "just", "like", "may", "more",
        "most", "much", "now", "only", "other", "over", "same", "should",
        "such", "then", "through", "under", "until", "up", "very", "well",
        "yes",
    }
    # fmt: on
)

# "did X do Y" / "where did X go" / "when did X meet Y"
_VERB_PATTERNS = {
    # question-word → candidate relation synonyms
    "where": ("traveled", "visited", "went", "moved", "lives_in", "born_in", "located"),
    "when": ("born", "died", "met", "graduated", "started", "ended", "happened"),
    "who": ("met", "married", "works_with", "parent_of", "child_of", "friend_of"),
    "what": ("bought", "owns", "likes", "prefers", "uses", "studied", "works_on"),
    "how many": ("has", "owns"),
    "which": ("works_at", "studied_at", "lives_in"),
}

# Attribute normalizers — lowercased, stripped.
_REL_ALIASES = {
    "travel": ("traveled_to", "traveled", "went_to", "visited", "trip"),
    "visit": ("visited", "went_to", "traveled_to"),
    "buy": ("bought", "purchased", "owns"),
    "own": ("owns", "bought", "purchased"),
    "live": ("lives_in", "located_in", "moved_to", "based_in"),
    "work": ("works_at", "works_on", "employed_by", "job"),
    "born": ("born_in", "birth_place", "birth_date"),
    "meet": ("met", "knows", "friend_of"),
    "like": ("likes", "prefers", "enjoys"),
    "study": ("studied", "studied_at", "student_of"),
}


def _tokenize(text: str) -> list[str]:
    """Simple alnum tokenizer. Keeps hyphenated words intact."""
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())


def _extract_candidates(query: str) -> tuple[list[str], list[str]]:
    """Pull candidate entities + attribute hints from a natural-language query.

    Entities  = capitalized multi-word sequences in the ORIGINAL query plus
                single capitalized words (Alice, New York).
    Attributes = lowercased verbs that appear in the query or aliases thereof,
                plus any question-word hints ("where" → travel/visit synonyms).

    We return raw strings; the DB side does the lowercase match.
    """
    # Capitalized sequences (preserve original casing, skip leading stopwords).
    ents: list[str] = []
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*)", query):
        chunk = match.group(1).strip()
        if chunk.lower() not in _STOPWORDS and len(chunk) > 1:
            ents.append(chunk)

    # Fallback: bare lowercase tokens that aren't stopwords, surfaced last.
    lowered = [t for t in _tokenize(query) if t not in _STOPWORDS and len(t) > 2]

    attrs: list[str] = []
    q_lower = query.lower()

    # Question-word → attribute synonyms.
    for qw, synonyms in _VERB_PATTERNS.items():
        if re.search(rf"\b{re.escape(qw)}\b", q_lower):
            attrs.extend(synonyms)

    # Stem-like match: if any alias root appears in the query, expand.
    for root, aliases in _REL_ALIASES.items():
        if root in q_lower or any(a in q_lower for a in aliases):
            attrs.extend(aliases)
            attrs.append(root)

    # Also keep verbs seen verbatim.
    for tok in lowered:
        if tok.endswith("ed") or tok.endswith("ing") or tok in {"is", "has", "own", "buy"}:
            attrs.append(tok)

    # Dedupe preserving order.
    seen_e: set[str] = set()
    ents = [e for e in ents if not (e.lower() in seen_e or seen_e.add(e.lower()))]
    seen_a: set[str] = set()
    attrs = [a for a in attrs if not (a in seen_a or seen_a.add(a))]

    return ents, attrs


# ──────────────────────────────────────────────
# DB adapter
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class FactHit:
    """One match from the structured index."""

    entity: str
    relation: str
    value: str
    knowledge_id: int | None
    weight: float
    context: str | None = None

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "relation": self.relation,
            "value": self.value,
            "knowledge_id": self.knowledge_id,
            "weight": self.weight,
            "context": self.context,
        }


# Relations that are structural / noisy — skip them from the returned hits so
# we don't flood downstream with co-occurrence edges.
_NOISY_RELATIONS = frozenset(
    {
        "mentioned_with",
        "co_occurred",
        "semantic_similarity",
        "shared_across",
        "supersedes",  # graph reconciliation artefact, not a domain fact
    }
)


class FactIndex:
    """Thin DB-backed adapter over ``graph_nodes`` / ``graph_edges``.

    The caller owns the connection. We do NOT mutate schema, do NOT run
    migrations, and do NOT prefetch anything — every lookup is a scoped
    indexed SQL query (<1ms at production scale).

    When ``build_semantic_index()`` has been called, ``lookup_semantic()``
    (and the ``semantic=True`` flag on ``lookup_query``) is available. It
    embeds the triple surface form "entity <relation_words> value" once and
    performs a cosine query at recall time — catches rephrased questions
    that substring match misses.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        # Semantic index (lazy — only built when build_semantic_index is called)
        self._semantic_rows: list[tuple] = []
        self._semantic_matrix = None
        self._semantic_model = None

    # -- public -----------------------------------------------------------

    def lookup(
        self,
        entity: str,
        attribute: str | None = None,
        limit: int = 10,
        project: str | None = None,
    ) -> list[FactHit]:
        """Return hits where ``entity`` appears as the subject of an edge.

        - ``attribute`` is matched via substring on ``relation_type`` when set.
        - ``project`` restricts to knowledge rows tagged with that project
          (via ``knowledge_nodes`` → ``knowledge.project``).
        """
        ent_lower = (entity or "").strip().lower()
        if not ent_lower:
            return []

        params: list = [ent_lower]
        clauses: list[str] = ["LOWER(src.name) = ?"]

        if attribute:
            attr_lower = attribute.strip().lower()
            clauses.append("LOWER(e.relation_type) LIKE ?")
            # Substring match: "travel" catches traveled_to, traveled, travels.
            params.append(f"%{attr_lower}%")

        # Noise filter is always on — structural edges are useless for QA.
        placeholders = ",".join(["?"] * len(_NOISY_RELATIONS))
        clauses.append(f"e.relation_type NOT IN ({placeholders})")
        params.extend(_NOISY_RELATIONS)

        if project:
            clauses.append(
                "EXISTS (SELECT 1 FROM knowledge_nodes kn "
                "JOIN knowledge k ON k.id = kn.knowledge_id "
                "WHERE kn.node_id = src.id AND k.project = ?)"
            )
            params.append(project)

        sql = f"""
            SELECT
                src.name  AS entity,
                e.relation_type AS relation,
                tgt.name  AS value,
                e.weight  AS weight,
                e.context AS edge_context,
                (SELECT kn.knowledge_id FROM knowledge_nodes kn
                  WHERE kn.node_id = src.id OR kn.node_id = tgt.id
                  ORDER BY kn.strength DESC LIMIT 1) AS knowledge_id
            FROM graph_edges e
            JOIN graph_nodes src ON src.id = e.source_id
            JOIN graph_nodes tgt ON tgt.id = e.target_id
            WHERE {' AND '.join(clauses)}
            ORDER BY e.weight DESC, e.reinforcement_count DESC
            LIMIT ?
        """
        params.append(int(limit))

        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()

        return [
            FactHit(
                entity=r[0],
                relation=r[1],
                value=r[2],
                weight=float(r[3] or 0.0),
                context=r[4],
                knowledge_id=(int(r[5]) if r[5] is not None else None),
            )
            for r in rows
        ]

    def lookup_query(
        self,
        query: str,
        limit: int = 10,
        project: str | None = None,
    ) -> list[FactHit]:
        """End-to-end: parse NL query, try every entity×attribute combo,
        merge top hits by weight."""
        ents, attrs = _extract_candidates(query)
        if not ents:
            return []

        seen: dict[tuple[str, str, str], FactHit] = {}

        # First pass: every entity crossed with every attribute hint.
        for ent in ents:
            if attrs:
                for attr in attrs:
                    for hit in self.lookup(ent, attr, limit=limit, project=project):
                        key = (hit.entity.lower(), hit.relation, hit.value.lower())
                        # Keep the higher-weight one on collision.
                        if key not in seen or hit.weight > seen[key].weight:
                            seen[key] = hit
            else:
                for hit in self.lookup(ent, None, limit=limit, project=project):
                    key = (hit.entity.lower(), hit.relation, hit.value.lower())
                    if key not in seen or hit.weight > seen[key].weight:
                        seen[key] = hit

        # Sort merged set by weight desc, trim.
        hits = sorted(seen.values(), key=lambda h: h.weight, reverse=True)
        return hits[:limit]

    # -- semantic (embedding-based) lookup -------------------------------

    def build_semantic_index(
        self,
        *,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_edges: int = 20000,
    ) -> int:
        """Embed every non-noisy edge as "subject <relation> object" text.
        Returns the number of entries indexed. Idempotent — skips rebuild
        if already populated.
        """
        if self._semantic_matrix is not None:
            return len(self._semantic_rows)

        placeholders = ",".join(["?"] * len(_NOISY_RELATIONS))
        sql = f"""
            SELECT
                s.name AS subject,
                e.relation_type AS relation,
                t.name AS value,
                e.weight AS weight,
                (SELECT kn.knowledge_id FROM knowledge_nodes kn
                  WHERE kn.node_id = s.id OR kn.node_id = t.id
                  ORDER BY kn.strength DESC LIMIT 1) AS knowledge_id,
                COALESCE(k.project, '') AS project
            FROM graph_edges e
            JOIN graph_nodes s ON s.id = e.source_id
            JOIN graph_nodes t ON t.id = e.target_id
            LEFT JOIN knowledge_nodes kn ON kn.node_id = s.id
            LEFT JOIN knowledge k ON k.id = kn.knowledge_id
            WHERE e.relation_type NOT IN ({placeholders})
            GROUP BY s.id, e.relation_type, t.id
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, (*_NOISY_RELATIONS, int(max_edges))).fetchall()
        if not rows:
            self._semantic_rows = []
            self._semantic_matrix = None
            return 0

        # Surface form for embedding
        texts = [
            f"{r[0]} {str(r[1]).replace('_', ' ')} {r[2]}"
            for r in rows
        ]

        # Lazy import — fastembed already pinned in repo
        from fastembed import TextEmbedding  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        self._semantic_model = TextEmbedding(model_name)
        # fastembed returns an iterator of 1D arrays
        embs = [e for e in self._semantic_model.embed(texts)]
        mat = np.asarray(embs, dtype=np.float32)
        # L2-normalise for cosine via dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.clip(norms, 1e-8, None)

        self._semantic_rows = [tuple(r) for r in rows]
        self._semantic_matrix = mat
        return len(rows)

    def lookup_semantic(
        self,
        query: str,
        limit: int = 5,
        project: str | None = None,
    ) -> list[FactHit]:
        """Cosine-similarity search against the embedded triple index.
        Returns empty if ``build_semantic_index()`` has not been called.
        """
        if self._semantic_matrix is None or not query.strip():
            return []

        import numpy as np  # noqa: PLC0415

        q_emb = next(iter(self._semantic_model.embed([query])))
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-8)

        scores = self._semantic_matrix @ q_emb
        top_idx = np.argsort(-scores)[: limit * 4]  # over-fetch for dedup

        out: list[FactHit] = []
        seen: set[tuple] = set()
        for i in top_idx:
            row = self._semantic_rows[int(i)]
            if project and row[5] and row[5] != project:
                continue
            key = (str(row[0]).lower(), row[1], str(row[2]).lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(FactHit(
                entity=row[0],
                relation=row[1],
                value=row[2],
                weight=float(scores[int(i)]),
                context=f"semantic_score={scores[int(i)]:.3f}",
                knowledge_id=(int(row[4]) if row[4] is not None else None),
            ))
            if len(out) >= limit:
                break
        return out

    def lookup_query_hybrid(
        self,
        query: str,
        limit: int = 10,
        project: str | None = None,
    ) -> list[FactHit]:
        """Union of substring lookup (lookup_query) + semantic lookup, merged
        by score. Semantic gets a boost so it surfaces when substring misses.
        """
        sub_hits = self.lookup_query(query, limit=limit, project=project)
        sem_hits = self.lookup_semantic(query, limit=limit, project=project)

        merged: dict[tuple[str, str, str], FactHit] = {}
        for h in sub_hits:
            key = (h.entity.lower(), h.relation, h.value.lower())
            merged[key] = h
        for h in sem_hits:
            key = (h.entity.lower(), h.relation, h.value.lower())
            if key not in merged:
                merged[key] = h
        return sorted(merged.values(), key=lambda x: x.weight, reverse=True)[:limit]

    def knowledge_ids_for(
        self,
        query: str,
        limit: int = 10,
        project: str | None = None,
    ) -> list[int]:
        """Convenience wrapper for retrieval pipelines: raw ids only."""
        ids: list[int] = []
        seen: set[int] = set()
        for hit in self.lookup_query(query, limit=limit, project=project):
            if hit.knowledge_id is not None and hit.knowledge_id not in seen:
                seen.add(hit.knowledge_id)
                ids.append(hit.knowledge_id)
        return ids

    # -- introspection ----------------------------------------------------

    def stats(self) -> dict:
        """Lightweight counters for dashboards / diagnostics."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT COUNT(*) FROM graph_nodes")
            n_nodes = c.fetchone()[0]
            c.execute(
                "SELECT COUNT(*) FROM graph_edges WHERE relation_type NOT IN "
                f"({','.join(['?'] * len(_NOISY_RELATIONS))})",
                tuple(_NOISY_RELATIONS),
            )
            n_real = c.fetchone()[0]
            c.execute("SELECT COUNT(DISTINCT relation_type) FROM graph_edges")
            n_rel_types = c.fetchone()[0]
        return {
            "nodes": int(n_nodes),
            "real_edges": int(n_real),
            "relation_types": int(n_rel_types),
            "noisy_filtered": sorted(_NOISY_RELATIONS),
        }


# ──────────────────────────────────────────────
# Module-level helpers (for ad-hoc benchmark use)
# ──────────────────────────────────────────────


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open a shared, read-only connection. Useful for benchmarks that
    don't want to mutate the index."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def extract_candidates(query: str) -> tuple[list[str], list[str]]:
    """Re-exported for tests / external callers."""
    return _extract_candidates(query)


__all__ = [
    "FactIndex",
    "FactHit",
    "extract_candidates",
    "open_readonly",
]
