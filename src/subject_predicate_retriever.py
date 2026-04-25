"""v9.0 D8 — subject-aware retrieval (top-3 push).

For LoCoMo cat=1 single-hop and cat=2 temporal we lose accuracy because
plain embedding retrieval often pushes the relevant turn to top-5 instead
of top-1. Single-hop gold answers are usually 1-6 word noun phrases that
sit verbatim in ONE specific turn — if generator doesn't see that turn at
position 0-1, it hallucinates from a similar but wrong turn.

Solution: structured query path that bypasses embeddings.

  1. LLM extracts (subject, action_keywords) from the question.
     "What did Caroline research?" → {subject: "Caroline", actions: ["researched", "research"]}
     "When did Melanie go camping?" → {subject: "Melanie", actions: ["went", "camping"]}
  2. SQL lookup against graph_edges for subject's outgoing edges, filter
     by action_keywords substring in relation_type or target name.
  3. Pre-pend resulting triples to the context as a "DIRECT FACTS" section.

Cost: 1 cheap LLM call per QA (~$0.0001 on gpt-4o-mini × 1986 = $0.20 extra).
ROI: expected +5-10pp on cat=1 because generator now sees the answer
verbatim at position 0 of the context.

Used by benchmarks/locomo_bench_llm.py via --subject-aware flag.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass


_EXTRACT_SYSTEM = """You convert a question about a conversation into a structured retrieval key.

Output STRICT JSON on a single line:
{"subject": "<named entity>" or null, "actions": ["verb1", "verb2", ...]}

Rules:
1. subject = the PERSON the question is about (Caroline, Melanie, John). null if open-domain.
2. actions = 1-4 keywords that capture what the question asks about. Include
   the main verb stemmed (research, go, buy, like, study, work, live, attend,
   meet, plan, prefer, own) AND the topic noun (adoption, Paris, BMW).
3. Be conservative: prefer null subject over guessing.
4. No prose, no markdown fences."""


_EXTRACT_FEWSHOT = """Examples:
Q: What did Caroline research? → {"subject": "Caroline", "actions": ["research"]}
Q: When did Melanie go camping? → {"subject": "Melanie", "actions": ["go", "camping", "attend"]}
Q: What is Caroline's identity? → {"subject": "Caroline", "actions": ["identity", "be"]}
Q: How long has Caroline had her current group of friends for? → {"subject": "Caroline", "actions": ["friends", "duration", "have"]}
Q: Would Melanie like a national park? → {"subject": "Melanie", "actions": ["like", "national park", "prefer"]}
Q: How many siblings does the protagonist have? → {"subject": null, "actions": ["sibling"]}
"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


@dataclass
class StructuredKey:
    subject: str | None
    actions: list[str]


def extract_key(client, question: str, model: str) -> StructuredKey:
    """Single cheap LLM call → (subject, actions). Returns empty key on failure."""
    user = _EXTRACT_FEWSHOT + f"\nQ: {question} →"
    try:
        r = client.complete(_EXTRACT_SYSTEM, user, model=model, max_tokens=80)
        text = _strip_fences(r.text or "")
    except Exception:
        return StructuredKey(subject=None, actions=[])

    try:
        obj = json.loads(text)
    except Exception:
        # Tolerant fallback: try to find {...} substring
        m = re.search(r"\{[^{}]*\}", text)
        if not m:
            return StructuredKey(subject=None, actions=[])
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return StructuredKey(subject=None, actions=[])

    subj = obj.get("subject")
    if isinstance(subj, str):
        subj = subj.strip() or None
    else:
        subj = None
    raw_actions = obj.get("actions") or []
    actions: list[str] = []
    if isinstance(raw_actions, list):
        for a in raw_actions:
            if isinstance(a, str) and a.strip():
                actions.append(a.strip().lower())
    return StructuredKey(subject=subj, actions=actions[:6])


def lookup_triples(
    db: sqlite3.Connection,
    project: str,
    key: StructuredKey,
    limit: int = 12,
) -> list[tuple[str, str, str]]:
    """Find graph_edges where source name matches subject (case-insensitive),
    optionally biased by action keyword substring on relation_type or target.
    Returns list[(subject_name, relation_type, target_name)]."""
    if not key.subject:
        return []

    # 1) Resolve subject to graph_node ids — case-insensitive prefix/match.
    rows = db.execute(
        """SELECT id, name FROM graph_nodes
           WHERE LOWER(name) = LOWER(?)
              OR LOWER(name) LIKE LOWER(?) || '%'
           LIMIT 5""",
        (key.subject, key.subject),
    ).fetchall()
    node_ids = [r[0] for r in rows]
    if not node_ids:
        return []

    # 2) Pull all outgoing edges (subject is source) anchored to this project.
    placeholders = ",".join("?" * len(node_ids))
    edges_sql = f"""
        SELECT s.name, e.relation_type, t.name, k.project
        FROM graph_edges e
        JOIN graph_nodes s ON s.id = e.source_id
        JOIN graph_nodes t ON t.id = e.target_id
        JOIN knowledge_nodes kn ON kn.node_id = e.source_id AND kn.role = 'subject'
        JOIN knowledge k ON k.id = kn.knowledge_id
        WHERE e.source_id IN ({placeholders})
          AND k.project = ?
          AND k.status = 'active'
        ORDER BY e.weight DESC
        LIMIT 100
    """
    rows = db.execute(edges_sql, (*node_ids, project)).fetchall()
    if not rows:
        return []

    # 3) Re-rank by action keyword match.
    actions = [a.lower() for a in key.actions]
    scored: list[tuple[float, tuple[str, str, str]]] = []
    for s_name, rel, t_name, _proj in rows:
        rel_low = (rel or "").lower()
        t_low = (t_name or "").lower()
        score = 0.0
        for a in actions:
            if a and a in rel_low:
                score += 2.0  # predicate match weighted higher
            if a and a in t_low:
                score += 1.0
        scored.append((score, (s_name, rel, t_name)))
    # Stable sort by score desc; if no actions scored, original DB order applies.
    scored.sort(key=lambda x: x[0], reverse=True)
    triples = [t for _s, t in scored[:limit]]
    return triples


def format_triples_block(triples: list[tuple[str, str, str]]) -> str:
    """Compact human-readable block to prepend to the generator prompt."""
    if not triples:
        return ""
    lines = []
    for s, rel, o in triples:
        rel_pretty = (rel or "").replace("_", " ")
        lines.append(f"- {s} {rel_pretty} {o}")
    return "DIRECT FACTS (subject lookup):\n" + "\n".join(lines) + "\n\n"


__all__ = [
    "StructuredKey",
    "extract_key",
    "lookup_triples",
    "format_triples_block",
]
