#!/usr/bin/env python3
"""Extract (subject, relation, object) triples from synthesized_fact rows
using OpenAI (default gpt-4o-mini). Writes into graph_nodes + graph_edges so
FactIndex can lookup (entity, attribute) → values.

Why this exists: the production triple_extraction_queue uses Ollama
(qwen2.5-coder:7b). On a fresh LoCoMo bench DB that model may not be
installed, so graph_edges stays empty. For the v9 LoCoMo push we pay ~$0.3
for gpt-4o-mini over ~4K facts to get structured edges for L2.

Usage:
    python scripts/extract_triples_openai.py \
        --db-path /tmp/locomo_bench_db \
        --concurrency 20 \
        --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
from _llm_adapter import LLMClient  # noqa: E402


# v9 D3 — schema-specific predicate list. Constraining the LLM to a closed
# vocabulary (vs the v1 "any snake_case relation" prompt) cuts predicate
# variance ~10x — same logical relation no longer splits into 4 surface forms
# (e.g. "went_to" / "traveled_to" / "visited" / "vacationed_in" → all become
# person_went_to). Lookup-by-attribute then actually finds matches in
# graph_edges, which is the bottleneck on multi-hop (cat=3) accuracy.
CANONICAL_PREDICATES_V2: list[str] = [
    # identity / demographics
    "person_name",
    "person_age",
    "person_lives_in",
    "person_born_in",
    "person_born_on",
    # work / education
    "person_works_at",
    "person_works_as",
    "person_studied_at",
    "person_studied_subject",
    # actions / events
    "person_did",
    "person_went_to",
    "person_visited",
    "person_attended",
    "person_plans_to",
    "person_received",
    "person_gave",
    # possessions / commerce
    "person_owns",
    "person_bought",
    "person_paid",
    "person_has_pet",
    # preferences / opinions
    "person_likes",
    "person_dislikes",
    "person_prefers",
    "person_has_hobby",
    # social
    "person_met",
    "person_knows",
    "person_relationship",  # object encodes "<relation>:<other_person>" e.g. "sister:Anna"
    "person_asked_about",
    # event-side facts (use "event:<short_label>" as subject)
    "event_occurred_on",
    "event_location",
]
# Closed set used for validation. "other:<suffix>" is a separate escape
# hatch handled by _normalize_predicate.
_CANONICAL_SET = set(CANONICAL_PREDICATES_V2)

SYSTEM_PROMPT_V1 = """Extract relational facts from a short sentence.

Output strict JSON on a single line:
{"triples": [{"s": "Subject", "r": "relation", "o": "Object"}, ...]}

Rules:
1. Subject = named entity (person, place, company, product). ALWAYS preserve original casing.
2. Relation = short snake_case verb phrase (traveled_to, works_at, owns, met, studied_at, born_in, lives_in, prefers, bought, likes, has_child).
3. Object = named entity or concrete value (date, place, thing).
4. Ignore vague/generic facts (e.g. "X is happy"). Skip opinions and questions.
5. Include date-only facts as relation=date_of_<event> when present.
6. 0-3 triples per sentence.

No prose, no markdown fences."""

SYSTEM_PROMPT_V2 = """Extract relational facts using a CLOSED predicate vocabulary.

Output strict JSON on a single line:
{"triples": [{"s": "Subject", "r": "predicate", "o": "Object"}, ...]}

ALLOWED PREDICATES (use EXACTLY these — no synonyms, no variations):
  person_name, person_age, person_lives_in, person_born_in, person_born_on,
  person_works_at, person_works_as, person_studied_at, person_studied_subject,
  person_did, person_went_to, person_visited, person_attended, person_plans_to,
  person_received, person_gave,
  person_owns, person_bought, person_paid, person_has_pet,
  person_likes, person_dislikes, person_prefers, person_has_hobby,
  person_met, person_knows, person_relationship, person_asked_about,
  event_occurred_on, event_location,
  other

Mapping cheatsheet (collapse synonyms to canonical predicate):
  traveled to / vacationed in / went on a trip to → person_went_to
  works as / is a / her job is → person_works_as
  is married to / dating / is the wife of → person_relationship (object: "spouse:NAME")
  has a daughter named X → person_relationship (object: "daughter:X")
  her dog is Rex → person_has_pet (object: "dog:Rex")
  paid $X for Y → person_paid (object: "$X for Y")
  birthday is X → person_born_on (object: ISO date)

Rules:
  1. Subject = named entity, original casing preserved.
  2. Predicate must be one of the ALLOWED list — pick the closest fit.
  3. Object = named entity or concrete value (place, ISO date, amount, name).
  4. Use ISO dates "YYYY-MM-DD" or "YYYY-MM" when month/day known.
  5. Skip opinions/feelings/questions. Skip vague facts ("X is happy").
  6. 0-3 triples per sentence. Quality over quantity.
  7. Use predicate "other" ONLY when nothing in the list fits AND the fact
     is concrete/factual. Then put your free-form predicate as "r" prefix:
     {"s": "X", "r": "other:plays_instrument", "o": "violin"}.

Few-shot:

INPUT: "Caroline went to Paris on May 7, 2023."
OUTPUT: {"triples": [{"s": "Caroline", "r": "person_went_to", "o": "Paris"}, {"s": "Caroline", "r": "person_did", "o": "trip to Paris on 2023-05-07"}]}

INPUT: "Melanie has a 5-year-old daughter named Anna."
OUTPUT: {"triples": [{"s": "Melanie", "r": "person_relationship", "o": "daughter:Anna"}, {"s": "Anna", "r": "person_age", "o": "5"}]}

INPUT: "John bought a BMW X5 in June 2023 for $65,000."
OUTPUT: {"triples": [{"s": "John", "r": "person_bought", "o": "BMW X5"}, {"s": "John", "r": "person_paid", "o": "$65,000 for BMW X5"}, {"s": "John", "r": "person_did", "o": "bought BMW X5 in 2023-06"}]}

INPUT: "Bob really enjoys playing tennis on weekends."
OUTPUT: {"triples": [{"s": "Bob", "r": "person_has_hobby", "o": "tennis"}, {"s": "Bob", "r": "person_likes", "o": "playing tennis on weekends"}]}

INPUT: "Hi, how are you?"
OUTPUT: {"triples": []}

No prose, no markdown fences. The first character must be '{'."""

PROMPT_VERSIONS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
}

SYSTEM_PROMPT = SYSTEM_PROMPT_V2  # default = v2 schema-specific


def _id(text: str) -> str:
    return hashlib.md5(text.lower().encode()).hexdigest()


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _normalize_predicate(rel: str, prompt_version: str) -> str | None:
    """Validate predicate against canonical vocabulary (v2) or accept any
    snake_case form (v1). Returns normalized predicate or None to drop."""
    rel = rel.strip().lower().replace(" ", "_")
    if not rel:
        return None
    if prompt_version != "v2":
        return rel  # v1: accept anything reasonable
    # v2: must be a canonical predicate, OR start with "other:" (escape hatch)
    if rel in _CANONICAL_SET:
        return rel
    if rel.startswith("other:") and 6 < len(rel) <= 60:
        return rel
    return None  # drop synonyms / hallucinated predicates


def extract(client: LLMClient, text: str, model: str, prompt_version: str = "v2") -> list[dict]:
    system = PROMPT_VERSIONS.get(prompt_version, SYSTEM_PROMPT)
    try:
        r = client.complete(system, text, model=model, max_tokens=240)
    except Exception:
        return []
    raw = _strip_fences(r.text)
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    triples = obj.get("triples", [])
    if not isinstance(triples, list):
        return []
    out: list[dict] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        s = str(t.get("s", "")).strip()
        rel_raw = str(t.get("r", ""))
        o = str(t.get("o", "")).strip()
        if not s or not o:
            continue
        rel = _normalize_predicate(rel_raw, prompt_version)
        if not rel:
            continue
        if len(s) > 120 or len(o) > 200 or len(rel) > 60:
            continue
        out.append({"s": s, "r": rel, "o": o})
    return out


def upsert_node(cur: sqlite3.Cursor, name: str, typ: str = "entity") -> str:
    nid = _id(name)
    cur.execute(
        "INSERT OR IGNORE INTO graph_nodes(id,type,name) VALUES(?,?,?)",
        (nid, typ, name),
    )
    return nid


def upsert_edge(
    cur: sqlite3.Cursor, src_id: str, tgt_id: str, relation: str, weight: float, context: str
) -> None:
    eid = _id(f"{src_id}|{relation}|{tgt_id}")
    cur.execute(
        "INSERT OR IGNORE INTO graph_edges(id,source_id,target_id,relation_type,weight,context) "
        "VALUES(?,?,?,?,?,?)",
        (eid, src_id, tgt_id, relation, weight, context),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="/tmp/locomo_bench_db")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "auto"])
    ap.add_argument("--project-prefix", default="locomo_",
                    help="Scope to LoCoMo conversation projects")
    ap.add_argument("--types", default="synthesized_fact,fact",
                    help="Comma-separated knowledge.type values to process")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-version", default="v2", choices=["v1", "v2"],
                    help="v9 D3: v2 = closed canonical-predicate vocabulary (default); "
                         "v1 = legacy free-form snake_case relation.")
    ap.add_argument("--reset", action="store_true",
                    help="Wipe graph_edges, graph_nodes, knowledge_nodes scoped to --project-prefix "
                         "before extraction. Use when re-running v2 over data extracted with v1.")
    args = ap.parse_args()

    client = LLMClient(provider=args.provider, default_model=args.model)
    print(f"[triples] provider={client.provider} model={args.model} prompt={args.prompt_version}")

    db_file = f"{args.db_path}/memory.db"
    main_conn = sqlite3.connect(db_file, check_same_thread=False)
    main_conn.row_factory = sqlite3.Row

    if args.reset:
        # Scope deletion to the project prefix to avoid blowing away other data.
        # We delete edges/nodes that link to knowledge rows of this project AND
        # are not referenced by other projects.
        cur = main_conn.cursor()
        n_kn = cur.execute(
            "DELETE FROM knowledge_nodes WHERE knowledge_id IN "
            "(SELECT id FROM knowledge WHERE project LIKE ?)",
            (f"{args.project_prefix}%",),
        ).rowcount
        # Drop edges & nodes that became orphans after knowledge_nodes wipe.
        n_edges = cur.execute(
            "DELETE FROM graph_edges WHERE id NOT IN "
            "(SELECT DISTINCT e.id FROM graph_edges e "
            " JOIN knowledge_nodes kn ON kn.node_id IN (e.source_id, e.target_id))"
        ).rowcount
        n_nodes = cur.execute(
            "DELETE FROM graph_nodes WHERE id NOT IN "
            "(SELECT DISTINCT node_id FROM knowledge_nodes)"
        ).rowcount
        main_conn.commit()
        print(f"[triples] reset: knowledge_nodes={n_kn} graph_edges={n_edges} graph_nodes={n_nodes}")

    type_list = [t.strip() for t in args.types.split(",") if t.strip()]
    placeholders = ",".join("?" * len(type_list))
    sql = (f"SELECT id, content, project FROM knowledge "
           f"WHERE status='active' AND type IN ({placeholders}) AND project LIKE ?")
    params: list = [*type_list, f"{args.project_prefix}%"]
    if args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"
    rows = main_conn.execute(sql, params).fetchall()
    print(f"[triples] {len(rows)} rows to scan, concurrency={args.concurrency}")

    tlocal = threading.local()

    def get_conn():
        c = getattr(tlocal, "conn", None)
        if c is None:
            c = sqlite3.connect(db_file, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            tlocal.conn = c
        return c

    stats = {"triples": 0, "rows_with_triples": 0, "rows_empty": 0, "tokens_in": 0, "tokens_out": 0}
    stats_lock = threading.Lock()

    def work(row: sqlite3.Row) -> None:
        content = (row["content"] or "").strip()
        if not content:
            return
        triples = extract(client, content, args.model, prompt_version=args.prompt_version)
        if not triples:
            with stats_lock:
                stats["rows_empty"] += 1
            return
        conn = get_conn()
        cur = conn.cursor()
        kid = int(row["id"])
        try:
            for t in triples:
                src_id = upsert_node(cur, t["s"])
                tgt_id = upsert_node(cur, t["o"], typ="value")
                upsert_edge(cur, src_id, tgt_id, t["r"], 1.0, f"from_knowledge={kid}")
                # Link knowledge → both ends so FactIndex can surface the row.
                cur.execute(
                    "INSERT OR IGNORE INTO knowledge_nodes(knowledge_id,node_id,role,strength) "
                    "VALUES(?,?,?,?)",
                    (kid, src_id, "subject", 1.0),
                )
                cur.execute(
                    "INSERT OR IGNORE INTO knowledge_nodes(knowledge_id,node_id,role,strength) "
                    "VALUES(?,?,?,?)",
                    (kid, tgt_id, "object", 0.8),
                )
            conn.commit()
            with stats_lock:
                stats["triples"] += len(triples)
                stats["rows_with_triples"] += 1
        except Exception as e:  # noqa: BLE001
            print(f"  write err kid={kid}: {e}", file=sys.stderr)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, r): r for r in rows}
        completed = 0
        for _ in as_completed(futs):
            completed += 1
            if completed % 500 == 0 or completed == len(rows):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.01)
                print(f"  [{completed}/{len(rows)}] rate={rate:.1f}/s "
                      f"triples={stats['triples']} empty={stats['rows_empty']}",
                      flush=True)

    elapsed = time.time() - t0
    print(f"[triples] done {stats['rows_with_triples']} rows → {stats['triples']} triples, "
          f"{stats['rows_empty']} empty, elapsed={elapsed:.1f}s")
    main_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
