"""Distill per-turn raw content into short, retrievable factual sentences.

Rationale: Mem0's main retrieval advantage on LoCoMo is that it stores
LLM-distilled facts ("Calvin bought a mansion in Japan on 2023-03-15.")
as the indexed chunks, instead of the raw conversational turn the fact
originated in. Query → fact retrieval is much closer to 1-nearest-neighbor
than query → turn retrieval which requires reasoning.

This module replicates that behaviour for Claude Total Memory:
  - For each raw knowledge record, call Haiku with a compact prompt
    that extracts 1-3 short factual sentences.
  - Save each sentence as a new knowledge record with
    type='synthesized_fact', carrying `parent_knowledge_id` in context
    and inheriting project + session_id + key tags so that project
    filtering and date filtering keep working.
  - Idempotent: records whose parent already has synthesized_fact
    children are skipped.

Usage:
    python fact_synthesizer.py --db-path /tmp/locomo_bench_db --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HAIKU_MODEL = "claude-haiku-4-5-20251001"

# v9 D2 — LoCoMo-tuned fact extraction prompt.
#
# Differences vs v1:
#   * Output format mirrors LoCoMo gold answers: "Entity ACTION specific_object
#     (on absolute_date)". Helps single-hop (cat 1) — top-K → fact match → done.
#   * Six few-shot pairs covering: explicit date, relative date resolution,
#     small-talk → empty, multi-fact turn, opinion/preference, question with
#     topic capture.
#   * Mandatory absolute-date resolution when turn metadata carries timestamp.
#   * 1-3 facts cap (down from 3-6) — LoCoMo ablation showed aggressive
#     extraction hurts -1pp overall (noise).
SYSTEM_PROMPT_V2 = """You distill one conversation turn into standalone facts in LoCoMo answer style.

A "fact" is one atomic statement that:
  - Names the speaker (or person referenced) as the subject explicitly.
  - States ONE concrete action, attribute, preference, plan, opinion, or relationship.
  - Includes ALL specific details from the turn: named entities, places, prices, counts, brand names.
  - Resolves relative dates using the TURN_DATE provided (e.g. "last Tuesday" → "Tuesday 2 May 2023").
  - Reads naturally as a sentence that could match a question's gold answer.

Rules:
  1. 1-3 facts per turn. Prefer fewer high-quality facts over many noisy ones.
  2. Skip pure small-talk, greetings, backchannels ("Hey!", "Yeah", "lol", emotional reactions only).
  3. Questions → extract the topic as a fact ("Alice is asking about adoption agencies").
  4. Inferences allowed only when clearly implied — never speculative.
  5. Use absolute dates whenever derivable from TURN_DATE; never echo "last week" / "yesterday".
  6. No pronouns as subject — replace with name from context.

Few-shot examples:

TURN_DATE: 2023-05-08
TURN: "Caroline: I joined the LGBTQ support group on May 7th, it's been amazing."
FACTS: ["Caroline joined the LGBTQ support group on May 7, 2023.", "Caroline finds the LGBTQ support group amazing."]

TURN_DATE: 2023-06-12
TURN: "Melanie: My daughter just turned 5 last Tuesday."
FACTS: ["Melanie's daughter turned 5 years old on Tuesday June 6, 2023.", "Melanie has a 5-year-old daughter."]

TURN_DATE: 2023-07-01
TURN: "John: I bought a new BMW X5 for $65,000 last month."
FACTS: ["John bought a new BMW X5 in June 2023.", "John paid $65,000 for the BMW X5."]

TURN_DATE: 2023-08-15
TURN: "Hi Mel, how are you?"
FACTS: []

TURN_DATE: 2023-09-04
TURN: "Alice: What time is it? Maybe we should head out."
FACTS: ["Alice is asking about the current time.", "Alice suggested leaving."]

TURN_DATE: 2023-10-20
TURN: "Bob: I really prefer indie coffee shops over Starbucks — too crowded for me."
FACTS: ["Bob prefers indie coffee shops over Starbucks.", "Bob finds Starbucks too crowded."]

Respond with STRICT JSON on a single line:
{"facts": ["fact 1.", "fact 2.", ...]}

No prose, no markdown fences. The first line must start with '{'."""

# Legacy prompt kept available for A/B comparisons via --prompt-version v1.
SYSTEM_PROMPT_V1 = """You distill one conversation turn into standalone facts about the speakers.

A "fact" captures ANY concrete piece of information:
  - what someone did, said, felt, plans, prefers, likes/dislikes
  - events, purchases, trips, jobs, relationships, hobbies, opinions
  - dates, places, objects, prices, counts, brand names
  - personality traits, values, beliefs, habits, routines
  - family/friends mentioned, people met, advice given/received

Rules:
  1. Each fact must be understandable WITHOUT the original turn (self-contained).
  2. Include the speaker's name as subject when the fact is about them ("Alice prefers quiet cafés").
  3. Include ALL specific details: dates, places, objects, prices, counts, named entities.
  4. Ignore small-talk, backchannel ("Hey!", "Yeah"), greetings, pure emotional reactions.
  5. Questions are NOT facts, but DO extract the topic someone is asking about ("Alice is asking about adoption agencies").
  6. Inferences are allowed when clearly implied (e.g. "Alice enjoys cooking" from "I made dinner last night and it was fun").
  7. 3-6 facts per turn whenever possible. Aim for maximum coverage; it's better to err on the side of extracting more facts than fewer. Only return empty if the turn is pure small-talk.

Respond with STRICT JSON on a single line:
{"facts": ["fact 1.", "fact 2.", ...]}

No prose, no markdown fences."""

SYSTEM_PROMPT = SYSTEM_PROMPT_V2  # default = LoCoMo-tuned v2

PROMPT_VERSIONS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
}


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _call_llm(client, system: str, user: str, max_tokens: int = 200,
              model: str | None = None) -> tuple[str, int, int]:
    """Delegate to the benchmarks/_llm_adapter.LLMClient which supports both
    Anthropic and OpenAI. The legacy _call_haiku signature is preserved as a
    thin alias so any external callers keep working."""
    resolved = model or getattr(client, "_default_model", None) or HAIKU_MODEL
    r = client.complete(system, user, model=resolved, max_tokens=max_tokens)
    return r.text, r.input_tokens, r.output_tokens


_call_haiku = _call_llm  # backwards-compat alias


def synthesize_facts(
    client,
    content: str,
    model: str | None = None,
    turn_date: str | None = None,
    prompt_version: str = "v2",
) -> tuple[list[str], int, int]:
    """Distill a turn into 1-3 facts. Returns (facts, tokens_in, tokens_out).

    ``turn_date`` is rendered into the prompt so the LLM can resolve relative
    dates ("last Tuesday") into absolute ones — critical for LoCoMo cat=2
    (temporal) accuracy. Pass ISO date or any human-readable form.
    ``prompt_version`` ∈ {"v1", "v2"}; v2 is the LoCoMo-tuned few-shot prompt.
    """
    system = PROMPT_VERSIONS.get(prompt_version, SYSTEM_PROMPT)
    if turn_date and prompt_version == "v2":
        prompt = f"TURN_DATE: {turn_date}\nTURN: {content}\n\nFACTS:"
    else:
        prompt = f"Turn content:\n{content}\n\nExtract facts."
    try:
        raw, tin, tout = _call_llm(client, system, prompt, max_tokens=200, model=model)
    except Exception:
        return [], 0, 0
    text = _strip_fences(raw)
    try:
        obj = json.loads(text)
        facts = obj.get("facts", [])
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if isinstance(f, str) and f.strip()], tin, tout
    except Exception:
        pass
    return [], tin, tout


def _parse_tags(tags_raw) -> list[str]:
    if not tags_raw:
        return []
    if isinstance(tags_raw, list):
        return list(tags_raw)
    try:
        v = json.loads(tags_raw)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return [t.strip() for t in str(tags_raw).split(",") if t.strip()]


def save_synth_fact(conn: sqlite3.Connection, parent_id: int, fact: str,
                    project: str, session_id: str, parent_tags: list[str],
                    ts: str) -> int:
    """Insert a synthesized_fact knowledge row linked to parent. Returns new id."""
    new_tags = list(parent_tags)
    if "synthesized_fact" not in new_tags:
        new_tags.append("synthesized_fact")
    context = f"distilled_from={parent_id}"
    conn.execute(
        """INSERT INTO knowledge
             (session_id, project, type, content, context, tags, confidence,
              created_at, last_confirmed, recall_count, status, branch)
           VALUES (?, ?, 'synthesized_fact', ?, ?, ?, 1.0, ?, ?, 0, 'active', '')""",
        (session_id, project, fact, context, json.dumps(new_tags), ts, ts),
    )
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    return int(row[0])


def has_synth_child(conn: sqlite3.Connection, parent_id: int) -> bool:
    q = "SELECT 1 FROM knowledge WHERE type='synthesized_fact' AND context LIKE ? LIMIT 1"
    return conn.execute(q, (f"%distilled_from={parent_id}%",)).fetchone() is not None


def select_parents(conn: sqlite3.Connection, project_prefix: str | None,
                   limit: int) -> list[sqlite3.Row]:
    sql = ("SELECT id, content, project, session_id, tags, created_at "
           "FROM knowledge WHERE status='active' AND type='fact'")
    params: list = []
    if project_prefix:
        sql += " AND project LIKE ?"
        params.append(f"{project_prefix}%")
    if limit > 0:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="/tmp/locomo_bench_db")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Cap parent rows processed (0 = all)")
    ap.add_argument("--project-prefix", default="locomo_", help="Filter parent rows by project prefix")
    ap.add_argument("--reset", action="store_true", help="Delete existing synthesized_fact rows before running")
    ap.add_argument("--provider", default="auto", choices=["auto", "openai", "anthropic"],
                    help="LLM provider (auto-detects from --model)")
    ap.add_argument("--model", default="gpt-4o",
                    help="Generator model. Default gpt-4o (v9 D2: stronger than mini for "
                         "LoCoMo-style atomic facts). Flip to gpt-4o-mini to economize.")
    ap.add_argument("--prompt-version", default="v2", choices=["v1", "v2"],
                    help="Fact-extraction prompt version. v2 = LoCoMo-tuned few-shot.")
    ap.add_argument("--max-facts-per-turn", type=int, default=3,
                    help="Cap facts saved per parent turn. v9 D2: 3 (down from 6) to reduce noise.")
    args = ap.parse_args()

    if args.reset:
        reset_conn = sqlite3.connect(f"{args.db_path}/memory.db")
        n = reset_conn.execute(
            "DELETE FROM knowledge WHERE type='synthesized_fact' AND project LIKE ?",
            (f"{args.project_prefix}%",)
        ).rowcount
        reset_conn.commit()
        print(f"[synth] reset: deleted {n} existing synthesized_fact rows")

    # Main connection (read-only for candidate selection)
    main_conn = sqlite3.connect(f"{args.db_path}/memory.db", check_same_thread=False)
    main_conn.row_factory = sqlite3.Row
    main_conn.execute("PRAGMA journal_mode=WAL")

    parents = select_parents(main_conn, args.project_prefix, args.limit)
    print(f"[synth] {len(parents)} parent turns to process  concurrency={args.concurrency}")

    # Per-thread connection factory
    tlocal = threading.local()

    def get_conn():
        c = getattr(tlocal, "conn", None)
        if c is None:
            c = sqlite3.connect(f"{args.db_path}/memory.db", check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            tlocal.conn = c
        return c

    # Unified provider client (OpenAI or Anthropic) via benchmarks/_llm_adapter.
    _bench_dir = Path(__file__).resolve().parent.parent / "benchmarks"
    sys.path.insert(0, str(_bench_dir))
    from _llm_adapter import LLMClient  # noqa: PLC0415
    client = LLMClient(provider=args.provider, default_model=args.model)
    # Stash default model for _call_llm helpers that don't receive it explicitly.
    client._default_model = args.model  # type: ignore[attr-defined]
    print(f"[synth] provider={client.provider} model={args.model}")

    def work(parent_row: sqlite3.Row) -> dict:
        conn = get_conn()
        parent_id = parent_row["id"]
        if has_synth_child(conn, parent_id):
            return {"parent": parent_id, "status": "skip", "facts": 0, "in": 0, "out": 0}
        # Pull the parent turn's timestamp; let LoCoMo prompt resolve relative
        # dates against it. Fallback to created_at for backward compat.
        turn_date = parent_row["created_at"] if "created_at" in parent_row.keys() else None
        facts, tin, tout = synthesize_facts(
            client,
            parent_row["content"] or "",
            model=args.model,
            turn_date=turn_date,
            prompt_version=args.prompt_version,
        )
        if not facts:
            return {"parent": parent_id, "status": "empty", "facts": 0, "in": tin, "out": tout}
        tags = _parse_tags(parent_row["tags"])
        ts = parent_row["created_at"]
        n_saved = 0
        for f in facts[: args.max_facts_per_turn]:
            if len(f) < 8 or len(f) > 400:  # sanity limits
                continue
            try:
                save_synth_fact(
                    conn,
                    parent_id=parent_id,
                    fact=f,
                    project=parent_row["project"],
                    session_id=parent_row["session_id"],
                    parent_tags=tags,
                    ts=ts,
                )
                n_saved += 1
            except Exception as e:
                print(f"  save error parent={parent_id}: {e}", file=sys.stderr)
        try:
            conn.commit()
        except Exception:
            pass
        return {"parent": parent_id, "status": "ok", "facts": n_saved,
                "in": tin, "out": tout}

    t0 = time.time()
    done = skip = empty = failed = 0
    tokens_in = tokens_out = facts_saved = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(work, row): row for row in parents}
        completed = 0
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                failed += 1
                print(f"  fut error: {e}", file=sys.stderr)
                continue
            if res["status"] == "ok":
                done += 1
            elif res["status"] == "skip":
                skip += 1
            else:
                empty += 1
            tokens_in += res["in"]
            tokens_out += res["out"]
            facts_saved += res["facts"]
            completed += 1
            if completed % 200 == 0 or completed == len(parents):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.01)
                eta = (len(parents) - completed) / max(rate, 0.01)
                print(f"  [{completed}/{len(parents)}] done={done} skip={skip} empty={empty} "
                      f"facts={facts_saved} rate={rate:.1f}/s eta={eta:.0f}s  "
                      f"tok={tokens_in}→{tokens_out}", flush=True)

    elapsed = time.time() - t0
    print(f"[synth] final: {done} parents produced {facts_saved} facts, "
          f"{skip} skipped, {empty} empty, {failed} failed. "
          f"elapsed={elapsed:.1f}s  cost≈${(tokens_in/1e6 + tokens_out*5/1e6):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
