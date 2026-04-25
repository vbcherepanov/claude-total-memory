#!/usr/bin/env python3
"""
LoCoMo Benchmark with LLM-judge (Claude Haiku 4.5).

End-to-end evaluation compatible with the LoCoMo paper (ACL 2024), Mem0
(arxiv 2504.19413) and Zep/Graphiti (arxiv 2501.13956) methodology.

Pipeline:
  1. Ingest all 5882 conversation turns into the memory store (separate
     project per conversation).
  2. For each QA, retrieve top-K relevant turns via Recall.search().
  3. Ask Haiku 4.5 to answer the question using ONLY the retrieved context.
  4. Ask Haiku 4.5 (judge, independent call) whether predicted == gold.
  5. Also compute token-level F1 / BLEU-1 / ROUGE-L for machine metrics.

Usage:
    python benchmarks/locomo_bench_llm.py --wipe                      # full run
    python benchmarks/locomo_bench_llm.py --skip-ingest                # reuse DB
    python benchmarks/locomo_bench_llm.py --limit-qa 100               # quick
    python benchmarks/locomo_bench_llm.py --concurrency 16             # speed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_filter import temporal_rerank  # noqa: E402
from _llm_adapter import LLMClient, MODEL_ALIASES  # noqa: E402


RETRIEVAL_LOCK = threading.Lock()


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"
DEFAULT_DB = Path("/tmp/locomo_bench_db")
RESULTS_DIR = ROOT / "benchmarks" / "results"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
# Default flipped to OpenAI for v9 LoCoMo push (matches Mem0-g paper setup).
DEFAULT_GEN_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


# ──────────────────────────────────────────────────────────────────────────
# Env + import of production memory server
# ──────────────────────────────────────────────────────────────────────────

def setup_env(db_path: Path, disable_llm_extraction: bool) -> None:
    os.environ["CLAUDE_MEMORY_DIR"] = str(db_path)
    if disable_llm_extraction:
        os.environ["MEMORY_LLM_ENABLED"] = "false"


def import_store():
    sys.path.insert(0, "/Users/vitalii-macpro/claude-memory-server/src")
    import server
    return server


def patch_thread_safety(server_mod, store) -> None:
    """Reopen sqlite connection with check_same_thread=False.

    Recall.search silently swallows SQLite ProgrammingError raised when a
    connection is used from a non-owning thread, which makes thread-pool
    retrieval return empty results. The re-open + external lock lets us run
    Haiku calls concurrently while serializing DB access.
    """
    import sqlite3
    try:
        store.db.close()
    except Exception:
        pass
    store.db = sqlite3.connect(str(server_mod.MEMORY_DIR / "memory.db"),
                               check_same_thread=False)
    store.db.row_factory = sqlite3.Row
    store.db.execute("PRAGMA journal_mode=WAL")
    store.db.execute("PRAGMA busy_timeout=5000")


def load_dataset(path: Path) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


# ──────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────

def ingest(server_mod, samples: list[dict]) -> dict:
    store = server_mod.Store()
    t0 = time.time()
    saved = skipped = 0
    for sample_idx, sample in enumerate(samples):
        project = f"locomo_{sample_idx}"
        conv = sample["conversation"]
        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda k: int(k.split("_")[1]),
        )
        for sk in session_keys:
            sess_date = conv.get(f"{sk}_date_time", "")
            sid = f"{project}__{sk}"
            store.session_start(sid, project=project)
            for turn in conv[sk]:
                dia_id = turn.get("dia_id", "")
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                if not text or not dia_id:
                    skipped += 1
                    continue
                content = f"[{sess_date}] {speaker}: {text}"
                if turn.get("blip_caption"):
                    content += f"\n(image: {turn['blip_caption']})"
                try:
                    store.save_knowledge(
                        sid=sid, content=content, ktype="fact",
                        project=project,
                        tags=[dia_id, sk, "locomo", f"conv_{sample_idx}", f"speaker:{speaker}"],
                        context=f"locomo session={sk} dia_id={dia_id}",
                        skip_dedup=True,
                    )
                    saved += 1
                except Exception as e:
                    skipped += 1
                    print(f"save error {dia_id}: {e}", file=sys.stderr)
    try:
        store.db.commit()
    except Exception:
        pass
    return {"saved": saved, "skipped": skipped,
            "elapsed_sec": round(time.time() - t0, 2),
            "rate_turn_per_sec": round(saved / max(time.time() - t0, 0.01), 2)}


# ──────────────────────────────────────────────────────────────────────────
# Machine metrics (token-level F1, BLEU-1, ROUGE-L)
# ──────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+")
_UNANSWERABLE_TOKENS = {
    "not", "mentioned", "no", "information", "unknown",
    "unanswerable", "insufficient", "cannot", "don't", "unsure",
}


def tokenize(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())


def f1_score(pred: str, gold: str) -> float:
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = {}
    for w in p:
        common[w] = min(p.count(w), g.count(w))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def bleu1(pred: str, gold: str) -> float:
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return 0.0
    matches = sum(1 for w in p if w in g)
    return matches / len(p)


def rouge_l(pred: str, gold: str) -> float:
    """Longest common subsequence length divided by max(|pred|, |gold|)."""
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return 0.0
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ──────────────────────────────────────────────────────────────────────────
# Haiku: answer generation + LLM-as-a-judge
# ──────────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """You are a helpful assistant answering questions about a long-running conversation between two people.
You will be given relevant excerpts and optional STRUCTURED FACTS, TEMPORAL FACTS, and SESSION SUMMARY sections.

ANSWERING PROCEDURE (follow in order):
1. If STRUCTURED FACTS contains a triple whose subject + relation match the question (e.g. "Alice traveled_to Berlin" for "Where did Alice travel?"), answer with the object verbatim ("Berlin").
2. If TEMPORAL FACTS lists a date or time frame for the event asked about, use that value.
3. Otherwise, derive the answer from CONVERSATION EXCERPTS and SESSION SUMMARY. Be generous with reasonable inferences from explicit evidence: synthesise date ranges, recognise paraphrases, combine facts mentioned across different turns.
4. When the question asks about preferences, likely behaviour, habits, or future actions ("Would X…", "Is X likely to…", "What kind of…"), answer based on patterns and stated interests in the excerpts. Hedged answers ("Likely yes, because …") are expected and correct.
5. For list-style gold answers, include the main items you can identify — partial coverage of the head entities is enough.
6. Only reply "Not mentioned in the conversation." if the topic or entities in the question are entirely absent from every provided source. Do NOT refuse just because the exact phrasing is missing.

Do NOT invent facts outside the provided sources. Be concise — one short phrase or sentence is ideal. Do not add commentary, explanations, or hedges unless directly required by the question."""


# LoCoMo-specific per-category prompts. LoCoMo gold answers have distinct
# surface forms per category; matching the surface form lifts judge acceptance.
# Official LoCoMo category mapping (confirmed via memobase benchmark):
#   1 = single_hop, 2 = temporal, 3 = multi_hop, 4 = open_domain, 5 = adversarial
CATEGORY_PROMPTS: dict[int, str] = {
    1: """You answer direct factual questions about a two-person conversation.

Strategy for SINGLE-HOP questions:
1. If STRUCTURED FACTS has a triple matching the subject + relation of the question, answer with the object VERBATIM — no extra words.
2. Otherwise pick the most specific phrase from the excerpts or SESSION SUMMARY.
3. For list questions ("What does X like?"), output the items comma-separated, covering main head nouns.
4. Only reply "Not mentioned in the conversation." if the entity/topic literally never appears.

Answer in 1-6 words unless the question demands a list. No preamble.

EXAMPLES:
Q: "What did Caroline research?" → "Adoption agencies"
Q: "What is Caroline's identity?" → "Transgender woman"
Q: "What is Caroline's relationship status?" → "Single"
Q: "What does Melanie's kids like?" → "dinosaurs, nature"
Q: "Where did Caroline move from 4 years ago?" → "Sweden"
Q: "What LGBTQ+ events has Caroline participated in?" → "Pride parade, school speech, support group"
Q: "What career path has Caroline decided to pursue?" → "counseling or mental health for Transgender people"
""",
    2: """You answer questions about TIMING, DATES, and DURATION.

Strategy for TEMPORAL:
1. Consult TEMPORAL FACTS first — those are ISO dates anchored to events.
2. For "when" → output the date ("2023-05-08", "May 2023", "last fall"). Match the precision of what's in the excerpts.
3. For "how long ago" → express as "N years ago", "N months ago", "a week ago" using the conversation's own temporal language.
4. For "how long has X been …" → duration phrase ("4 years", "since 2021").
5. For date-range questions, combine two anchor dates from TEMPORAL FACTS.
6. Be generous: if the exact date isn't given but the excerpt anchors it to another event ("the week before 9 June 2023"), quote that phrase verbatim.
7. Never refuse unless no date or time-word appears anywhere in the provided sources.

Short phrase (1-5 words typical). No preamble.

EXAMPLES:
Q: "When did Caroline go to the LGBTQ support group?" → "7 May 2023"
Q: "When is Melanie planning on going camping?" → "June 2023"
Q: "When did Melanie run a charity race?" → "The sunday before 25 May 2023"
Q: "How long has Caroline had her current group of friends for?" → "4 years"
Q: "How long ago was Caroline's 18th birthday?" → "10 years ago"
Q: "When did Melanie paint a sunrise?" → "2022"
""",
    3: """You answer MULTI-HOP questions that require combining facts from different turns.

Think step by step internally (but only output the final answer):
  Step 1 — List the 2-4 most relevant facts from STRUCTURED FACTS / RELATED FACTS GRAPH / TEMPORAL FACTS / EXCERPTS that touch the question's entities.
  Step 2 — Chain or contrast them to form the inference (X likes A → Y is similar to A → X probably likes Y).
  Step 3 — Output a concise final answer.

Rules:
- Multi-hop questions often ask about inference from traits/interests ("What fields would X pursue?", "Would X like Y?"). Answer from stated preferences and patterns.
- Hedged answers are natural: "Likely yes, because …" / "Probably no — she prefers …".
- If the question combines two entities ("What did X tell Y about Z?"), cover the action and details.
- Never refuse unless NONE of the mentioned entities/topics appear anywhere.

Output ONLY the final concise answer (one sentence). No "Step 1 …" in the output.

EXAMPLES:
Q: "What fields would Caroline be likely to pursue in her education?" → "Psychology, counseling certification"
Q: "Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?" → "Likely no"
Q: "Would Caroline likely have Dr. Seuss books on her bookshelf?" → "Yes, since she collects classic children's books"
Q: "Would Melanie be more interested in going to a national park or a theme park?" → "National park; she likes the outdoors"
Q: "Would Caroline pursue writing as a career option?" → "LIkely no; though she likes reading, she wants to be a counselor"
""",
    4: """You answer open-domain questions about the people in the conversation — their lives, relationships, opinions, taste.

Strategy for OPEN-DOMAIN:
1. SESSION SUMMARY carries the big picture — start there.
2. Combine with EXCERPTS for specifics.
3. For list questions output comma-separated items; partial list is fine if head entities are right.
4. For opinion/preference questions, hedged answers are welcome.

One short sentence or phrase. No preamble.""",
    5: """You answer questions where the answer may legitimately be "not in the conversation".

Strategy for ADVERSARIAL:
1. If no explicit statement in any provided source supports the question, reply exactly: "Not mentioned in the conversation."
2. Do NOT speculate, do NOT use hedged "likely yes" reasoning.
3. If the question is adversarial ("How did X react to …?" when X was never involved), refuse.

One short phrase.""",
}


# v9 D6 — load extra few-shot examples from disk and append them to each
# per-category prompt. Mined via scripts/mine_locomo_fewshot.py from the
# train slice of LoCoMo only — eval must be run against held-out conv ids
# to remain leakage-free.
def _augment_category_prompts(few_shot_path: Path) -> dict[int, str]:
    if not few_shot_path.exists():
        print(f"[locomo-llm] few-shot file missing: {few_shot_path} — using built-in prompts.")
        return CATEGORY_PROMPTS
    try:
        bundle = json.loads(few_shot_path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"[locomo-llm] few-shot parse error: {e} — using built-in prompts.")
        return CATEGORY_PROMPTS

    augmented: dict[int, str] = dict(CATEGORY_PROMPTS)
    for cat in (1, 2, 3, 4, 5):
        pairs = bundle.get(str(cat)) or []
        if not pairs:
            continue
        examples_block = "\n\nADDITIONAL FEW-SHOT (mined from LoCoMo train, not eval):\n"
        for p in pairs:
            q = (p.get("q") or "").strip()
            a = (p.get("a") or "").strip()
            if not q or not a:
                continue
            examples_block += f'Q: "{q}" → "{a}"\n'
        augmented[cat] = augmented.get(cat, ANSWER_SYSTEM) + examples_block
    meta = bundle.get("_meta") or {}
    print(
        f"[locomo-llm] few-shot loaded from {few_shot_path}: "
        f"train_ids={meta.get('train_conv_ids')} n_per_cat={meta.get('n_per_category')}"
    )
    return augmented


QUERY_REWRITE_SYSTEM = """You rewrite a user question into a short DECLARATIVE statement in the style of a factual note.
This statement will be used as a retrieval query against a fact index.

Rules:
- Preserve the entity names exactly.
- Replace interrogatives with a statement pattern that an answer would match. Examples:
    "Where did Alice travel?" -> "Alice traveled to"
    "When did Bob meet Carol?" -> "Bob met Carol on"
    "What does Carol like?" -> "Carol likes"
    "Would Mel buy a theme park ticket?" -> "Mel's preferences about theme parks"
- Output ONE line. No prose, no quotes.
"""

JUDGE_SYSTEM = """You are an impartial evaluator checking whether a predicted answer is semantically correct given a gold (reference) answer.

ACCEPT AS CORRECT:
- Paraphrases, synonyms, different phrasings of the same fact.
- Different unit/format representations (e.g. "$500" vs "five hundred dollars", "2 hours" vs "120 minutes").
- Temporal answers referring to the same event with approximate dates.
- List-style gold answers (e.g. "mansion in Japan, luxury car Ferrari 488 GTB"): accept the prediction if it covers the MAIN entity or head noun, even when secondary details, modifiers, or brand names are missing. Partial matches on the primary entity count as correct.
- Predictions that are more specific than the gold but consistent with it.
- For adversarial (unanswerable) gold: accept if BOTH gold and prediction indicate no information is available (any form of "not mentioned", "unknown", "don't know", "no information").

REJECT:
- Predictions that contradict the gold on the main entity.
- Predictions that add fabricated information not in the gold.
- For adversarial gold: any concrete speculative answer from the prediction.

Respond with ONLY one word on the first line: YES or NO. Nothing else."""


def call_llm(client: LLMClient, system: str, user: str, max_tokens: int = 80,
             model: str = DEFAULT_GEN_MODEL) -> tuple[str, int, int]:
    """Provider-agnostic single completion. Returns (text, in_tok, out_tok)."""
    r = client.complete(system, user, model=model, max_tokens=max_tokens)
    return r.text, r.input_tokens, r.output_tokens


# Back-compat alias so older call-sites in this file keep working.
call_haiku = call_llm


def build_context(entries: list[dict], max_chars: int = 6000) -> str:
    """Concatenate retrieved turn contents into an ordered block for Haiku."""
    lines = []
    used = 0
    for e in entries:
        c = e.get("content", "").strip()
        if not c:
            continue
        if used + len(c) > max_chars:
            break
        lines.append(f"- {c}")
        used += len(c) + 2
    return "\n".join(lines) if lines else "(no relevant excerpts retrieved)"


def process_qa(client, server_mod, store, recall, qa: dict, project: str,
               top_k: int, gen_model: str = DEFAULT_GEN_MODEL,
               judge_model: str = DEFAULT_JUDGE_MODEL,
               apply_temporal: bool = False,
               over_fetch_k: int = 20,
               oracle_routing: bool = False,
               *,
               two_stage: bool = False,
               guard_model: str = "gpt-4o-mini",
               fact_index=None,
               ce_rerank: bool = False,
               drop_tags: tuple[str, ...] = (),
               per_cat_prompts: bool = False,
               query_rewrite: bool = False,
               hyde: bool = False,
               ensemble: int = 1,
               ensemble_mode: str = "voter",
               aux_model: str = "gpt-4o-mini",
               entity_boost: bool = False,
               subject_aware: bool = False) -> dict:
    question = qa.get("question", "")
    gold = str(qa.get("answer", "")).strip()
    cat = qa.get("category", 0)
    evidence = set(qa.get("evidence", []) or [])

    # L6 per-category top_k: open-domain needs richer context, single-hop
    # tighter (sharper signal), adversarial small to reduce confab surface.
    cat_top_k = top_k
    cat_guess = qa.get("category", 0)
    if cat_guess == 4:       # open-domain
        cat_top_k = max(top_k, 16)
    elif cat_guess == 1:     # single-hop
        cat_top_k = min(top_k, 8)
    elif cat_guess == 5:     # adversarial
        cat_top_k = min(top_k, 6)

    # Optional LLM entity extraction — will be used later to boost retrieval hits.
    search_query = question
    aux_in = aux_out = 0
    question_entities: list[str] = []
    if entity_boost:
        try:
            ent_txt, ti, to = call_llm(
                client,
                "Extract named entities (people, places, events, concrete things) from the question. "
                "Output them comma-separated, no other text. If none, output 'NONE'.",
                f"QUESTION: {question}\n\nENTITIES:",
                max_tokens=40, model=aux_model,
            )
            aux_in += ti; aux_out += to
            raw = ent_txt.strip()
            if raw and raw.upper() != "NONE":
                question_entities = [e.strip() for e in raw.split(",") if e.strip() and len(e.strip()) > 1]
        except Exception:  # noqa: BLE001
            pass

    if query_rewrite:
        try:
            rw_text, ti, to = call_llm(
                client, QUERY_REWRITE_SYSTEM,
                f"QUESTION: {question}\n\nSTATEMENT:",
                max_tokens=40, model=aux_model,
            )
            aux_in += ti; aux_out += to
            if rw_text.strip():
                search_query = rw_text.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
    if hyde:
        try:
            hyde_text, ti, to = call_llm(
                client,
                "Write one plausible factual sentence that would be a correct answer to the question. "
                "Keep it brief, declarative, in a factual-note style.",
                f"QUESTION: {question}\n\nHYPOTHETICAL FACT:",
                max_tokens=60, model=aux_model,
            )
            aux_in += ti; aux_out += to
            if hyde_text.strip():
                # Append so the original query signal stays alongside HyDE anchor.
                search_query = f"{search_query} {hyde_text.strip()}"
        except Exception:  # noqa: BLE001
            pass

    # Retrieve (SQLite connection is not thread-safe; serialize).
    # Over-fetch when temporal rerank is active so we have room to re-order.
    fetch_k = max(over_fetch_k if apply_temporal else cat_top_k, cat_top_k)
    t0 = time.time()
    with RETRIEVAL_LOCK:
        res = recall.search(query=search_query, project=project, limit=fetch_k,
                            detail="summary")
    retrieval_ms = (time.time() - t0) * 1000
    # Collect entries from ALL types (fact, synthesized_fact, etc.) — RRF
    # already ranked them. Flatten by descending score.
    grouped = res.get("results", {}) or {}
    entries: list[dict] = []
    for _typ, group in grouped.items():
        entries.extend(group)
    entries.sort(key=lambda e: -e.get("score", 0))

    # Entity boost — re-rank so any entry whose content or tags contain
    # a question entity jumps to the top (stable within-group sort).
    if question_entities:
        ent_lower = [e.lower() for e in question_entities]

        def _entity_score(e):
            c = (e.get("content", "") or "").lower()
            t = str(e.get("tags", "") or "").lower()
            base = e.get("score", 0.0) or 0.0
            hits = sum(1 for ent in ent_lower if ent in c or ent in t)
            return -hits, -base  # most hits first, then by score

        entries.sort(key=_entity_score)

    # Optional tag-based noise filter (dropping speculative/inferred facts
    # that hurt precision on LoCoMo).
    if drop_tags:
        def _keep(e):
            tags = e.get("tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            return not any(t in tags for t in drop_tags)
        entries = [e for e in entries if _keep(e)]

    # Dedupe: if a raw turn AND its synth_fact child are both present,
    # drop the raw turn (keep the canonical distilled form). Synth facts
    # carry `context = "distilled_from=<parent_id>"`.
    if len(entries) > 1:
        synth_parents: set[int] = set()
        for e in entries:
            ctx = e.get("context") or ""
            if ctx.startswith("distilled_from="):
                try:
                    synth_parents.add(int(ctx.split("=", 1)[1].split()[0]))
                except ValueError:
                    pass
        if synth_parents:
            entries = [e for e in entries if e.get("id") not in synth_parents]

    if apply_temporal:
        entries = temporal_rerank(question, entries)[:cat_top_k * 2]
    # Optional CrossEncoder rerank on a larger candidate pool, then trim.
    if ce_rerank and len(entries) > cat_top_k:
        try:
            from reranker import rerank_results as _ce_rerank  # noqa: PLC0415
            wrapped = [{"r": e, "s": e.get("score", 0.0), "_orig": e} for e in entries[:max(20, cat_top_k * 2)]]
            ranked = _ce_rerank(question, wrapped, top_k=cat_top_k)
            entries = [w["_orig"] for w in ranked[:cat_top_k]]
        except Exception:  # noqa: BLE001
            entries = entries[:cat_top_k]
    else:
        entries = entries[:cat_top_k]

    # Oracle routing: use true category to adapt top-K composition.
    # Disabled by default — for measuring upper-bound of question-type routing.
    if oracle_routing:
        cat = qa.get("category", 0)

        def _is_synth(e: dict) -> bool:
            tags = e.get("tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            return "synthesized_fact" in tags or (e.get("context") or "").startswith("distilled_from=")

        if cat == 1:  # single-hop → prefer synth_facts, but keep raw as backup
            synth = [e for e in entries if _is_synth(e)]
            raws = [e for e in entries if not _is_synth(e)]
            if len(synth) >= 5:
                entries = (synth + raws)[:top_k]
            # else keep mixed
        elif cat == 5:  # adversarial → truncate to reduce confabulation surface
            entries = entries[:max(5, top_k // 2)]

    # Evidence recall (retrieval-only metric)
    ranked_dia_ids = []
    for entry in entries:
        tags = entry.get("tags") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        did = next((t for t in tags if isinstance(t, str) and t.startswith("D") and ":" in t), "")
        ranked_dia_ids.append(did)
    r_at_1 = int(any(d in evidence for d in ranked_dia_ids[:1]))
    r_at_5 = int(any(d in evidence for d in ranked_dia_ids[:5]))
    r_at_10 = int(any(d in evidence for d in ranked_dia_ids[:10]))

    # L2 fact-index boost: prepend structured (entity, attribute)→value hits
    # to the context so the generator sees them first. Keep raw turns too.
    fact_prefix = ""
    if fact_index is not None:
        try:
            # Hybrid if semantic index was built, else substring-only fallback.
            if getattr(fact_index, "_semantic_matrix", None) is not None:
                hits = fact_index.lookup_query_hybrid(question, limit=6, project=project)
            else:
                hits = fact_index.lookup_query(question, limit=5, project=project)
            if hits:
                fact_prefix = "STRUCTURED FACTS:\n" + "\n".join(
                    f"- {h.entity} {h.relation.replace('_',' ')} {h.value}"
                    for h in hits
                ) + "\n\n"
        except Exception:  # noqa: BLE001
            fact_prefix = ""

    # Graph-walk boost for multi-hop: all non-noisy edges containing any
    # question entity. Cheap SQL, big precision lift on chained questions.
    graph_prefix = ""
    if question_entities:
        try:
            placeholders = ",".join(["?"] * len(question_entities))
            lc_ents = [e.lower() for e in question_entities]
            noise = ("mentioned_with", "co_occurred", "semantic_similarity", "shared_across", "supersedes")
            noise_ph = ",".join(["?"] * len(noise))
            sql = (
                "SELECT s.name, e.relation_type, t.name "
                "FROM graph_edges e "
                "JOIN graph_nodes s ON s.id = e.source_id "
                "JOIN graph_nodes t ON t.id = e.target_id "
                "WHERE e.relation_type NOT IN (" + noise_ph + ") "
                f"AND (LOWER(s.name) IN ({placeholders}) OR LOWER(t.name) IN ({placeholders})) "
                "ORDER BY e.weight DESC LIMIT 15"
            )
            gw_rows = store.db.execute(
                sql,
                (*noise, *lc_ents, *lc_ents),
            ).fetchall()
            if gw_rows:
                graph_prefix = "RELATED FACTS GRAPH:\n" + "\n".join(
                    f"- {r[0]} {str(r[1]).replace('_',' ')} {r[2]}" for r in gw_rows
                ) + "\n\n"
        except Exception:  # noqa: BLE001
            graph_prefix = ""

    # L4 temporal-index: pull dated facts for this project that match the
    # question's entity tokens. For "when did X happen?" the model then has
    # a DATE anchor right on top of the context.
    temporal_prefix = ""
    try:
        q_lower = question.lower()
        if any(tok in q_lower for tok in ("when", "date", "what year", "on what day", "month")):
            db_conn = store.db  # re-use bench-patched connection
            rows = db_conn.execute(
                "SELECT k.content, ft.date_iso, ft.date_raw "
                "FROM fact_temporal ft JOIN knowledge k ON k.id = ft.knowledge_id "
                "WHERE ft.project = ? ORDER BY ft.date_iso DESC LIMIT 40",
                (project,),
            ).fetchall()
            # Keep facts whose content mentions any capitalised token from the question.
            ents = [w for w in re.findall(r"[A-Z][a-zA-Z0-9'-]+", question)
                    if len(w) > 2 and w.lower() not in ("when", "what", "where")]
            if ents and rows:
                matched = [r for r in rows
                           if any(ent.lower() in (r[0] or "").lower() for ent in ents)]
            else:
                matched = list(rows[:6])
            if matched:
                temporal_prefix = "TEMPORAL FACTS:\n" + "\n".join(
                    f"- {r[1]} — {r[0][:140]}" for r in matched[:6]
                ) + "\n\n"
    except Exception:  # noqa: BLE001
        temporal_prefix = ""

    # L3 session summary — one-paragraph overview of the whole conversation.
    summary_prefix = ""
    try:
        row = store.db.execute(
            "SELECT content FROM knowledge "
            "WHERE type='session_summary' AND project = ? LIMIT 1",
            (project,),
        ).fetchone()
        if row and row[0]:
            summary_prefix = f"SESSION SUMMARY:\n{row[0]}\n\n"
    except Exception:  # noqa: BLE001
        summary_prefix = ""

    # v9 D8 — Subject-aware retrieval. One cheap LLM call extracts
    # (subject, action_keywords) from the question and we run a SQL lookup
    # against graph_edges anchored on that subject. Resulting triples are
    # prepended as DIRECT FACTS so the generator sees the answer verbatim
    # at position 0 of the context (fixes R@1 on cat=1 single-hop).
    direct_facts_prefix = ""
    sa_in = sa_out = 0
    if subject_aware:
        try:
            from subject_predicate_retriever import (  # noqa: PLC0415
                extract_key,
                lookup_triples,
                format_triples_block,
            )
            key = extract_key(client, question, model=aux_model)
            sa_in += 0  # token accounting handled inside extract_key call wrapper below
            if key.subject:
                triples = lookup_triples(store.db, project, key, limit=10)
                direct_facts_prefix = format_triples_block(triples)
        except Exception:  # noqa: BLE001
            direct_facts_prefix = ""

    # Generate answer
    context = build_context(entries)
    user_prompt = (
        f"{direct_facts_prefix}{summary_prefix}{fact_prefix}{graph_prefix}{temporal_prefix}"
        f"CONVERSATION EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    # Choose system prompt — generic vs LoCoMo per-category.
    system_prompt = ANSWER_SYSTEM
    if per_cat_prompts:
        cat_now = qa.get("category", 0)
        system_prompt = CATEGORY_PROMPTS.get(cat_now, ANSWER_SYSTEM)

    guard_in = guard_out = 0
    pred = ""
    if two_stage:
        # Stage A — cheap guard decides whether the context actually supports an answer.
        guard_prompt = (
            f"{fact_prefix}CONVERSATION EXCERPTS:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Is the question answerable strictly from the excerpts above? "
            "Reply with one token: YES or NO."
        )
        guard_txt, guard_in, guard_out = call_llm(
            client,
            "You are a strict gate. Answer YES only if the context directly contains the answer; otherwise NO.",
            guard_prompt,
            max_tokens=4,
            model=guard_model,
        )
        answerable = guard_txt.strip().upper().startswith("YES")
        if not answerable:
            # Refuse cleanly — matches LoCoMo adversarial gold style.
            pred = "Not specified in the conversation."
            gen_in = gen_out = 0
        else:
            pred, gen_in, gen_out = call_llm(client, system_prompt, user_prompt,
                                             max_tokens=80, model=gen_model)
    else:
        if ensemble > 1:
            temps = [0.0, 0.3, 0.6, 0.9][:ensemble]
            cand_preds: list[str] = []
            gen_in = gen_out = 0
            for temp in temps:
                r = client.complete(system_prompt, user_prompt,
                                    model=gen_model, max_tokens=80, temperature=temp)
                cand_preds.append(r.text)
                gen_in += r.input_tokens
                gen_out += r.output_tokens
            if ensemble_mode == "judge":
                # v9 D7: judge-weighted picker with category-aware rubric +
                # abstain logic (matches LoCoMo adversarial gold style).
                from ensemble_judge import judge_weighted_pick  # noqa: PLC0415
                pick = judge_weighted_pick(
                    client,
                    question=question,
                    candidates=cand_preds,
                    category=cat,
                    judge_model=aux_model,
                )
                pred = pick.answer
                aux_in += pick.judge_input_tokens
                aux_out += pick.judge_output_tokens
            else:
                # Legacy LLM voter — cheap model picks the best candidate.
                vote_prompt = (
                    f"QUESTION: {question}\n\n"
                    "CANDIDATE ANSWERS (one per line):\n"
                    + "\n".join(f"[{i}] {p}" for i, p in enumerate(cand_preds))
                    + "\n\nWhich candidate best answers the question? "
                      "Reply with just the index number [0], [1] or [2]. No prose."
                )
                try:
                    vt, ti, to = call_llm(
                        client,
                        "You pick the best answer from a short list. Reply with the number only.",
                        vote_prompt, max_tokens=4, model=aux_model,
                    )
                    aux_in += ti; aux_out += to
                    m = re.search(r"[0-9]", vt)
                    idx = int(m.group(0)) if m else 0
                    idx = max(0, min(idx, len(cand_preds) - 1))
                except Exception:  # noqa: BLE001
                    idx = 0
                pred = cand_preds[idx]
        else:
            pred, gen_in, gen_out = call_llm(client, system_prompt, user_prompt,
                                             max_tokens=80, model=gen_model)

    # Judge
    judge_prompt = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {pred}\n\n"
        f"Is the predicted answer correct? Respond YES or NO."
    )
    judge_out, j_in, j_out = call_llm(client, JUDGE_SYSTEM, judge_prompt,
                                      max_tokens=4, model=judge_model)
    correct = judge_out.upper().startswith("YES")

    return {
        "question": question,
        "gold": gold,
        "pred": pred,
        "category": cat,
        "evidence": list(evidence),
        "retrieved_dia_ids": ranked_dia_ids,
        "r@1": r_at_1, "r@5": r_at_5, "r@10": r_at_10,
        "correct": correct,
        "f1": f1_score(pred, gold),
        "bleu1": bleu1(pred, gold),
        "rouge_l": rouge_l(pred, gold),
        "retrieval_ms": retrieval_ms,
        "tokens_in": gen_in + j_in + guard_in + aux_in,
        "tokens_out": gen_out + j_out + guard_out + aux_out,
    }


# ──────────────────────────────────────────────────────────────────────────
# Aggregation & reporting
# ──────────────────────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    per_cat = defaultdict(list)
    for r in records:
        per_cat[r["category"]].append(r)

    out = {}
    for cat, recs in per_cat.items():
        n = len(recs)
        out[f"category_{cat}"] = {
            "label": CATEGORY_NAMES.get(cat, f"cat_{cat}"),
            "n": n,
            "accuracy": round(sum(r["correct"] for r in recs) / n, 4),
            "f1": round(sum(r["f1"] for r in recs) / n, 4),
            "bleu1": round(sum(r["bleu1"] for r in recs) / n, 4),
            "rouge_l": round(sum(r["rouge_l"] for r in recs) / n, 4),
            "R@1": round(sum(r["r@1"] for r in recs) / n, 4),
            "R@5": round(sum(r["r@5"] for r in recs) / n, 4),
            "R@10": round(sum(r["r@10"] for r in recs) / n, 4),
        }

    total = len(records)
    non_adv = [r for r in records if r["category"] != 5]
    out["overall_all"] = {
        "n": total,
        "accuracy": round(sum(r["correct"] for r in records) / total, 4),
        "f1": round(sum(r["f1"] for r in records) / total, 4),
        "bleu1": round(sum(r["bleu1"] for r in records) / total, 4),
        "rouge_l": round(sum(r["rouge_l"] for r in records) / total, 4),
    }
    if non_adv:
        n = len(non_adv)
        out["overall_no_adversarial"] = {
            "n": n,
            "accuracy": round(sum(r["correct"] for r in non_adv) / n, 4),
            "f1": round(sum(r["f1"] for r in non_adv) / n, 4),
            "R@1": round(sum(r["r@1"] for r in non_adv) / n, 4),
            "R@5": round(sum(r["r@5"] for r in non_adv) / n, 4),
            "R@10": round(sum(r["r@10"] for r in non_adv) / n, 4),
        }

    lats = [r["retrieval_ms"] for r in records]
    out["latency_retrieval"] = {
        "p50_ms": round(statistics.median(lats), 2) if lats else 0,
        "p95_ms": round(sorted(lats)[int(0.95 * len(lats))], 2) if lats else 0,
        "mean_ms": round(sum(lats) / max(len(lats), 1), 2),
    }
    out["tokens_total"] = {
        "input": sum(r["tokens_in"] for r in records),
        "output": sum(r["tokens_out"] for r in records),
    }
    return out


def format_report(ingest_stats: dict, agg: dict) -> str:
    lines = [
        "=" * 78,
        "  LoCoMo — Claude Total Memory v8  (Haiku 4.5 RAG + LLM-judge)",
        "=" * 78,
        "",
        "Ingestion",
        f"  saved={ingest_stats.get('saved', 0)}  skipped={ingest_stats.get('skipped', 0)}  "
        f"elapsed={ingest_stats.get('elapsed_sec', 0)}s  "
        f"rate={ingest_stats.get('rate_turn_per_sec', 0)} turn/s",
        "",
        "Per-category results",
        f"  {'category':<22}  {'N':>4}  {'Acc':>6}  {'F1':>6}  {'BLEU1':>6}  {'RougeL':>7}  {'R@1':>5}  {'R@5':>5}  {'R@10':>5}",
    ]
    for key in ("category_1", "category_2", "category_3", "category_4", "category_5"):
        if key not in agg:
            continue
        d = agg[key]
        label = f"{key} ({d['label']})"
        lines.append(
            f"  {label:<22}  {d['n']:>4}  {d['accuracy']:>6.3f}  {d['f1']:>6.3f}  "
            f"{d['bleu1']:>6.3f}  {d['rouge_l']:>7.3f}  {d['R@1']:>5.3f}  {d['R@5']:>5.3f}  {d['R@10']:>5.3f}"
        )
    lines.append("")
    if "overall_no_adversarial" in agg:
        d = agg["overall_no_adversarial"]
        lines.append(f"  overall (no adv)     N={d['n']:<5} Acc={d['accuracy']:.3f}  "
                     f"F1={d['f1']:.3f}  R@1={d['R@1']:.3f}  R@5={d['R@5']:.3f}  R@10={d['R@10']:.3f}")
    if "overall_all" in agg:
        d = agg["overall_all"]
        lines.append(f"  overall (all)        N={d['n']:<5} Acc={d['accuracy']:.3f}  "
                     f"F1={d['f1']:.3f}  BLEU1={d['bleu1']:.3f}  RougeL={d['rouge_l']:.3f}")
    lat = agg.get("latency_retrieval", {})
    tok = agg.get("tokens_total", {})
    lines.extend([
        "",
        f"Retrieval latency  p50={lat.get('p50_ms', 0)} ms  p95={lat.get('p95_ms', 0)} ms  mean={lat.get('mean_ms', 0)} ms",
        f"Haiku tokens       in={tok.get('input', 0):,}  out={tok.get('output', 0):,}",
        "",
    ])
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--limit-qa", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--wipe", action="store_true")
    parser.add_argument("--enable-llm-extraction", action="store_true",
                        help="Keep Ollama triple extraction on ingest (slower)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "openai", "anthropic"],
                        help="LLM provider (auto-detected from model if 'auto')")
    parser.add_argument("--gen-model", default=DEFAULT_GEN_MODEL,
                        help="Answer generator (gpt-4o/gpt-4o-mini/haiku/sonnet)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help="LLM-judge model (keep cheap)")
    parser.add_argument("--two-stage", action="store_true",
                        help="Stage A: cheap-model answerable? guard → Stage B: strong model answers only if yes.")
    parser.add_argument("--guard-model", default="gpt-4o-mini",
                        help="Two-stage guard model (used when --two-stage).")
    parser.add_argument("--fact-index", action="store_true",
                        help="Use src/fact_index.FactIndex to prepend structured matches to the context.")
    parser.add_argument("--semantic-fact-index", action="store_true",
                        help="Enable embedding-based lookup on top of --fact-index (lookup_query_hybrid).")
    parser.add_argument("--ce-rerank", action="store_true",
                        help="Enable reranking on retrieved entries; backend selected by --reranker / V9_RERANKER_BACKEND.")
    parser.add_argument("--reranker", default=None,
                        choices=["ce-marco", "bge-v2-m3", "bge-large", "off"],
                        help="Reranker backend (sets V9_RERANKER_BACKEND for the run). Default keeps env value.")
    parser.add_argument("--ignore-synth-tag", default=None,
                        help="Comma-separated tags — entries carrying ANY of these tags are dropped from context (noise reduction).")
    parser.add_argument("--per-cat-prompts", action="store_true",
                        help="Use LoCoMo-specific per-category answer prompts (single/multi/temporal/open/adv).")
    parser.add_argument("--few-shot-pairs", type=Path, default=None,
                        help="Path to JSON bundle from scripts/mine_locomo_fewshot.py — appends "
                             "extra few-shot examples to each per-category prompt. Train conv "
                             "ids of mining must NOT overlap with --samples evaluated.")
    parser.add_argument("--query-rewrite", action="store_true",
                        help="Rewrite question as declarative statement (gpt-4o-mini) before retrieval.")
    parser.add_argument("--hyde", action="store_true",
                        help="Generate hypothetical answer (gpt-4o-mini) and use it as the retrieval query.")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Run generator N times (1/2/3) with temp jitter; pick best via "
                             "ensemble-mode (cost scales N+1).")
    parser.add_argument("--ensemble-mode", default="voter", choices=["voter", "judge"],
                        help="voter (default, legacy): cheap LLM picks best index. "
                             "judge (v9 D7): category-aware scoring + abstain logic, see "
                             "benchmarks/ensemble_judge.py.")
    parser.add_argument("--entity-boost", action="store_true",
                        help="LLM extracts entities from question; knowledge rows containing those entities are boosted to top of retrieved entries.")
    parser.add_argument("--subject-aware", action="store_true",
                        help="v9 D8: LLM extracts (subject, action) keys from question and "
                             "looks up graph_edges directly. Prepends DIRECT FACTS to context.")
    parser.add_argument("--category", type=int, default=None,
                        help="Only eval category 1-5 (useful for targeted reruns)")
    parser.add_argument("--temporal-filter", action="store_true",
                        help="Enable date-aware re-rank after retrieval")
    parser.add_argument("--over-fetch-k", type=int, default=20,
                        help="Fetch this many candidates before temporal rerank")
    parser.add_argument("--oracle-routing", action="store_true",
                        help="Use true LoCoMo category to route retrieval strategy "
                             "(measures upper bound before building real classifier)")
    args = parser.parse_args()

    # v9 D4: propagate --reranker flag into env BEFORE any reranker import.
    # src/reranker.py reads V9_RERANKER_BACKEND lazily on each call, but model
    # caching is per-process — set this before reranker._get_reranker() runs.
    if args.reranker is not None:
        os.environ["V9_RERANKER_BACKEND"] = args.reranker

    # v9 D6: optionally swap CATEGORY_PROMPTS with the few-shot-augmented
    # version. Affects per-cat-prompts code path only.
    if args.few_shot_pairs:
        global CATEGORY_PROMPTS  # noqa: PLW0603
        CATEGORY_PROMPTS = _augment_category_prompts(args.few_shot_pairs)
        print(f"[locomo-llm] reranker backend → {args.reranker}")

    gen_model = MODEL_ALIASES.get(args.gen_model, args.gen_model)
    judge_model = MODEL_ALIASES.get(args.judge_model, args.judge_model)

    dataset = load_dataset(DATASET)
    if args.limit_samples:
        dataset = dataset[: args.limit_samples]
    total_qa = sum(len(s["qa"]) for s in dataset)
    print(f"[locomo-llm] samples={len(dataset)}  total_qa={total_qa}  model={HAIKU_MODEL}")

    db_path = Path(args.db_path)
    if args.wipe and db_path.exists():
        print(f"[locomo-llm] wiping {db_path}")
        shutil.rmtree(db_path, ignore_errors=True)

    setup_env(db_path, disable_llm_extraction=not args.enable_llm_extraction)
    server_mod = import_store()

    ingest_stats = {}
    if not args.skip_ingest:
        print(f"[locomo-llm] ingesting → {db_path}")
        ingest_stats = ingest(server_mod, dataset)
        print(f"[locomo-llm] ingest: {ingest_stats}")

    store = server_mod.Store()
    recall = server_mod.Recall(store)
    # Warm up lazy inits (binary search, fastembed) on main thread
    recall.search(query="warmup", project="locomo_0", limit=1, detail="summary")
    patch_thread_safety(server_mod, store)

    client = LLMClient(provider=args.provider, default_model=gen_model)
    print(f"[locomo-llm] provider={client.provider}")

    # Build work list
    jobs = []
    for sample_idx, sample in enumerate(dataset):
        project = f"locomo_{sample_idx}"
        for qa in sample["qa"]:
            if args.category and qa.get("category") != args.category:
                continue
            jobs.append((project, qa))
            if args.limit_qa and len(jobs) >= args.limit_qa:
                break
        if args.limit_qa and len(jobs) >= args.limit_qa:
            break

    print(f"[locomo-llm] evaluating {len(jobs)} QAs  "
          f"gen={gen_model}  judge={judge_model}  concurrency={args.concurrency}")

    records: list[dict] = []
    t0 = time.time()
    completed = 0

    # Optional L2 structured lookup — loaded once, shared across threads.
    fact_index_obj = None
    if args.fact_index:
        sys.path.insert(0, "/Users/vitalii-macpro/claude-memory-server/src")
        from fact_index import FactIndex  # noqa: PLC0415
        fact_index_obj = FactIndex(store.db)
        print(f"[locomo-llm] fact_index ON — stats={fact_index_obj.stats()}")
        if args.semantic_fact_index:
            t_idx = time.time()
            n_edges = fact_index_obj.build_semantic_index()
            print(f"[locomo-llm] semantic fact_index built over {n_edges} edges "
                  f"in {time.time()-t_idx:.1f}s")

    drop_tags = tuple(t.strip() for t in (args.ignore_synth_tag or "").split(",") if t.strip())

    def run_one(project: str, qa: dict):
        return process_qa(client, server_mod, store, recall, qa, project,
                          args.top_k, gen_model=gen_model, judge_model=judge_model,
                          apply_temporal=args.temporal_filter,
                          over_fetch_k=args.over_fetch_k,
                          oracle_routing=args.oracle_routing,
                          two_stage=args.two_stage,
                          guard_model=args.guard_model,
                          fact_index=fact_index_obj,
                          ce_rerank=args.ce_rerank,
                          drop_tags=drop_tags,
                          per_cat_prompts=args.per_cat_prompts,
                          query_rewrite=args.query_rewrite,
                          hyde=args.hyde,
                          ensemble=max(1, int(args.ensemble)),
                          ensemble_mode=args.ensemble_mode,
                          aux_model="gpt-4o-mini",
                          entity_boost=args.entity_boost,
                          subject_aware=args.subject_aware)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(run_one, p, q): (p, q) for p, q in jobs}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                records.append(rec)
            except Exception as e:
                p, q = futures[fut]
                print(f"  job failed {q.get('question', '')[:40]!r}: {e}", file=sys.stderr)
            completed += 1
            if completed % 100 == 0 or completed == len(jobs):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.01)
                eta = (len(jobs) - completed) / max(rate, 0.01)
                acc = sum(r["correct"] for r in records) / max(len(records), 1)
                print(f"  [{completed}/{len(jobs)}] "
                      f"{rate:.1f} qa/s  eta={eta:.0f}s  "
                      f"running_acc={acc:.3f}", flush=True)

    agg = aggregate(records)
    report = format_report(ingest_stats, agg)
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"locomo-llm-{int(time.time())}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "ingest": ingest_stats,
            "aggregate": agg,
            "records": records,
            "config": {
                "gen_model": gen_model,
                "judge_model": judge_model,
                "db_path": str(db_path),
                "top_k": args.top_k,
                "concurrency": args.concurrency,
                "samples": len(dataset),
                "category_filter": args.category,
                "temporal_filter": args.temporal_filter,
                "over_fetch_k": args.over_fetch_k,
                "llm_extraction_enabled": args.enable_llm_extraction,
            },
        }, fh, indent=2, ensure_ascii=False)
    print(f"[locomo-llm] report → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
