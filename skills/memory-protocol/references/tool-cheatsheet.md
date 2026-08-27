# MCP Tool Cheatsheet — total-agent-memory v10.5

Every MCP tool exposed by `total-agent-memory`. Grouped by purpose,
with: argument shape, return shape (truncated), when to use, and at
least one short example.

If you only ever read three sections, read **Knowledge CRUD**,
**Recall & search**, and **Sessions**. The rest are specialised.

---

## Knowledge CRUD

### `memory_save(type, content, project?, tags?, context?, importance?, coref?, filter?)`

Save a fact / decision / solution / lesson / convention.

```
type         : 'fact' | 'decision' | 'solution' | 'lesson' | 'convention'
content      : structured digest (see SKILL.md template, ≤ 30 lines)
project      : 'general' by default — almost always override it
tags         : ['reusable', '<tech>', ...] for cross-project recipes
context      : for type=decision, the WHY paragraph
importance   : 'critical' | 'high' | 'medium' (default) | 'low'
coref        : true → opt in to pre-write pronoun resolution
filter       : 'pytest' | 'cargo' | 'git_status' | 'docker_ps' | 'generic_logs'
               for noisy CLI output (auto-trims while preserving URLs/paths/code)
```

**Returns:** `{saved: true, id: <int>, deduplicated, quality_score}`
or `{saved: false, rejected_by_quality_gate: true, score, reason}`.

**Trigger:** every significant action — decision, fix, recipe, config,
new convention. Never batch.

```
memory_save(
  type='solution',
  content='ЧТО: ...\nПРОЕКТ: vito\nФАЙЛЫ: src/x.go:42\nСТЕК: Go 1.25\nПОДХОД: ...\nНЮАНСЫ: ...',
  project='vito',
  tags=['reusable', 'go', 'concurrency'],
  importance='high',
)
```

### `memory_recall(query, project?, type?, branch?, limit?, mode?, detail?, decisions_only?, expand_context?, fusion?)`

The main retrieval tool. 6-stage pipeline (FTS5+BM25 → semantic →
fuzzy → graph → optional CrossEncoder → optional MMR). Measured
retrieval quality is published in the project README; the numbers are
reproducible with `benchmarks/locomo_bench.py` and
`benchmarks/beam_bench.py`.

```
query           : what to search
project         : filter; pass current project for relevance
type            : 'decision'|'fact'|'solution'|'lesson'|'convention'|'all'
mode            : 'search' (default) | 'index' (compact) | 'timeline'
detail          : 'full' (default) | 'summary' | 'compact' | 'auto'
decisions_only  : true → only structured decisions
expand_context  : true → +1-hop graph neighbours
diverse         : true → MMR diversity for broad queries
rerank          : true → CrossEncoder (slower, +precision)
```

**Trigger:** before any non-trivial task; whenever the user references
prior work or asks "what did we decide".

**Tip:** `mode='index'` returns ~50 tokens/result with id+title+score.
Use it as the *index* of a 3-layer flow: index → pick ids → `memory_get(ids=...)`.

### `memory_get(ids, detail?)`

Batched fetch by ID. Companion to `memory_recall(mode='index')`.

```
ids    : list of ints, max 50 per call (extras dropped silently)
detail : 'full' (default) | 'summary'
```

### `memory_update(find, new_content, reason?)`

Find-and-replace by query. Marks old row `superseded`,
`superseded_by=<new_id>`, returns the new id.

```
memory_update(find='Postgres connection pool size', new_content='...', reason='increased to 20')
```

### `memory_delete(id)`

Soft delete. Removes from search and ChromaDB; row stays for audit.
Re-issuing with the same id is a no-op.

### `memory_history(id)`

Walk the supersede chain (newest → oldest). Use when the user asks
"how did this evolve" or before relying on a recalled record.

### `memory_search_by_tag(tag, project?, limit?)`

Pure tag filter. Cheap; no embedding cost. Useful for `tag='reusable'`
audits.

### `memory_export(project?, ktype?, branch?, limit?)`

Dump active records to JSON in `<MEMORY_DIR>/backups/`. Returns the
file path. Hand-off / migration.

### `memory_forget(id?, dry_run?, archive_after_days?, purge_after_days?)`

Configurable retention. With `dry_run=true` shows what would be
archived/purged without doing it. **Always dry-run first** if you are
not certain.

### `memory_stats()`

Health snapshot: rows by type/project, retention zones, storage size,
config flags. Cheap. Good first call after a restart.

### `memory_timeline(limit?)`

Chronological recap of recent saves grouped by session. For "what did
we ship this week".

---

## Sessions

### `session_init(project)`

**Call this first in any new session, before `memory_recall`.**
Returns the previous session's `summary + next_steps + pitfalls` and
marks them consumed so they don't repeat next turn.

```
{
  summary: "Last session we shipped v10.1. Worker tests pass.",
  next_steps: ["benchmark on WSL2", "update install.sh"],
  pitfalls: ["soft-drop semantic — recall must filter quality_dropped"],
  ...
}
```

If `{message: "no pending summary"}` is returned, the previous session
either didn't end cleanly or hasn't run yet — proceed normally.

### `session_end(session_id, summary, next_steps, pitfalls?, highlights?, open_questions?, auto_compress?)`

End-of-session capture. The **next** `session_init` will read it.

```
session_id   : current session id (from your tool call context)
summary      : 2–4 sentences — what changed, what works, what's open
next_steps   : ["concrete action 1", "concrete action 2"]
pitfalls     : ["mistake to avoid next time"]
auto_compress: true → LLM generates summary/next_steps from session artifacts
```

Triggered by user saying "сохранись" / "save" / "закругляемся" / on
session-stop hook.

### `memory_extract_session(action='list'|'load'|'process', session_id?, n?)`

Maintain the extract queue (transcripts queued for offline LLM
processing). `list` shows pending, `load` reads one, `process` runs
the extractor.

### `memory_consolidate(project?, threshold?, similarity?)`

Run reflection consolidation: dedupe near-identical, decay stale,
detect contradictions. Heavy. Schedule, don't call inline.

### `memory_reflect_now()`

Manual trigger of the reflection cycle (otherwise runs every 6h via
LaunchAgent). Returns a report.

### `memory_self_assess()`

Returns a level/confidence summary across the corpus and known blind
spots. Cheap.

---

## Knowledge graph (entities & relations)

### `memory_concepts(query?, type?, limit?, include_memories?)`

List or search concept nodes. Concepts are extracted from saves by
the deep enricher.

### `memory_graph(node, depth?, types?)`

Neighborhood query around a node. `node` accepts a name or an id.
`depth=2` is the safe default; `depth=3` can be huge.

### `memory_graph_index(limit?)`

Re-index claude.md / skills / rules into the graph. Idempotent.

### `memory_graph_stats()`

Node/edge counts by type. Fast sanity check.

### `memory_relate(from_id, to_id, relation)`

Manual edge creation. Most edges come from extraction; use this only
for explicit cross-references the extractor missed.

### `memory_associate(query, project?, mode?, max_results?, min_coverage?)`

Spreading-activation recall around a query — different shape than
`memory_recall`, surfaces *associated* concepts/memories rather than
text matches. Good for "what else relates to X".

---

## Knowledge graph — temporal facts

### `kg_add_fact(subject, predicate, object, project?, valid_from?, valid_to?)`

Persist a typed (subject, predicate, object) triple. Old facts about
the same (subject, predicate) auto-invalidate via `valid_to`.

```
kg_add_fact(subject='vito', predicate='uses', object='PostgreSQL 18')
```

### `kg_invalidate_fact(assertion_id, valid_to?)`

Manually expire a fact. Rare — usually obsoleted by a new
`kg_add_fact`.

### `kg_at(timestamp)`

"What was true at time T?" — returns all assertions valid at the
given ISO timestamp. Time-travel query.

### `kg_timeline(subject?, predicate?, object?, project?, limit?)`

Chronological view of all triples filtered by any of (s, p, o).

---

## Episodes (narrative memory)

### `memory_episode_save(narrative, outcome, concepts?, key_insight?, approaches_tried?, frustration_signals?, impact_score?, project?)`

Save a story — what was tried, what failed, what worked, the aha
moment. Different shape than `memory_save` (story-form, not digest-form).

```
outcome      : 'breakthrough' | 'failure' | 'routine' | 'discovery'
impact_score : 0.0–1.0
```

### `memory_episode_recall(query?, concepts?, outcome?, project?, min_impact?, limit?)`

Search episodes by concepts / outcome / impact. Use for "show me
similar struggles" or "find a breakthrough on X".

---

## Self-improvement (rules, errors, patterns)

### `self_error_log(description, category, severity?, fix?, context?, project?, tags?)`

One-time error note. No auto-rule. Returns `error_id` and any pattern
it matched.

```
category : 'bug' | 'database' | 'api' | 'ui' | 'config' | 'logic' | 'test' | ...
severity : 'low' | 'medium' (default) | 'high' | 'critical'
```

### `learn_error(file, error, root_cause, fix, pattern)`

**Preferred** for reproducible errors. After **N≥3** identical patterns
auto-consolidates into a rule.

### `self_insight(action='list'|'add'|...)`

Manage learned insights. Read-mostly. Use `action='list'` for a
quick audit.

### `self_rules(action='list'|'add'|'update'|'remove')`

Manage behavioural rules. Auto-populated by `learn_error`
consolidation.

### `self_rules_context(project?, categories?, phase?)`

**Load behavioural rules at session start.** Returns a curated set
filtered by current project + phase. The `phase` filter uses
`phase:X` tags (zero-migration mechanic).

### `self_patterns(view?, project?, days?)`

Frequency analysis of error categories / repetitive contexts. Good
for retros.

### `self_reflect(reflection, task_summary, outcome?, project?, tags?)`

Save a meta-reflection record (`type='reflection'`, auto-tagged
`self-reflection`). For weekly retrospectives.

### `rule_set_phase(rule_id, phase?)`

Tag a rule with a phase (`van`/`plan`/`creative`/`build`/`reflect`/
`archive`) — `self_rules_context(phase=)` then filters by it.

---

## Tasks & phases (workflow state machine)

### `classify_task(description, project?)`

Returns `{level: 1-4, suggested_phases, estimated_tokens}`. Use it
before deciding whether a task needs the full workflow.

### `task_create(task_id, description, level?)`

Open a task in the `van` phase. If `level` is omitted, it's auto-
classified.

### `phase_transition(task_id, new_phase, artifacts?, notes?)`

Advance the task. Constrained by the level (e.g. L1 only allows
`van → build → reflect → archive`; you cannot enter `plan`).

### `task_phases_list(task_id)`

Chronological list of phases for a task. Good for postmortems.

### `save_decision(title, options, criteria_matrix, selected, rationale, discarded?, project?, tags?)`

**Structured** architectural decision. Auto-tagged `structured` and
`type=decision`. Surfaces in `memory_recall(decisions_only=True)`.

```
options          : [{name, pros[], cons[], unknowns[]}, ...]
criteria_matrix  : {speed: {A: 5, B: 2}, cost: {A: 3, B: 4}, ...}
selected         : "A"
rationale        : why A won — 1–3 sentences
```

---

## Intents (user-prompt capture)

### `save_intent(prompt, project?, session_id?, source?)`

Persist a user prompt for later analysis. **Auto-called** by the
`user-prompt-submit` hook if installed.

### `list_intents(limit?, project?)`

Recent captured intents. For "what was the user asking about
yesterday".

### `search_intents(query, project?, limit?)`

Full-text search on captured intents. Useful when the same question
keeps coming back.

---

## Skills (procedural knowledge)

### `memory_skill_get(name)`

Look up a skill by name. Returns trigger pattern, steps, success
rate. If unknown, returns `{error: 'Skill not found'}`.

### `memory_skill_update(name, ...)`

Bump usage / success / steps. Auto-called when a skill triggers.

---

## Workflows (learning from past task structure)

### `workflow_predict(task_description)`

Returns `{found, success_probability, avg_duration_ms, suggested_steps}`.
**If `confidence < 0.3`, ask the user about approach** before
diving in.

### `workflow_learn(name, steps)`

Record a successful workflow shape so the predictor can match it
later.

### `workflow_track(workflow_id, outcome)`

Close the loop on a tracked workflow with the actual outcome —
trains the predictor.

---

## Cross-project / corpus tools

### `analogize(text, exclude_project?, only_types?, limit?, min_score?)`

**Cross-project analogy.** Jaccard similarity + Dempster-Shafer
fusion. Use when you want to bring sibling-project context
deliberately ("we did this in vito; can we adapt it here?").

### `benchmark(scenarios?, limit?)`

Run the retrieval test suite (R@1/5/10, latency p50/p95). Good after
tuning recall pipeline.

### `ingest_codebase(path, languages?, project?)`

Index a codebase via tree-sitter (9 languages: Python, JS/TS, Go,
Rust, Java, C#, PHP, Ruby, C/C++). Cheaper than per-line reads,
yields structured graph.

### `file_context(path)`

**Pre-edit guard.** Returns `{file, risk_score, warnings, hot_spots}`.
Call before `Edit`/`Write` if `risk_score > 0.5`.

---

## Reflection / observability

### `memory_observe(tool_name, summary, observation_type?, files_affected?, project?)`

Record a tool-use observation. Auto-called by the `post-tool-use`
hook (opt-in). Useful raw for reflection.

### `memory_context_build(query, project?)`

Spreading-activation context bundle (knowledge + episodes + skills)
for a query. Heavier than `memory_recall` — use when you want
breadth over precision.

### `memory_wiki_generate(project?)`

Render the per-project Markdown digest (Top Decisions / Active
Solutions / Conventions / Recent Changes) to
`<MEMORY_DIR>/wikis/<project>.md`. Deterministic, no LLM.

### `memory_export(project?, limit?)`

JSON dump for backup or migration (see Knowledge CRUD).

### `memory_graph_stats()` / `memory_graph_index(limit?)`

(See Knowledge graph section above.)

---

## Quick reference — one-liners

```
session_init(project='X')                       # session opens
self_rules_context(project='X')                 # load behavioural rules
memory_recall(query='...', project='X')         # before any task
file_context(path='/abs/path')                  # before Edit/Write
memory_save(type='solution', content='...', tags=['reusable','tech'])  # after work
learn_error(file=..., error=..., root_cause=..., fix=..., pattern=...) # on bug
session_end(session_id='...', summary='...', next_steps=[...])         # session closes
```

When in doubt: **`session_init` → `memory_recall` → work → `memory_save`
→ `session_end`.**
