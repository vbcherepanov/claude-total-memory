---
name: memory-protocol
version: 10.5.0
description: >
  Universal protocol for total-agent-memory MCP server.
  Activate at session start, before any non-trivial task, after every significant
  action, on errors, and at session end.
  Relevant whenever the user mentions: memory, recall, past context, decisions
  history, conventions, lessons learned, "продолжаем", "сохранись", "resume",
  or any reference to prior sessions / cross-session knowledge.
  Works with Claude Code, Codex CLI, Cursor, Cline, Continue, Aider,
  Windsurf, Gemini CLI, OpenCode, and any client that speaks MCP.
keywords:
  - memory_recall
  - memory_save
  - session_init
  - session_end
  - kg_add_fact
  - workflow_predict
  - self_rules_context
---

# Memory Protocol — Universal Skill

You have access to a persistent cross-session memory via the
**total-agent-memory** MCP server (60+ tools). Knowledge survives between
sessions and is shared across agents working on the same project.

This skill is **universal** — the same MCP tools, the same triggers, the
same templates work in every supported environment (Claude Code, Codex,
Cursor, Cline, Continue, Aider, Windsurf, Gemini CLI, OpenCode). Hooks
and sub-agents are environment-specific and documented separately
(see `references/hooks-explained.md` and `references/ide-setup.md`).

## The five non-negotiables

1. **Session start → `session_init` first, then `memory_recall`.** Always.
   Skip only if you have already called `session_init` in this session.
2. **Before any non-trivial task → `memory_recall(query, project)`.** Use
   the recipe instead of guessing the convention.
3. **After every significant action → `memory_save` immediately.** Don't
   batch. A decision, a fix, a new convention — one save per fact, not
   one save per session.
4. **On error / bash non-zero / stuck → `learn_error` (or `self_error_log`)
   with root cause + fix.** Pattern auto-consolidates after 3 occurrences.
5. **End of session → `session_end` with summary + next_steps + pitfalls.**

If you only remember one thing: **`session_init` → `memory_recall` →
work → `memory_save` → `session_end`.**

## Trigger table — when to call what

| Event | Tool | Why |
|---|---|---|
| Session opens | `session_init(project)` | Returns previous summary + next_steps + pitfalls and marks them consumed. **Call first.** |
| Any task starts | `memory_recall(query, project)` | Recipe-first; never invent a convention twice. |
| Before edit/write to a file | `file_context(path)` | Returns risk_score + warnings (past errors on this path, hot spots). |
| Before architecture choice | `memory_recall` + `analogize(query, exclude_project)` | Cross-project analogy. |
| Bash returns non-zero (and reproducible) | `learn_error(file, error, root_cause, fix, pattern)` | Auto-consolidates to a rule after N≥3. |
| Architectural decision made | `save_decision(title, options, criteria_matrix, selected, rationale)` | Structured; auto-tagged `structured`; goes into recall with `decisions_only=True` filter. |
| Tech-stack/dependency/config change | `kg_add_fact(subject, predicate, object)` | Temporal; auto-invalidates old facts. |
| Solution that other projects could reuse | `memory_save(type='solution', tags=['reusable', '<tech>'])` | Surfaces in `analogize()` for sibling projects. |
| Reusable pattern / idiom | `memory_save(type='convention')` | |
| Lesson learned (regression / postmortem) | `memory_save(type='lesson')` | Higher recall weight on similar tasks. |
| Episode (story of how something was done) | `memory_episode_save(narrative, outcome)` | Narrative form — "what was tried, what failed, what worked". |
| Error worth a one-time note | `self_error_log(description, category, fix?)` | Cheaper than `learn_error`; no auto-rule. |
| User prompt that should be remembered | `save_intent(prompt, project)` | Auto-captured by hook if installed. |
| Major task starting | `workflow_predict(task_description)` | If `confidence < 0.3` ask the user about approach. |
| Major task ending | `workflow_track(workflow_id, outcome)` | Trains the predictor. |
| "What was the stack at date X?" | `kg_at(timestamp)` | Time-travel query. |
| "Find similar in other projects" | `analogize(text, exclude_project)` | Jaccard + Dempster-Shafer fusion. |
| Indexing an external repo | `ingest_codebase(path, languages)` | AST tree-sitter, 9 languages. |
| Show recent saves chronologically | `memory_timeline(limit)` | Replays session → session. |
| Build per-project digest | `memory_wiki_generate(project)` | Markdown to `<MEMORY_DIR>/wikis/<project>.md`. |
| Session closes ("сохранись" / "save") | `session_end(session_id, summary, next_steps, pitfalls)` + dual-write to Obsidian if available | Picked up by next `session_init`. |

A complete reference for every tool with arguments, return shape,
common mistakes, and short examples is in
`references/tool-cheatsheet.md`.

## Save discipline (template)

`memory_save` content **must** be a structured digest, not a terminal dump:

```
ЧТО:        one-line summary
ПРОЕКТ:     project name
ФАЙЛЫ:      absolute paths to the key files
СТЕК:       language/framework/version
ПОДХОД:     3–7 key steps, no filler
НЮАНСЫ:     gotchas, edge cases, what didn't work and why
```

For `type=decision` add **WHY**: context, alternatives considered,
trade-offs, when this rule applies and when it does not.

**Tags must include `["reusable", "<tech>"]`** when the recipe applies
to other projects (it surfaces in `analogize()`).

**Length cap: ≤ 30 lines.** If your save is longer, you have not
distilled — go again.

## Recall discipline

1. `memory_recall(query, project=<current>)` — start with the current project.
2. If empty / contradicting / stale → `analogize(text=query)` for sibling
   projects.
3. Still empty → `WebSearch` / `context7` / first-principles.
4. **Never guess a convention you can recall.**

If a recalled record names a file/function/flag and you are about to
*act on* it (not just answer a question about history), verify it
exists first (Read / grep). Memory is a claim about the past;
the source is the truth now.

## Self-improvement loop

- After 3 identical errors → stop, do not guess again.
  - `memory_save(type='lesson', stuck=...)`
  - Ask the user with A/B/C options (use `AskUserQuestion` if your
    environment supports it; otherwise plain text question).
- After every reusable solution → `memory_save(type='solution',
  tags=['reusable', ...])` so the next session does not redo the work.
- Periodically (or via `MEMORY_REFLECT_EVERY_N`):
  `memory_reflect_now()` consolidates duplicates, decays stale, and
  finds contradictions.

## Cross-cutting concerns

- **Privacy.** Wrap secrets in `<private>...</private>` — they are
  redacted before save. Never put real tokens / passwords / personal
  data in `memory_save` content; the redactor is best-effort, not
  absolute.
- **Per-project isolation.** Always pass `project=` so cross-project
  noise stays out of recall. `analogize()` is the right way to bring
  sibling-project context in deliberately.
- **Languages.** Russian and English work equally well in `query` and
  `content`. The smart router detects relational queries in both.

## Sub-agents

Sub-agents (e.g. `Agent(subagent_type='php-pro')`) get the **same**
MCP tools as the main agent. The protocol is identical:
`memory_recall` first, `memory_save` after. Pass any recipes you
already recalled into the sub-agent's prompt so it doesn't duplicate
the search.

For sub-agents that don't have direct MCP access, use the bash bridge:
```bash
lookup-memory "<query>"
lookup-memory --project NAME "<query>"
```

See `references/subagent-protocol.md` for the full sub-agent header
template.

## Hooks (optional but recommended)

Hooks let the protocol fire automatically. They are **opt-in** and
each one is documented in detail in `references/hooks-explained.md`:

| Hook | Trigger | Default action |
|---|---|---|
| `session-start.sh` | Session opens | Calls `session_init` and prints summary. |
| `pre-edit.sh` | Before file write | Calls `file_context(path)` and warns if `risk_score > 0.5`. |
| `post-tool-use.sh` | After every tool call | Optional: captures observation via `memory_observe`. |
| `user-prompt-submit.sh` | User prompt sent | Calls `save_intent`. |
| `on-stop.sh` | Session interrupted (context limit, Ctrl-C) | Writes recovery file; next `session_start` picks it up. |

A bash version (`*.sh`) and a PowerShell version (`*.ps1`) ship for
every hook so the same setup works on macOS, Linux, WSL2, and
native Windows.

## Per-IDE setup

Installation, hook wiring, and skill location differ per IDE.
The full matrix and per-IDE commands are in
`references/ide-setup.md`. Quick path:

```bash
~/total-agent-memory/install.sh --ide claude-code   # or codex|cursor|cline|continue|aider|windsurf|gemini-cli|opencode
```

The installer writes the right files to the right place (skill,
hooks, MCP server registration) and is idempotent — safe to re-run
on upgrade.

## Anti-patterns

These are the failure modes the protocol is designed to prevent:

- **Asking "do you want me to save this?"** Don't ask, save. The save
  is cheap; the question wastes a turn.
- **Batching saves until the end of the session.** A `decision` from
  10:15 stored at 17:00 already lost its context — store it now.
- **Generic queries.** `memory_recall(query="auth")` returns noise.
  `memory_recall(query="JWT refresh rotation in php-symfony auth middleware")`
  returns the recipe. Be specific.
- **Recalling and ignoring.** If you ran `memory_recall` and got a
  result, cite it briefly (id + 1-line gist) before applying it. The
  user can correct stale memory only if they see it.
- **Guessing a convention you could recall.** L3 first principles is
  great; "L0 made it up" is not.
- **Saving a terminal dump.** Distill to the digest template above.
  30-line cap, no filler, no diff (git remembers diffs).

## Versioning

This skill targets total-agent-memory **v10.5+**.

- v10.0 added the 10 Beever-Atlas-style features (quality gate,
  canonical tags, importance, coref, contradiction, outbox,
  entity dedup, episodic, smart router, wiki).
- v10.1 added the async enrichment worker
  (`MEMORY_ASYNC_ENRICHMENT=true`).
- v10.5 ships this universal skill, expanded `install.sh` for all
  major IDEs, cross-platform hooks, sub-agent protocol, and
  documentation.

If your server is older, the protocol still applies — calls to
unsupported tools will return a "not implemented" error and
fall through.
