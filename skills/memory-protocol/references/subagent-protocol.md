# Sub-agent memory protocol

Sub-agents (e.g. `Agent(subagent_type='php-pro')`,
`Agent(subagent_type='golang-pro')`, `code-reviewer`,
`security-auditor`, etc.) get the **same MCP tools** as the main
agent. The save/recall discipline is identical — the only difference
is they should *receive* recalled context from the parent rather than
re-search.

---

## Header to inject into every sub-agent prompt

Paste this near the top of any sub-agent prompt (after role / domain
description). The placeholders in `{{...}}` are filled by the parent.

```
## Memory protocol (mandatory)

This task runs in project **{{project}}**, branch **{{branch}}**.

Context already retrieved by the parent — apply before re-searching:
{{recalled_records — one bullet per record: id + 1-line gist}}

Before producing your answer:

1. memory_recall(query="{{specific task description}}", project="{{project}}", limit=3)
   — only if the recalled records above are insufficient.
2. If still empty: analogize(text="{{task description}}", exclude_project="{{project}}")
3. If still empty: proceed with first-principles reasoning.

After producing your answer (mandatory if reusable):

4. memory_save(
       type='{{solution|decision|convention|lesson}}',
       content='ЧТО: ...\nПРОЕКТ: {{project}}\nФАЙЛЫ: ...\nСТЕК: ...\nПОДХОД: ...\nНЮАНСЫ: ...',
       project='{{project}}',
       tags=['reusable', '{{tech}}'],
   )

Stuck rules: after 3 identical failures STOP, do NOT guess. Save the
stuck-state with memory_save(type='lesson', tags=['stuck',...]) and
return BLOCKED to the parent with the question that needs human input.
```

---

## When sub-agents lack direct MCP access

Some sub-agents run in restricted environments without MCP tools.
For those, use the bash bridge:

```bash
# Recall (read-only):
lookup-memory "<query>"
lookup-memory --project NAME "<query>"

# Save (returns the new id):
~/total-agent-memory/ollama/save_memory.sh \
    --type solution \
    --project NAME \
    --tags "reusable,go,concurrency" \
    --content "ЧТО: ...\nПРОЕКТ: ...\n..."
```

Both scripts wrap the local Ollama instance for embedding and the
SQLite DB directly — no MCP server needed.

---

## Per-sub-agent tuning

Different sub-agents care about different recall facets. Hint the
sub-agent by passing an extra `recall_focus` line in its prompt:

| Sub-agent | Recall focus |
|---|---|
| `architect-reviewer` / `code-reviewer` | `decisions_only=true` + `expand_context=true` |
| `security-auditor` | `tag='security'` + `severity!='low'` filter |
| `database-administrator` / `sql-pro` | `project=<current>` + `tag='database'` |
| `frontend-developer` / `vue-expert` / `react-specialist` | `tag='ui'` + recent only (last 30 days) |
| `golang-pro` / `php-pro` / `python-pro` | `tag='<lang>'` + cross-project `analogize` |
| `data-scientist` | `tag='ml'` + cross-project |
| `performance-engineer` | `tag='performance'` + `benchmark()` snapshots |

---

## Save tags discipline (cross-agent)

For a save to be findable across sub-agents, tags need to follow a
canonical vocabulary. The repo ships
`vocabularies/canonical_topics.txt` with ~86 entries — `canonical_tags.normalise_tags()`
auto-maps common synonyms (e.g. `js → javascript`, `pg → postgresql`).
Free-form tags are kept alongside the canonical ones, so legacy synonym
recall still works.

When in doubt, use:

- A **language tag** (`go`, `python`, `typescript`, `php`, ...)
- A **layer tag** (`backend`, `frontend`, `database`, `infra`,
  `testing`, ...)
- An **intent tag** (`reusable` if the recipe applies elsewhere;
  `incident` for postmortems; `decision` for architectural choices)

---

## Parallelisation rules

When the parent spawns 3+ sub-agents in parallel:

1. Each sub-agent gets its own `session_id` (`<parent_session>-<sub>-<n>`)
   so the audit trail is clean.
2. Pass the `branch` argument to every sub-agent so saves don't
   cross-pollute branches.
3. Disjoint scopes — never have two parallel sub-agents writing to
   the same file (memory will dedup the saves but file edits will
   conflict).
4. After all sub-agents return, the parent does a **single**
   `memory_save(type='solution')` summarising the combined result.
   Don't have each sub-agent save the same outer summary.

This is the lesson from the v7.1+v8.0 wave — 10 parallel agents in 3
waves, zero merge conflicts (memory id `3624`).

---

## Anti-patterns specific to sub-agents

- **Re-recalling what the parent already passed in.** Wastes tokens
  and clock; if the parent gave you "id 3883: argpartition fix",
  don't re-search for `argpartition`.
- **Saving without tags.** Untagged saves are nearly invisible to
  other sub-agents and to `analogize`.
- **Saving terminal output.** Same digest discipline as main agent —
  ≤ 30 lines, no diff dumps.
- **Skipping the BLOCKED return on `stuck`.** If the sub-agent guesses
  past the stuck point, the parent can't intervene.
