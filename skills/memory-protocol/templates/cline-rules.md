# Memory Protocol — Cline rules

You have access to a persistent cross-session memory via the
`memory` MCP server. Knowledge survives between sessions and is
shared across agents working on the same project.

## The five non-negotiables

1. **Session start → call `session_init` first, then `memory_recall`.**
2. **Before any non-trivial task → `memory_recall(query, project)`.**
3. **After every significant action → `memory_save` immediately.**
4. **On error / stuck → `learn_error` (or `self_error_log`).**
5. **End of session → `session_end` with summary + next_steps + pitfalls.**

## Save discipline

Use this digest template (≤ 30 lines):

```
ЧТО:    one-line summary
ПРОЕКТ: project name
ФАЙЛЫ:  absolute paths to the key files
СТЕК:   language / framework / version
ПОДХОД: 3–7 key steps
НЮАНСЫ: gotchas, edge cases
```

For decisions add a **WHY** section (alternatives, trade-offs).

Tags must include `["reusable", "<tech>"]` for cross-project recipes.

## Recall discipline

1. `memory_recall(query, project=<current>)` — current project first.
2. Empty / contradicting → `analogize(text=query, exclude_project=...)`.
3. Still empty → context7 / WebSearch / first principles.

## Stuck rule

After 3 identical errors STOP. `memory_save(type='lesson', stuck=...)`
+ ask the user with A/B/C options.

## Anti-patterns

- Never ask "should I save this?" — save without asking.
- Never batch saves until end of session.
- Never recall and ignore — cite the id + 1-line gist.
- Never save raw terminal dumps.
- Never guess a convention you can recall.

For the full tool reference and per-IDE setup see
`~/total-agent-memory/skills/memory-protocol/references/`.
