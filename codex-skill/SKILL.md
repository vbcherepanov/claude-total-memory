---
name: memory
description: >
  Activate this skill when starting a new session, beginning a new task,
  saving knowledge, recalling past decisions, or after completing significant work.
  Also activate on errors to log them for pattern analysis.
  Relevant when the user asks about memory, past context, lessons learned,
  decisions history, project conventions, or self-improvement.
---

## Persistent Memory System

You have access to a persistent cross-session memory via MCP tools.
Knowledge survives between sessions and is shared across agents working on the same project.

### Session Start

Always run these two calls before any work:

```
self_rules_context(project="<project>")
memory_recall(query="<current task description>", project="<project>")
```

If relevant knowledge is found, mention it briefly and apply it.

### Auto-Save Rules

Save knowledge automatically -- never ask the user whether to save.

| Event | type | What to save |
|-------|------|--------------|
| Architectural decision | `decision` | Decision + WHY + rejected alternatives |
| Non-trivial bug fix | `solution` | Symptom -> root cause -> fix |
| Gotcha or pitfall discovered | `lesson` | Expected vs actual + takeaway |
| Infrastructure or config setup | `fact` | Config details + key parameters |
| Project pattern established | `convention` | Rule + code example |
| Session ending | `fact` | Summary of what was done + what remains |

Format:

```
memory_save(
    content="Concise, actionable description",
    type="decision|solution|lesson|fact|convention",
    project="<project>",
    tags=["relevant", "tags"],
    context="Why this matters. For decisions: always explain WHY."
)
```

Do NOT save: trivial edits, intermediate steps, information obvious from the code.

### Error Logging

On any error (command failure, wrong assumption, API error, timeout), log it automatically:

```
self_error_log(
    description="What went wrong",
    category="code_error|logic_error|config_error|api_error|timeout|loop_detected|wrong_assumption|missing_context",
    severity="low|medium|high|critical",
    fix="How it was fixed (empty if unresolved)",
    project="<project>"
)
```

When `pattern_detected: true` is returned (3+ similar errors), extract an insight:

```
self_insight(action="add", content="Generalized lesson", category="<error_category>", source_error_ids=[...])
```

### Self-Improvement Pipeline

Errors -> Insights -> Rules (SOUL).

- Insights with importance >= 5 and confidence >= 0.8 can be promoted to rules
- Rules are loaded at session start via `self_rules_context`
- Rate rules after task completion: `self_rules(action="rate", id=N, success=true/false)`
- At session end, reflect: `self_reflect(reflection="...", task_summary="...", outcome="success|partial|failure")`

### Privacy

The server auto-redacts API keys, JWTs, passwords, emails, and credit cards before storage.
Use `<private>...</private>` tags to explicitly exclude sensitive content:

```
memory_save(content="Connected to <private>prod-host:5432</private>", type="fact", project="<project>")
```

### Quick Reference

| Tool | Purpose |
|------|---------|
| `memory_recall` | Search past knowledge (4-tier: keyword, semantic, fuzzy, graph) |
| `memory_save` | Store new knowledge with type, project, tags |
| `memory_update` | Supersede existing knowledge with a new version |
| `memory_search_by_tag` | Browse knowledge by tag |
| `memory_observe` | Lightweight file change tracking (30-day retention) |
| `memory_timeline` | Browse session history |
| `memory_stats` | Health metrics and storage info |
| `self_error_log` | Log structured errors for pattern analysis |
| `self_insight` | Manage insights (add/upvote/downvote/promote) |
| `self_rules` | Manage behavioral rules (list/fire/rate/suspend) |
| `self_rules_context` | Load active rules at session start |
| `self_patterns` | Analyze error patterns and improvement trends |
| `self_reflect` | Save session reflections |
