# total-agent-memory

> **The only memory layer that learns _how_ you work — not just _what_ you said.**
> Persistent, local memory for AI coding agents: Claude Code, Codex CLI, Cursor, any MCP client.
> Temporal knowledge graph · procedural memory · AST codebase ingest · cross-project analogy · 3D WebGL visualization.

[![Version](https://img.shields.io/badge/version-9.0.0-8ad.svg)]()
[![Tests](https://img.shields.io/badge/tests-830%2B%20passing-4a9.svg)]()
[![LongMemEval R@5](https://img.shields.io/badge/LongMemEval%20R@5-96.2%25-4a9.svg)](evals/longmemeval-2026-04-17.json)
[![LoCoMo Acc](https://img.shields.io/badge/LoCoMo%20Acc-0.596-4a9.svg)](benchmarks/results/)
[![vs Supermemory](https://img.shields.io/badge/vs%20Supermemory-%2B10.8pp-4a9.svg)](docs/vs-competitors.md)
[![p50 latency](https://img.shields.io/badge/p50%20warm-0.065ms-4a9.svg)](evals/results-2026-04-17.json)
[![Local-First](https://img.shields.io/badge/100%25-local-4a9.svg)]()
[![License](https://img.shields.io/badge/license-MIT-fa4.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/protocol-MCP-blue.svg)](https://modelcontextprotocol.io)
[![npm](https://img.shields.io/badge/npm-%40vbch%2Ftotal--agent--memory--client-cb3837.svg)](https://www.npmjs.com/package/@vbch/total-agent-memory-client)
[![Donate](https://img.shields.io/badge/PayPal-Donate-00457C.svg?logo=paypal&logoColor=white)](https://PayPal.Me/vbcherepanov)

**Why this, not mem0 / Letta / Zep / Supermemory / Cognee?** → [docs/vs-competitors.md](docs/vs-competitors.md)

---

## Table of contents

- [The problem it solves](#the-problem-it-solves)
- [60-second demo](#60-second-demo)
- [Benchmarks — how it compares](#benchmarks--how-it-compares)
- [Competitor comparison](#competitor-comparison)
- [What you get](#what-you-get)
- [Architecture](#architecture)
- [Install](#install)
- [Quick start](#quick-start)
- [CLI: `lookup-memory` for sub-agents](#cli-lookup-memory-for-sub-agents)
- [MCP tools reference](#mcp-tools-reference-60-tools)
- [TypeScript SDK](#typescript-sdk)
- [Dashboard](#dashboard-localhost37737)
- [Update](#update)
- [Upgrading from v8.x to v9.0](#upgrading-from-v8x-to-v90)
- [Upgrading from v7.x to v8.0](#upgrading-from-v7x-to-v80)
- [Ollama setup](#ollama-setup-optional-but-recommended)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Support the project](#support-the-project)
- [Philosophy & license](#philosophy)

---

## The problem it solves

**AI coding agents have amnesia.** Every new Claude Code / Codex / Cursor session starts from zero. Yesterday's architectural decisions, bug fixes, stack choices, and hard-won lessons vanish the moment you close the terminal. You re-explain the same things, re-discover the same solutions, paste the same context into every new chat.

**`total-agent-memory` gives the agent a persistent brain — on your machine, not in someone else's cloud.**

Every decision, solution, error, fact, file change, and session summary is:

- **Captured** — explicitly via `memory_save` or implicitly via hooks on file edits / bash errors / session end
- **Linked** — automatically extracted into a knowledge graph (entities, relations, temporal facts)
- **Searchable** — 6-stage hybrid retrieval (BM25 + dense + graph + CrossEncoder + MMR + RRF fusion), **96.2% R@5 on public LongMemEval**
- **Private** — 100% local. SQLite + FastEmbed + optional Ollama. No data leaves your machine.

---

## 60-second demo

```
You:     "remember we picked pgvector over ChromaDB because of multi-tenant RLS"
Claude:  ✓ memory_save(type=decision, content="Chose pgvector over ChromaDB",
                       context="WHY: single Postgres, per-tenant RLS")

[3 days later, different session, possibly different project directory:]

You:     "why did we pick pgvector again?"
Claude:  ✓ memory_recall(query="vector database choice")
         → "Chose pgvector over ChromaDB for multi-tenant RLS. Single DB
            instance, row-level security per tenant."
```

It's not just retrieval. It's procedural too:

```
You:     "migrate auth middleware to JWT-only session tokens"
Claude:  ✓ workflow_predict(task_description="migrate auth middleware...")
         → confidence 0.82, predicted steps:
             1. read src/auth/middleware.go + tests
             2. update session fixtures in tests/
             3. run migration 0042
             4. regenerate OpenAPI spec
           similar past: wf#118 (success), wf#93 (success)
```

---

## Benchmarks — how it compares

**Public LongMemEval benchmark** ([xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), 470 questions, the dataset everyone publishes against):

```
                   R@5 (recall_any) on public LongMemEval
                   ─────────────────────────────────────────
  100% ─┤
        │
  96.2% ┤  ████  ← total-agent-memory v7.0  (LOCAL, 38.8 ms, MIT)
  95.0% ┤  ████  ← Mastra "Observational"    (cloud)
        │  ████
        │  ████
  85.4% ┤  ████  ← Supermemory                (cloud, $0.01/1k tok)
        │  ████
        │  ████
        │  ████
   80%  ┤  ████
        └──────────────────────────────────────────
```

Reproducible: [`evals/longmemeval-2026-04-17.json`](evals/longmemeval-2026-04-17.json) · Runner: [`benchmarks/longmemeval_bench.py`](benchmarks/longmemeval_bench.py)

### Per-question-type breakdown (R@5 recall_any)

| Question type | Count | Our R@5 |
|---|---:|---:|
| knowledge-update | 72 | **100.0%** |
| single-session-user | 64 | **100.0%** |
| multi-session | 121 | 96.7% |
| single-session-assistant | 56 | 96.4% |
| temporal-reasoning | 127 | 95.3% ← bi-temporal KG pays off |
| single-session-preference | 30 | 80.0% ← weakest spot |
| **TOTAL** | **470** | **96.2%** |

### Latency profile

```
  p50 (warm)   ▌ 0.065 ms
  p95 (warm)   ▌▌ 2.97 ms
  LongMemEval  ▌▌▌▌▌ 38.8 ms/query   ← includes embedding + CrossEncoder rerank
  p50 (cold)   ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌ 1333 ms  ← first query after process start
```

Warm / cold reproducible from [`evals/results-2026-04-17.json`](evals/results-2026-04-17.json).

---

## Competitor comparison

We're not replacing chatbot memory — we're occupying the **coding-agent + MCP + local** niche.

| | mem0 | Letta | Zep | Supermemory | Cognee | LangMem | **total-agent-memory** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Funding / status | $24M YC | $10M seed | $12M seed | $2.6M seed | $7.5M seed | in LangChain | self-funded OSS |
| Runs 100% local | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | **✅** |
| MCP-native | via SDK | ❌ | 🟡 Graphiti | 🟡 | ❌ | ❌ | **✅ 60+ tools** |
| Knowledge graph | 🔒 $249/mo | ❌ | ✅ | ✅ | ✅ | ❌ | **✅** |
| **Temporal facts** (`kg_at`) | ❌ | ❌ | ✅ | ❌ | 🟡 | ❌ | **✅** |
| **Procedural memory** | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | **✅ `workflow_predict`** |
| **Cross-project analogy** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ `analogize`** |
| **Self-improving rules** | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | **✅ `learn_error`** |
| **AST codebase ingest** | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | **✅ tree-sitter 9 lang** |
| **Pre-edit risk warnings** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ `file_context`** |
| 3D WebGL graph viewer | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | **✅** |
| Price for graph features | $249/mo | free | cloud | usage | free | free | **free** |

Full side-by-side with pricing, latency, accuracy, "when to pick each" → [docs/vs-competitors.md](docs/vs-competitors.md).

---

## What you get

### Eight capabilities nobody else ships

| Capability | Tool | One-liner |
|---|---|---|
| 🧠 **Procedural memory** | `workflow_predict` / `workflow_track` | "How did I solve this last time?" — predicts steps with confidence |
| 🔗 **Cross-project analogy** | `analogize` | "Was there something like this in another repo?" — Jaccard + Dempster-Shafer |
| ⚠️ **Pre-edit risk warnings** | `file_context` | Surfaces past errors / hot spots on the file you're about to edit |
| 🛡 **Self-improving rules** | `learn_error` + `self_rules_context` | Bash failures → patterns → auto-consolidated behavioral rules at N≥3 |
| 🕰 **Temporal facts** | `kg_add_fact` / `kg_at` | Append-only KG with `valid_from`/`valid_to` — query what was true at any point |
| 🎯 **Task workflow phases** | `classify_task` / `phase_transition` | Automatic L1-L4 complexity classification, state machine across van/plan/creative/build/reflect/archive |
| 🧩 **Structured decisions** | `save_decision` | Options + criteria matrix + rationale + discarded → searchable decision records with per-criterion embeddings |
| 💸 **Token-efficient retrieval** | `memory_recall(mode="index")` + `memory_get` | 3-layer workflow: compact IDs → timeline → batched full fetch. ~83% token saving on typical queries |

### Plus the basics done well

- **6-stage hybrid retrieval** (BM25 + dense + fuzzy + graph + CrossEncoder + MMR, RRF fusion) — 96.2% R@5 public
- **Multi-representation embeddings** — each record embedded as raw + summary + keywords + questions + compressed
- **AST codebase ingest** — tree-sitter across 9 languages (Python, TS/JS, Go, Rust, Java, C/C++, Ruby, C#)
- **Auto-reflection pipeline** — `memory_save` → LaunchAgent file-watch → graph edges appear ~30 s later
- **rtk-style content filters** — strip noise from pytest / cargo / git / docker logs while preserving URLs, paths, code
- **3D WebGL knowledge graph viewer** — 3,500+ nodes, 120,000+ edges, click-to-focus, filters
- **Hive plot & adjacency matrix** — alternate graph views sorted by node type
- **A2A protocol** — memory shared between multiple agents (backend + frontend + mobile in a team)
- **`design-explore` skill** — drop-in Claude Code skill that walks L3-L4 tasks through options → criteria matrix → `save_decision` before code (see `examples/skills/design-explore/SKILL.md`)
- **`<private>...</private>` inline redaction** in any saved content
- **Cloud LLM/embed providers** with per-phase routing (OpenAI / Anthropic / OpenRouter / Together / Groq / Cohere / any OpenAI-compat)
- **`activeContext.md` Obsidian projection** for human-readable session state
- **Phase-scoped rules** (`self_rules_context(phase="build")`) — ~70% token reduction

---

## Architecture

```
                  ┌─────────────────────────────────────────────────┐
                  │             Your AI coding agent                │
                  │   (Claude Code · Codex CLI · Cursor · any MCP)  │
                  └──────────────────────┬──────────────────────────┘
                                         │ MCP (stdio or HTTP)
                                         │ 60+ tools
                  ┌──────────────────────▼──────────────────────────┐
                  │            total-agent-memory server             │
                  │    ┌──────────────┐  ┌────────────────────┐     │
                  │    │ memory_save  │  │  memory_recall      │     │
                  │    │ memory_upd   │  │  6-stage pipeline:  │     │
                  │    │ kg_add_fact  │  │  BM25  (FTS5)       │     │
                  │    │ learn_error  │  │  + dense (FastEmbed)│     │
                  │    │ file_context │  │  + fuzzy            │     │
                  │    │ workflow_*   │  │  + graph expansion  │     │
                  │    │ analogize    │  │  + CrossEncoder †   │     │
                  │    │ ingest_code  │  │  + MMR diversity †  │     │
                  │    └──────┬───────┘  │  → RRF fusion       │     │
                  │           │          └──────────┬──────────┘     │
                  └───────────┼─────────────────────┼────────────────┘
                              │                     │
                  ┌───────────▼─────────────────────▼────────────────┐
                  │                   Storage                         │
                  │  ┌────────────┐  ┌────────────┐  ┌─────────────┐ │
                  │  │  SQLite    │  │  FastEmbed │  │   Ollama    │ │
                  │  │  + FTS5    │  │  HNSW      │  │  (optional) │ │
                  │  │  + KG tbls │  │  binary-q  │  │  qwen2.5-7b │ │
                  │  └────────────┘  └────────────┘  └─────────────┘ │
                  └───────────────────────────────────────────────────┘
                              │
                              │ file-watch + debounce
                  ┌───────────▼────────────────────────────────────┐
                  │  Auto-reflection pipeline  (LaunchAgent)        │
                  │  triple_extraction → deep_enrichment → reprs   │
                  │  (async, 10s debounce, drains in background)   │
                  └─────────────────────────────────────────────────┘
                              │
                  ┌───────────▼─────────────────────────────────────┐
                  │  Dashboard (localhost:37737)                     │
                  │   /           - stats, savings, queue depths   │
                  │   /graph/live - 3D WebGL force-graph           │
                  │   /graph/hive - D3 hive plot                   │
                  │   /graph/matrix - adjacency matrix             │
                  └─────────────────────────────────────────────────┘

  † CrossEncoder + MMR are on-demand via `rerank=true` / `diverse=true`
```

---

## Install

Two paths. Same 60+ tools, same dashboard, different deployment shapes.

### Platform matrix

| OS | Command | Background services |
|---|---|---|
| macOS 10.15+ | `./install.sh --ide claude-code` | LaunchAgents (`launchctl`) |
| Linux (Ubuntu 22.04+, Debian 12+, Fedora 38+) | `./install.sh --ide claude-code` | systemd `--user` |
| WSL2 (Windows 11 + Ubuntu/Debian) | `./install.sh --ide claude-code` | systemd `--user` — requires `/etc/wsl.conf` with `[boot] systemd=true`; otherwise falls back to shell-loop autostart |
| Windows 10/11 native | `.\install.ps1 -Ide claude-code` | Task Scheduler |

Full per-platform walkthrough, WSL2 Windows-host-vs-WSL IDE nuances, the
`wsl -e` MCP-command pattern, IDE coverage matrix, and uninstall/diagnostic
flows: **[docs/installation.md](docs/installation.md)**.

### Path A — native (macOS / Linux / WSL2)

```bash
git clone https://github.com/vbcherepanov/claude-total-memory.git ~/claude-memory-server
cd ~/claude-memory-server
bash install.sh --ide claude-code   # or: cursor | gemini-cli | opencode | codex
```

The installer:

1. Clones + creates `~/claude-memory-server/.venv/`
2. Installs deps from `requirements.txt` and `requirements-dev.txt`
3. Pre-downloads the FastEmbed multilingual MiniLM model
4. Registers the MCP server via `claude mcp add-json memory ...` (stored in `~/.claude.json`, the canonical store Claude Code actually reads)
5. Copies **all hooks** (`session-*`, `user-prompt-submit.sh`, `post-tool-use.sh`, `pre-edit.sh`, `on-bash-error.sh`, etc.) into `~/.claude/hooks/` and registers them in `~/.claude/settings.json`
6. Grants `permissions.allow` for 20+ `mcp__memory__*` tools so hook-driven calls don't prompt for confirmation
7. Installs **background services** for the current OS:
   - **macOS** — 4 LaunchAgents (`reflection`, `orphan-backfill`, `check-updates`, `dashboard`) under `~/Library/LaunchAgents/`
   - **Linux / WSL2** — 7 systemd `--user` units (`*.service`, `*.timer`, `*.path`) under `~/.config/systemd/user/`; gracefully degrades if `systemd --user` is unavailable (WSL without `/etc/wsl.conf`)
8. Applies all migrations to a fresh `memory.db`
9. Starts the dashboard at `http://127.0.0.1:37737`

Restart Claude Code → `/mcp` → `memory` should show **Connected** with 60+ tools.

### Path A — native (Windows 10/11)

```powershell
git clone https://github.com/vbcherepanov/claude-total-memory.git $HOME\claude-memory-server
cd $HOME\claude-memory-server
powershell -ExecutionPolicy Bypass -File install.ps1 -Ide claude-code
```

Same 9 steps as Unix, but:

- MCP config path is `%USERPROFILE%\.claude\settings.json` (or `.cursor\mcp.json`, etc.)
- Hooks copied to `%USERPROFILE%\.claude\hooks\` — `.ps1` versions (auto-capture, memory-trigger, user-prompt-submit, post-tool-use, pre-edit, on-bash-error, session-start/end, on-stop, codex-notify)
- Background services via **Task Scheduler**:
  - `total-agent-memory-reflection` — every 5 min (no native FileSystemWatcher equivalent)
  - `total-agent-memory-orphan-backfill` — daily 00:00 + 6h repetition
  - `total-agent-memory-check-updates` — weekly Mon 09:00
  - `ClaudeTotalMemoryDashboard` — AtLogon

### Uninstall

All installers preserve `~/.claude-memory/memory.db` and your config files; only services + hook registrations are removed.

```bash
./install.sh --uninstall          # macOS/Linux/WSL2 — removes LaunchAgents OR systemd units
.\install.ps1 -Uninstall          # Windows — unregisters Scheduled Tasks + cleans settings.json
```

### Diagnose

One-shot health check — prints ✓/✗ for each subsystem (OS detect, venv, MCP import, services, dashboard HTTP, Ollama, DB migrations):

```bash
bash scripts/diagnose.sh          # macOS / Linux / WSL2
.\scripts\diagnose.ps1            # Windows
```

Exit code 0 = all green, 1 = something broken.

### Path B — Docker (everything containerized, cross-platform)

```bash
git clone https://github.com/vbcherepanov/claude-total-memory.git
cd claude-total-memory
bash install-docker.sh --with-compose
```

Brings up 5 services:

| Service | Role | Exposed |
|---|---|---|
| `mcp` | MCP server (HTTP transport) | `127.0.0.1:3737/mcp` |
| `dashboard` | Web UI | `127.0.0.1:37737` |
| `ollama` | Local LLM runtime | `127.0.0.1:11434` |
| `reflection` | File-watch queue drainer | internal |
| `scheduler` | Ofelia cron (backfill + update check) | internal |

First run pulls `qwen2.5-coder:7b` (~4.7 GB) + `nomic-embed-text` (~275 MB) — 5–10 min cold start.

**GPU note:** Docker Desktop on macOS doesn't forward Metal. Native install is faster on Mac. On Linux with NVIDIA Container Toolkit, uncomment the `deploy.resources.reservations.devices` block in `docker-compose.yml`.

### Verify (both paths)

```
memory_save(content="install works", type="fact")
memory_stats()
```

Open <http://127.0.0.1:37737/> — dashboard, knowledge graph, token savings.

---

## Quick start

Once installed, in any Claude Code / Codex CLI / Cursor session:

**1. Resume where you left off** (auto on session start, but you can also invoke)

```
session_init(project="my-api")
→ {summary: "yesterday: migrated auth middleware to JWT",
   next_steps: ["update OpenAPI spec", "notify frontend team"],
   pitfalls: ["don't revert migration 0042 — dev DB already migrated"]}
```

**2. Save a decision (agent does this automatically after hooks are registered)**

```
memory_save(
  type="decision",
  content="Chose pgvector over ChromaDB for multi-tenant RLS",
  context="WHY: single Postgres instance, per-tenant row-level security",
  project="my-api",
  tags=["database", "multi-tenant"],
)
```

**3. Recall across sessions / projects**

```
memory_recall(query="vector database choice", project="my-api", limit=5)
→ RRF-fused results from 6 retrieval tiers
```

**4. Predict approach before starting a task**

```
workflow_predict(task_description="migrate auth middleware to JWT-only")
→ {confidence: 0.82, predicted_steps: [...], similar_past: [...]}
```

**5. Check a file's risk before editing** (auto via hook, also manual)

```
file_context(path="/Users/me/my-api/src/auth/middleware.go")
→ {risk_score: 0.71, warnings: ["last 3 edits caused test failures in ..."], hot_spots: [...]}
```

**6. Get full stats**

```
memory_stats()
→ {sessions: 515, knowledge: {active: 1859, ...}, storage_mb: 119.5, ...}
```

---

## CLI: `lookup-memory` for sub-agents

**New in v9.** Bash-friendly memory search for sub-agent workflows where launching the full MCP server would be overkill (e.g. `Bash(lookup-memory "fix slow Wave query")` from inside a Claude Code agent prompt).

Two equivalent commands ship with the package (registered as `[project.scripts]` entries — installed automatically by `./install.sh` or `./update.sh`):

```bash
lookup-memory "Caroline researched"          # human-readable bullets
ctm-lookup "Caroline researched"             # alias

lookup-memory --project myproj --limit 5 "auth flow"
lookup-memory --type solution --tag reusable "fix bug"
lookup-memory --json "claude code hooks"     # structured stdout for piping
```

**How it works:** opens the same `$CLAUDE_MEMORY_DIR/memory.db` the running MCP server uses → BM25 ranking via FTS5 → falls back to LIKE on older DBs. **Zero deps beyond the package.** No Ollama, no rag_chat.py, no ChromaDB required for the CLI path. Works on macOS, Linux, Windows.

```text
$ lookup-memory --project locomo_0 --limit 2 "adoption"
1. [synthesized_fact|locomo_0] Caroline is researching adoption agencies.
2. [synthesized_fact|locomo_0] Melanie congratulates Caroline on her adoption.
```

**Why two names?** `lookup-memory` matches the legacy bash script that older docs and sub-agent prompts reference (`~/claude-memory-server/ollama/lookup_memory.sh`). `ctm-lookup` is the project-prefixed canonical form. Both call into `claude_total_memory.lookup:main`.

**Migration note:** v7/v8 docs that pointed at `~/claude-memory-server/ollama/lookup_memory.sh` should be updated — the bash version still works for users with a manual install, but `./install.sh` / `./update.sh` clients on v9+ now get `lookup-memory` on PATH directly via the package's `[project.scripts]` entry.

---

## MCP tools reference (60+ tools)

### Tool categories

**Core retrieval (9):** `memory_save`, `memory_recall`, `memory_get`, `memory_update`, `memory_delete`, `memory_history`, `memory_extract_session`, `memory_relate`, `memory_search_by_tag`

**Knowledge graph (8):** `kg_add_fact`, `kg_invalidate_fact`, `kg_at`, `kg_timeline`, `memory_graph`, `memory_graph_index`, `memory_graph_stats`, `memory_concepts`

**Episodic / session (6):** `memory_episode_save`, `memory_episode_recall`, `session_init`, `session_end`, `memory_timeline`, `memory_history`

**Procedural / workflows (4):** `workflow_learn`, `workflow_predict`, `workflow_track`, `classify_task`

**Task phases (4, v8.0):** `task_create`, `phase_transition`, `task_phases_list`, `complete_task`

**Decisions (1, v8.0):** `save_decision`

**Intents (3, v8.0):** `save_intent`, `list_intents`, `search_intents`

**Self-improvement (5):** `self_rules`, `self_rules_context`, `self_insight`, `self_patterns`, `self_error_log`, `rule_set_phase` (v8.0)

**Pre-edit guard / error learning (3):** `file_context`, `learn_error`, `self_error_log`

**Analogy / cross-project (2):** `analogize`, `ingest_codebase`

**Reflection / consolidation (4):** `memory_reflect_now`, `memory_consolidate`, `memory_forget`, `memory_observe`

**Stats / export (5):** `memory_stats`, `memory_export`, `memory_self_assess`, `memory_context_build`, `benchmark`

**Skills (3):** `memory_skill_get`, `memory_skill_update`, `file_context`

Total: **60+ tools.** Each is documented below with input schema and example.

### Token-efficient 3-layer workflow

When you only know the topic but not which records matter, use progressive disclosure:

1. **Index** — `memory_recall(query="auth refactor", mode="index", limit=20)` → ~2 KB of `{id, title, score, type, project, created_at}` per hit. No content, no cognitive expansion.
2. **Timeline** — `memory_recall(query="auth refactor", mode="timeline", limit=5, neighbors=2)` → top-K hits padded with ±neighbours from the same session, sorted chronologically.
3. **Fetch** — `memory_get(ids=[3622, 3606])` → full content for ONLY the IDs you chose (max 50 per call, `detail="summary"` truncates to 150 chars).

**Typical saving:** 80-90 %% fewer tokens vs `memory_recall(detail="full", limit=20)` when you end up using 2-3 of the 20 hits.

<details>
<summary><b>Core memory (15)</b></summary>

`memory_recall` · `memory_get` · `memory_save` · `memory_update` · `memory_delete` · `memory_search_by_tag` · `memory_history` · `memory_timeline` · `memory_stats` · `memory_consolidate` · `memory_export` · `memory_forget` · `memory_relate` · `memory_extract_session` · `memory_observe`

</details>

<details>
<summary><b>Knowledge graph (6)</b></summary>

`memory_graph` · `memory_graph_index` · `memory_graph_stats` · `memory_concepts` · `memory_associate` · `memory_context_build`

</details>

<details>
<summary><b>Episodic memory & skills (4)</b></summary>

`memory_episode_save` · `memory_episode_recall` · `memory_skill_get` · `memory_skill_update`

</details>

<details>
<summary><b>Reflection & self-improvement (7)</b></summary>

`memory_reflect_now` · `memory_self_assess` · `self_error_log` · `self_insight` · `self_patterns` · `self_reflect` · `self_rules` · `self_rules_context`

</details>

<details>
<summary><b>Temporal knowledge graph (4)</b></summary>

`kg_add_fact` · `kg_invalidate_fact` · `kg_at` · `kg_timeline`

</details>

<details>
<summary><b>Procedural memory (3)</b></summary>

`workflow_learn` · `workflow_predict` · `workflow_track`

</details>

<details>
<summary><b>Pre-flight guards & automation (8)</b></summary>

`file_context` (pre-edit risk scoring) · `learn_error` (auto-consolidating error capture) · `session_init` / `session_end` · `ingest_codebase` (AST, 9 languages) · `analogize` (cross-project analogy) · `benchmark` (regression gate)

</details>

Full JSON schemas: `python -m claude_total_memory.cli tools --json` or open the dashboard at `localhost:37737/tools`.

---

## TypeScript SDK

For Node.js / browser / any TS project that isn't an MCP-native agent:

```bash
npm i @vbch/total-agent-memory-client
```

```ts
import { connectStdio } from "@vbch/total-agent-memory-client";

const memory = await connectStdio();

await memory.save({
  type: "decision",
  content: "Picked pgvector over ChromaDB for multi-tenant RLS",
  project: "my-api",
});

const hits = await memory.recallFlat({
  query: "vector database choice",
  project: "my-api",
  limit: 5,
});
```

Also ships LangChain adapter example, procedural-memory integration, and HTTP transport (for team / serverless setups).

Package repo: [github.com/vbcherepanov/total-agent-memory-client](https://github.com/vbcherepanov/total-agent-memory-client)

---

## Dashboard (localhost:37737)

- **`/`** — live stats, queue depths, token savings from filters, representation coverage
- **`/graph/live`** — 3D WebGL force-graph (Three.js), 3,500+ nodes / 120,000+ edges, click-to-focus, type filters, search
- **`/graph/hive`** — D3 hive plot, nodes on radial axes by type
- **`/graph/matrix`** — canvas adjacency matrix sorted by type
- **`/knowledge`** — paginated knowledge browser, tag filters
- **`/sessions`** — last 50 sessions with summaries + next steps
- **`/errors`** — consolidated error patterns
- **`/rules`** — active behavioral rules + fire counts
- **SSE-pill in header** — live reconnect indicator

Screenshots → [docs/screenshots/](docs/screenshots/) (coming)

---

## Update

```bash
cd ~/claude-memory-server
./update.sh
```

**7 stages:**

1. **Pre-flight** — disk check + DB snapshot (keeps last 7)
2. **Source pull** (git) or SHA-256-verified tarball
3. **Deps** — `pip install -r requirements.txt -r requirements-dev.txt` (only if hash changed)
4. **Full pytest suite** — aborts with snapshot if red
5. **Schema migrations** — `python src/tools/version_status.py`
6. **LaunchAgent reload** — reflection + backfill + update-check
7. **MCP reconnect notification** — in-app `/mcp` → `memory` → Reconnect

Manual equivalent:

```bash
cd ~/claude-memory-server
git pull
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python src/tools/version_status.py
.venv/bin/python -m pytest tests/
# in Claude Code: /mcp → memory → Reconnect
```

---

## Upgrading from v8.x to v9.0

v9 is **backward compatible**. Existing v8 calls and DB schema work unchanged — v9 is an infra release that adds pluggable backends, a public CLI for sub-agents, and LoCoMo benchmark wiring. Nothing is forcibly enabled.

### One-command upgrade

```bash
cd ~/claude-memory-server && ./update.sh
# pulls v9 src, installs new entry-points (ctm-lookup / lookup-memory),
# keeps existing memory.db untouched.
```

After upgrade, verify the new CLI is on PATH:

```bash
lookup-memory --limit 1 "any-query-from-your-history"
```

### What's new (no action required)

- **`lookup-memory` / `ctm-lookup`** CLI now installed alongside `claude-total-memory` MCP server (registered as `[project.scripts]` so `./install.sh` and `./update.sh` put them on PATH automatically). Sub-agent prompts that reference the legacy `~/claude-memory-server/ollama/lookup_memory.sh` script keep working; new prompts should prefer the package-installed name.
- **Embedding backends** stay on `fastembed` by default. Switch via `V9_EMBED_BACKEND=openai-3-large` (set `MEMORY_EMBED_API_KEY`) — costs ~$0.10/5k rows for re-embed, expected R@5 lift on conversational data.
- **Reranker backend** stays on `ce-marco` by default. `V9_RERANKER_BACKEND=bge-v2-m3` (or `off`) switches at runtime.
- **Subject-aware retrieval** is opt-in via `--subject-aware` in `benchmarks/locomo_bench_llm.py`. Future: surface as MCP tool flag.
- **No migrations.** Schema unchanged from v8.

### What requires manual action

- **Re-embed** (only if switching embedding model, otherwise skip):
  ```bash
  python -m scripts.reembed --backend openai-3-large --confirm
  ```
- **Old bash sub-agent prompts** that hardcode `~/claude-memory-server/ollama/lookup_memory.sh "query"` will keep working. To ride the new package install, replace with `lookup-memory "query"`.

### Breaking changes

None. All v8 MCP tools, env vars, hooks, and DB tables behave identically.

---

## Upgrading from v7.x to v8.0

v8.0 is **backward compatible** — your existing v7 installation keeps working unchanged. All new features are opt-in via MCP tool calls or env vars.

### One-command upgrade

```bash
cd ~/claude-memory-server && ./update.sh
# Applies migrations 011-013 idempotently, restarts LaunchAgents, updates dependencies
```

Then restart Claude Code: `/mcp restart memory`.

### What changes automatically

- **Migrations 011–013** apply on MCP startup (privacy_counters, task_phases, intents). Zero-downtime, idempotent.
- **Existing `memory_save`** calls keep working — they now additionally strip `<private>...</private>` sections if present.
- **Existing `memory_recall`** calls keep working — default mode is still `"search"`. New `mode="index"` is opt-in.
- **Existing `session_end`** calls keep working — `auto_compress=False` by default. Pass `auto_compress=True` to opt in.
- **Existing `self_rules_context`** calls keep working — default returns all rules (no phase filter).

### What requires manual setup

**1. Cloud providers** (only if you want to replace/augment Ollama):
```bash
export MEMORY_LLM_PROVIDER=openai       # or "anthropic"
export MEMORY_LLM_API_KEY=sk-...
export MEMORY_LLM_MODEL=gpt-4o-mini     # or "claude-haiku-4-5"
```
See [Cloud providers](#cloud-providers-optional) for OpenRouter / per-phase routing / Cohere examples.

**2. Install additional hooks** (for UserPromptSubmit capture + citation):
```bash
./install.sh --ide claude-code   # re-run installer; it now registers user-prompt-submit.sh hook
```
The hook is additive — existing hooks keep working.

**3. activeContext.md Obsidian integration** (if you want markdown projection):
```bash
export MEMORY_ACTIVECONTEXT_VAULT=~/Documents/project/Projects   # default
# Disable: export MEMORY_ACTIVECONTEXT_DISABLE=1
```
Each `session_end` writes `<vault>/<project>/activeContext.md`.

### Breaking changes

**None.** All v7 MCP tool signatures are preserved. New parameters are optional with safe defaults.

### Embedding dimension note

If you switch to a cloud embedding provider (`MEMORY_EMBED_PROVIDER=openai/cohere`), the server **will refuse to start** if existing DB embeddings have a different dimension than the new provider returns. This is deliberate — it prevents silent data corruption.

Either:
- Keep `MEMORY_EMBED_PROVIDER=fastembed` (default 384d) and only change the LLM provider, OR
- Re-embed the DB: `python src/tools/reembed.py --provider openai --model text-embedding-3-small`

### New MCP tools in v8.0

Quick reference — see full docs in [MCP tools reference](#mcp-tools-reference-60-tools):

| Tool | Purpose |
|---|---|
| `classify_task(description)` | Returns {level 1-4, suggested_phases, estimated_tokens} |
| `task_create(task_id, description)` | Starts state machine in "van" phase |
| `phase_transition(task_id, new_phase, artifacts?)` | Moves task through van/plan/creative/build/reflect/archive |
| `task_phases_list(task_id)` | Chronological phase history |
| `save_decision(title, options, criteria_matrix, selected, rationale, ...)` | Structured decision with per-criterion indexing |
| `memory_get(ids, detail)` | Batched full-content fetch for IDs from `memory_recall(mode="index")` |
| `save_intent` / `list_intents` / `search_intents` | UserPromptSubmit-captured prompts |
| `rule_set_phase(rule_id, phase)` | Tag a rule for phase-scoped loading |

Extended tools:
- `memory_recall(mode="index"|"timeline", decisions_only=False, ...)` — 3-layer token-efficient workflow
- `session_end(auto_compress=True, transcript=None, ...)` — LLM-generated summary
- `self_rules_context(phase="build"|"plan"|...)` — phase filter
- `save_knowledge(...)` — now strips `<private>...</private>` sections automatically

### Rollback plan

v8.0 doesn't remove any v7 functionality. If you hit an issue, you can:

1. Set env var to revert behaviour:
   ```bash
   export MEMORY_LLM_PROVIDER=ollama           # revert to local LLM
   export MEMORY_EMBED_PROVIDER=fastembed      # revert to local embeddings
   export MEMORY_ACTIVECONTEXT_DISABLE=1       # disable markdown projection
   export MEMORY_POST_TOOL_CAPTURE=0           # disable opt-in capture (default anyway)
   ```

2. Migrations 011/012/013 are additive (no `DROP` / `ALTER` on existing tables), so DB downgrade is not destructive — old code continues reading older tables.

3. Worst case: `git checkout v7.0.0 && ./update.sh --skip-migrations`.

---

## Ollama setup (optional but recommended)

**Without Ollama:** works fully — raw content is saved, retrieval via BM25 + FastEmbed dense embeddings.

**With Ollama:** you also get LLM-generated summaries, keywords, question-forms, compressed representations, and deep enrichment (entities, intent, topics).

```bash
brew install ollama     # or: curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5-coder:7b        # default — best quality/speed on M-series
ollama pull nomic-embed-text        # optional, alternative embedder
```

### Cloud providers (optional)

Use OpenAI, Anthropic, or any OpenAI-compat endpoint (OpenRouter, Together, Groq, DeepSeek, LM Studio, llama.cpp) instead of local Ollama.

**OpenAI:**
```bash
export MEMORY_LLM_PROVIDER=openai
export MEMORY_LLM_API_KEY=sk-...
export MEMORY_LLM_MODEL=gpt-4o-mini
```

**Anthropic:**
```bash
export MEMORY_LLM_PROVIDER=anthropic
export MEMORY_LLM_API_KEY=sk-ant-...
export MEMORY_LLM_MODEL=claude-haiku-4-5
```

**OpenRouter (100+ models via one endpoint):**
```bash
export MEMORY_LLM_PROVIDER=openai
export MEMORY_LLM_API_BASE=https://openrouter.ai/api/v1
export MEMORY_LLM_API_KEY=sk-or-...
export MEMORY_LLM_MODEL=anthropic/claude-haiku-4.5
```

**Per-phase routing** (cheap model for bulk, quality for compression):
```bash
export MEMORY_TRIPLE_PROVIDER=openai
export MEMORY_TRIPLE_MODEL=gpt-4o-mini
export MEMORY_ENRICH_PROVIDER=anthropic
export MEMORY_ENRICH_MODEL=claude-haiku-4-5
```

**Embeddings** (dimension must match existing DB or re-embed required):
```bash
export MEMORY_EMBED_PROVIDER=openai
export MEMORY_EMBED_MODEL=text-embedding-3-small  # 1536d
# or Cohere:
export MEMORY_EMBED_PROVIDER=cohere
export MEMORY_EMBED_API_KEY=...
```

### Model choice

| Model | Size | Use case |
|---|---|---|
| `qwen2.5-coder:7b` | 4.7 GB | **default** — best quality/speed ratio |
| `qwen2.5-coder:32b` | 19 GB | highest quality, needs 32 GB+ RAM |
| `llama3.1:8b` | 4.9 GB | general-purpose alternative |
| `phi3:mini` | 2.3 GB | low-RAM machines |

---

## Configuration

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_DB` | `~/.claude-memory/memory.db` | SQLite location |
| `MEMORY_LLM_ENABLED` | `auto` | `auto\|true\|false\|force` — LLM enrichment toggle |
| `MEMORY_LLM_MODEL` | `qwen2.5-coder:7b` | Ollama model for enrichment |
| `MEMORY_LLM_PROBE_TTL_SEC` | `60` | Cache TTL for Ollama availability probe |
| `MEMORY_LLM_TIMEOUT_SEC` | `60` | Global fallback timeout for Ollama requests (s) |
| `MEMORY_TRIPLE_TIMEOUT_SEC` | `30` | Timeout for deep triple extraction (s) |
| `MEMORY_ENRICH_TIMEOUT_SEC` | `45` | Timeout for deep enrichment (s) |
| `MEMORY_REPR_TIMEOUT_SEC` | `60` | Timeout for representation generation (s) |
| `MEMORY_TRIPLE_MAX_PREDICT` | `2048` | `num_predict` cap for triple extraction |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `MEMORY_EMBED_MODE` | `fastembed` | `fastembed\|sentence-transformers\|ollama` |
| `DASHBOARD_PORT` | `37737` | HTTP dashboard port |
| `MEMORY_MCP_PORT` | `3737` | HTTP MCP transport port (Docker path) |

> CPU-only / WSL hosts: if Ollama keeps timing out, lower `MEMORY_TRIPLE_MAX_PREDICT` before raising timeouts. `install-codex.sh` writes conservative defaults automatically.

Full config: see `claude_total_memory/config.py`.

---

## Roadmap

### Shipped in v9.0 (2026-04-25)
- ✅ **`lookup-memory` / `ctm-lookup` CLI** — bash entry-point for sub-agents, registered as `[project.scripts]` and installed by `./install.sh` / `./update.sh` (replaces manual `~/claude-memory-server/ollama/lookup_memory.sh`)
- ✅ **Pluggable embedding backends**: `openai-3-small`, `openai-3-large` (3072d), `bge-m3`, `e5-large`, `locomo-tuned-minilm` (fine-tuned on user data)
- ✅ **Pluggable reranker backends**: `ce-marco`, `bge-v2-m3`, `bge-large`, `off` (env `V9_RERANKER_BACKEND`, hot-swap)
- ✅ **Subject-aware retrieval** — LLM extracts (subject, action) from question → SQL graph lookup → DIRECT FACTS prepended to context (LoCoMo cat 1/2 lift)
- ✅ **Judge-weighted ensemble** — category-aware scoring rubric + abstain logic for LoCoMo-style adversarial gold
- ✅ **Fine-tune embedding pipeline** (`scripts/finetune_embedding.py`) — mine triplets from your data, train on top of MiniLM via `sentence-transformers`
- ✅ **Few-shot pair mining** (`scripts/mine_locomo_fewshot.py`) — augment per-category prompts with held-in (Q,A) pairs
- ✅ **Schema-specific graph extractor** (closed canonical predicate vocabulary, optional)
- ✅ **SSL fix for macOS Python.org installs** — `urllib` requests now use certifi by default
- ✅ **HTTP retry with exponential backoff** for embedding providers (5xx/timeout)
- ✅ LoCoMo benchmark integration (`benchmarks/locomo_bench_llm.py` with 14 ablation flags)

### Shipped in v8.0 (2026-04-19)
- ✅ Task workflow phases (L1-L4 classifier + 6-phase state machine)
- ✅ Structured `save_decision` with criteria matrix + multi-representation criterion indexing
- ✅ Cloud LLM/embed providers (OpenAI, Anthropic, Cohere, any OpenAI-compat)
- ✅ `session_end(auto_compress=True)` via LLM provider
- ✅ Progressive disclosure: `memory_recall(mode="index")` + `memory_get(ids)`
- ✅ `activeContext.md` Obsidian live-doc projection
- ✅ Phase-scoped rules via tag filter
- ✅ `<private>...</private>` inline redaction
- ✅ HTTP citation endpoints `/api/knowledge/{id}` + `/api/session/{id}`
- ✅ UserPromptSubmit + PostToolUse (opt-in) capture hooks
- ✅ Unified `install.sh --ide {claude-code|cursor|gemini-cli|opencode|codex}`

### Planned (v8.1+)
- Plugin marketplace publish (when Claude Code API opens)
- `has_llm()` per-phase provider caching
- GitHub Actions: install smoke tests + LongMemEval nightly

### Under research
- "Endless mode" — continuous session without hard boundaries (virtual sessions by idle >N hours)
- MLX local LLM integration (A1 plan from memory #3583)
- Speculative decoding for local path (+1.5-1.8× LLM speed)

---

## Support the project

**`total-agent-memory` is, and will always be, free and MIT-licensed.** No paid tier, no gated features, no "enterprise edition". The benchmarks on this page are the entire product.

If it's saving you hours of context-pasting every week and you want to help keep development going — or just say thanks — a donation means a lot.

<p align="center">
  <a href="https://PayPal.Me/vbcherepanov">
    <img src="https://img.shields.io/badge/Donate%20via%20PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white" alt="Donate via PayPal" height="42">
  </a>
</p>

### What your support funds

| | Goal |
|---|---|
| ☕ **$5** — a coffee | One evening of focused OSS work |
| 🍕 **$25** — a pizza | A new MCP tool end-to-end (design, code, tests, docs) |
| 🎧 **$100** — a weekend | A major feature: e.g. the preference-tracking module that closes the 80% gap on LongMemEval |
| 💎 **$500+** — a sprint | A release cycle: new subsystem + migrations + docs + benchmark artifact |

### Non-monetary ways to help (equally appreciated)

- ⭐ **Star the repo** — GitHub discovery runs on this
- 🐦 **Share benchmarks on X / HN / Reddit** — reach matters more than donations
- 🐛 **Open issues** with repro cases — bug reports are pure gold
- 📝 **Write a blog post** about how you use it
- 🔧 **Submit a PR** — fixes, new tools, new integrations
- 🌍 **Translate the README** — first docs in RU / DE / JA / ZH very welcome
- 💬 **Tell your team** — peer recommendations convert 10× better than marketing

### Commercial / consulting

- Building something that would benefit from a custom integration, on-prem deployment, or team-shared memory? **Email `vbcherepanov@gmail.com`** — open to contract work and partnerships.
- AI / dev-tools company whose roadmap overlaps? Same email — happy to talk.

---

## Philosophy

**MIT forever.** No commercial-license switch, no VC money, no dark patterns. The memory layer belongs to the developers using it, not to a SaaS vendor.

**Local-first is the product.** If you want a cloud memory service, mem0 and Supermemory are great. If you want your data on your disk, untouched by anyone else — this.

**Honest benchmarks.** Every number on this page is reproducible from the artifacts in `evals/` and the scripts in `benchmarks/`. If you can't reproduce a claim, open an issue — it's a bug.

---

## Contributing

- Open an issue before a large PR — saves everyone time.
- `pytest tests/` must stay green. Add tests for new tools.
- Update `evals/scenarios/*.json` if you change retrieval behavior.
- Docs-only / typo PRs welcome without discussion.

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <b>Built for coding agents. Runs on your machine. Free forever.</b><br>
  <a href="docs/vs-competitors.md">Compare to mem0 / Letta / Zep / Supermemory</a> ·
  <a href="evals/longmemeval-2026-04-17.json">Benchmark artifact</a> ·
  <a href="https://github.com/vbcherepanov/total-agent-memory-client">TypeScript SDK</a> ·
  <a href="https://PayPal.Me/vbcherepanov">Donate</a>
</p>
