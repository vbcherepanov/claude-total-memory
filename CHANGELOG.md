# Changelog

All notable changes to total-agent-memory are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions use [Semantic Versioning](https://semver.org/).

## [13.0.0] — 2026-08-27 — MCP 2026-07-28, and honest benchmarks

**Every install created since the MCP Python SDK went 2.0 was broken.** The
2.x line dropped the `@Server.list_tools()` / `@Server.call_tool()` decorators
this server was built on, and the dependency was floored at `mcp[cli]>=1.0.0`,
so a fresh `pip` / `uvx` / `npx` / `brew` / `docker` install resolved 2.x and
died at import with `AttributeError: 'Server' object has no attribute
'list_tools'`. Existing installs kept working only because their pinned 1.x
never moved. **If you installed after mcp 2.0 shipped, upgrade.**

Fixing it turned out to be the whole story: once tools registered again, the
server was already speaking the stateless 2026-07-28 protocol, because the SDK
serves both eras from one process. This release makes that explicit, adds the
2026 result shapes, and then goes back over the benchmark numbers — where a
feedback loop had been quietly inflating them.

### Fixed — MCP SDK 2.x (breaking for anyone on a fresh install)
- `src/server.py` registers `tools/list` and `tools/call` through whichever
  API the installed SDK exposes: the 1.x decorators, or 2.x's
  `add_request_handler(method, params_model, handler)`. `MCP_SDK_ERA` reports
  which path is live.
- `mcp[cli]>=1.9,<3` in both `requirements.txt` and `pyproject.toml`. The
  upper bound is the actual lesson: an unbounded floor is how 2.0 reached
  users unannounced.
- A tool call arriving before `_bootstrap_session()` crashed the transport —
  `store.raw_append` sat outside the `try`. Now guarded.
- An unknown tool name returned `{"error": "Unknown tool"}` with
  `isError: false`. It now raises, so clients see a real error result.

### Added — protocol revision 2026-07-28
- **Stateless era served end-to-end.** `tools/list`, `server/discover` and
  `tools/call` all answer without an `initialize` handshake, with protocol
  metadata carried per-request in `params._meta`. The legacy handshake era
  keeps working from the same process, so clients on older SDKs are
  unaffected. `tests/test_mcp_protocol_e2e.py` drives both eras against a
  real subprocess.
- **`structuredContent` on every JSON-answering tool.** 2026-07-28 lets
  structured content be any JSON value, so clients receive the parsed object
  instead of re-parsing a string. Error results carry text only. No
  `outputSchema` is declared, so this is purely additive.
- **Behaviour annotations on all 74 tools** — `readOnlyHint`,
  `destructiveHint`, `idempotentHint`, `openWorldHint`. Clients use these to
  decide what runs without a confirmation prompt. 38 tools are read-only;
  `memory_delete`, `memory_forget`, `memory_update`, `kg_invalidate_fact` and
  the two rebuild tools are marked destructive. A test fails if a new tool
  ships unclassified.

### Added — Claude Code plugin
- `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `.mcp.json`
  and `hooks/hooks.json`: the MCP server, the `memory-protocol` skill and the
  seven capture hooks install in one step.

      /plugin marketplace add vbcherepanov/total-agent-memory
      /plugin install total-agent-memory@vbcherepanov

- `bin/tam_plugin_bootstrap.py` resolves a runnable server in order —
  an existing `.venv`, `PATH`, the plugin's own checkout, `uvx`/`npx`, and
  finally a venv it creates — then `exec`s it, so the client always sees one
  process. The first branch touches no network.
- `hooks/pre-edit.sh` and `hooks/on-bash-error.sh` moved from
  `examples/hooks/` into `hooks/` so the plugin references one directory.
  `install.sh` copies the whole directory as before.

### Fixed — benchmarks measured their own history
`Recall.search` bumps `recall_count` on every row it returns, and the scorer
adds `recall_boost = min(0.3, recall_count * 0.05)`. Spaced repetition is
wanted in normal use and fatal for measurement: successive runs against one
database scored 0.547 → 0.565 → 0.588 → 0.607 R@5 without a line of retrieval
code changing.

- `Recall.search(..., record_usage=False)` opts out. Both benchmark runners
  and `memory_explain_search` now pass it — a diagnostic must not change what
  it diagnoses. A clean run and a re-run are now byte-identical.
- `benchmarks/locomo_bench.py` had categories 2 and 3 **swapped** (it printed
  "multi-hop" over the temporal numbers and vice versa), and never computed
  overall MRR. Verified against `locomo10.json` and corrected.

### Added — BEAM (ICLR 2026)
- `benchmarks/beam_bench.py` scores retrieval on
  [BEAM](https://github.com/mohammadtavakoli78/BEAM) at the 100K / 500K / 1M
  scales across its ten memory abilities, using each probe's `source_chat_ids`
  as gold evidence. No LLM in the loop, so it is deterministic and free.
- Measured: **R@5 0.575** at 100K (5,732 messages, p50 17.7 ms) and **0.490**
  at 500K (38,058 messages, p50 58.5 ms). Contradiction resolution, knowledge
  update and temporal reasoning hold above 0.78 at both scales;
  `instruction_following` and `event_ordering` are near-zero at both, which is
  a statement about the primitive rather than the tuning.
- Ingest throughput halved between the two scales (25.6 → 10.8 msg/s), so the
  write path scales with store size. Not yet profiled; recorded rather than
  explained away.

### Added — LongMemEval measures the product now
- The runner reimplemented its own BM25 / RRF / MMR / CrossEncoder stack, so
  the published 96.2% described *an algorithm*, not this software. A new
  `--modes store` ingests each question's haystack into a real `Store` and
  queries `Recall.search` — the exact path an agent takes — and is now the
  default mode. The reference implementations stay available for ablations,
  with the docstring saying plainly that `full` and `store` are not the same
  claim.

### Fixed — the server was carrying ~450 MB it never used
Reported by **d.snezhinskiy**, who noticed an MCP server sitting at ~1.5 GB and
sent a patch.

- `chromadb` and `sentence_transformers` were imported at module scope in
  `src/server.py`. Both are *fallback* paths — when fastembed is healthy
  neither is used — and `sentence_transformers` pulls in torch. Availability is
  now decided by `importlib.util.find_spec` and the real import deferred to
  first use. Measured on this machine:

  | | before | after |
  |---|---:|---:|
  | `import server` | 558 MB | **116 MB** |
  | serving, steady state | 1367 MB | **909 MB** |

  torch is no longer loaded at all unless the fallback is actually reached.
- A failed fastembed init used to be one log line, after which the server
  quietly fell back to sentence-transformers and gained ~400 MB. It now names
  the model cache and the memory cost, because the usual cause is macOS purging
  the system tmp dir where fastembed caches models by default.
- New `TAM_MODEL_CACHE` opts into a durable cache location. It is **not** the
  default: moving the cache orphans models the user already downloaded, so
  every install would re-fetch ~500 MB once and offline test runs would fail.
  The contributor's patch pinned it unconditionally; this ships the diagnosis
  loudly and leaves the move to the people who need it.

> The patch's other half pinned `mcp[cli]<2`. That is the same bug this release
> fixes, and pinning would cap every user at the 1.x SDK forever, so it is not
> taken — but it independently confirmed the failure on a client machine, with
> the symptom `MCP error -32000: Connection closed`.

### Fixed — other
- **`tree-sitter-language-pack` was in no requirements file.** The README sold
  "AST codebase ingest, 9 languages" as a differentiator while every user's
  `ingest_codebase` silently degraded to whole-file chunks. Now a dependency
  (2 MB wheel).
- **The enrichment worker shared the Store's sqlite connection.**
  `check_same_thread=False` permits cross-thread *reads*; it does not make
  writes safe, because sqlite3's implicit transaction lives on the Connection.
  Concurrent writes interleaved into `cannot start a transaction within a
  transaction`, which is what made long ingests die. The worker now opens its
  own connection — which is what WAL mode is for.
- **Migration 028 failed on every fresh database**, not just some. Root cause
  reported by [@juicetin](https://github.com/juicetin) in the "separate
  migration observation" section of #12: `Store._migrate()` added the
  subagent-lineage columns *before* `_apply_sql_migrations()` ran
  `028_agent_lineage.sql`, so 028 always hit `duplicate column name: agent_id`.
  Because SQLite has no `ADD COLUMN IF NOT EXISTS` and `executescript` is
  all-or-nothing, it aborted before its `CREATE INDEX` statements and was never
  recorded as applied — so it retried on every startup, forever.
  - The schema change now has exactly one owner: the SQL migration.
    `_migrate()` no longer duplicates it, and a test fails if any column is
    ever added by both `Store._migrate()` and a SQL migration again.
  - The runner additionally replays a duplicate-column script statement by
    statement, so databases already wedged by the old behaviour recover
    instead of needing the operator to notice.
- **`ai_layer/verifier.py` hardcoded `~/.claude-memory/nli_calibration.json`**
  and stopped finding calibrations after the `~/.tam` migration. It now
  follows the resolved memory dir, with the legacy path as a fallback.
- **The graph node cache was invalidated on every write.** `_ensure_node`
  dropped the whole `graph_nodes` name cache each time it created a node, and
  `extract_and_link` dropped it again at the end — so the 60s TTL never
  applied and the next save re-read the table. Created nodes are now inserted
  into the cache incrementally; the TTL still picks up writes from other
  processes. A clean A/B over 4,000 saves measured **103.0 vs 99.9 saves/s** —
  about 3%, within noise at that size, since the re-read cost grows with the
  node table. It is reported as measured rather than as the fix for the ingest
  slowdown it was chasing: **that slowdown (25.6 → 10.8 msg/s between a 5.7k
  and a 38k record store) is not explained by this and remains open.** Both
  arms of the A/B degrade on the same curve.
- **`vocabularies/` and `filters/` were missing from the wheel and the Docker
  image.** Same shape as the migrations bug PR #12 fixed, and just as quiet:
  `src/` resolves them as `parent.parent / <dir>`, so in a git checkout
  everything worked, while every `pip` / `uvx` / `npx` / `docker` install fell
  back to an empty tag vocabulary and turned every `memory_save(filter=…)`
  into a no-op. An installed wheel now loads all 54 canonical topics and 11
  filter configs. `tests/test_wheel_contents.py` builds the wheel, checks each
  asset directory, and greps `src/` so a *new* sibling directory cannot go
  missing the same way.

### Fixed — test suite
The suite was red on a clean checkout: 21 failed, 9 errors.
- Eight modules stub the LLM seam but still had to pass `config.has_llm()`,
  which probes for a live Ollama in `auto` mode — so they failed on any
  machine without one. New `llm_enabled` fixture; the stub is what those tests
  measure, so no network call should ever be attempted.
- `benchmarks/data/` is gitignored, so the LoCoMo few-shot and NLI calibration
  tests failed instead of skipping on a fresh clone.
- Embedding tests asserted raw vectors; the production OpenAI path
  L2-normalises so cosine equals dot product.

**1881 passing, 0 failing.**

## [12.4.0] — 2026-05-26 — 100% functional through every install path

`npx connect`, `bash install.sh`, `docker run`, `docker compose up` — same
4 services, same MCP HTTP API, same dashboard. No path is missing pieces
anymore.

### Added — MCP Streamable HTTP transport
- `src/server.py` now supports `MCP_TRANSPORT=http` (also accepts
  `streamable-http`). Boots a uvicorn ASGI app exposing:
  - `POST/GET/DELETE /mcp` — MCP Streamable HTTP (spec 2025-03-26)
  - `GET /healthz` — DB-independent liveness probe
- Same `Server` instance powers both stdio and HTTP, so all 60+ MCP tools
  are reachable over the network — IDEs, sidecars, k8s pods can all
  connect without a stdio bridge.
- ``stateless=False`` keeps in-process session multiplexing; clean
  SIGTERM via uvicorn's `timeout_graceful_shutdown=5` matches Docker
  stop-grace.

### Added — scheduler service (cron-style)
- `docker/scheduler_daemon.py` — stdlib-only scheduler that runs
  `orphan-backfill` 4×/day and `check-updates` weekly inside the same
  image. Replaces the `mcuadros/ofelia` external container and its
  Docker-socket mount.
- npm wrapper 1.3.0 ships matching scheduler services on macOS
  (`StartCalendarInterval`) and Linux (`OnCalendar` systemd timers).

### Added — full single-image stack
- `tam-entrypoint` shim routes `docker run … mcp|dashboard|reflection|scheduler|all`
  through the supervisor.
- Default `TAM_SUPERVISOR_SERVICES=mcp,dashboard,reflection,scheduler`
  → `docker run -p 3737:3737 -p 37737:37737 -v vol:/data ghcr.io/.../total-agent-memory`
  brings the whole product up with one command.
- `EXPOSE 3737 37737`; HEALTHCHECK probes both `/healthz` endpoints.
- ML caches pinned to the data volume (`HF_HOME`, `TRANSFORMERS_CACHE`,
  `TORCHINDUCTOR_CACHE_DIR`) — prevents torch._dynamo from crashing on
  containers with tmpfs `/tmp` < 100 MB.
- Default `OLLAMA_URL=http://host.docker.internal:11434` — single-image
  finds host Ollama out of the box on Docker Desktop / Windows / WSL2.
  Graceful degradation to FTS+embeddings if Ollama unreachable.

### Added (v12.3 carry-over — never tagged separately)
- `docker/tam_supervisor.py` — stdlib supervisor (prefix logging,
  exponential-backoff restarts, signal propagation).
- `/healthz` endpoint on the dashboard.
- Auto-initialise `memory.db` on first boot inside the container.
- `launchagents/com.total-agent-memory.dashboard.plist` (installed by
  `install.sh` on macOS).
- `.github/workflows/docker.yml` — multi-arch (amd64+arm64) build+push
  to ghcr on every git tag, with `/healthz` + `/api/stats` + docker
  healthcheck smoke before publish.
- `.github/workflows/smoke.yml` — install-matrix smoke on every PR.
- `tests/smoke/install-docker-pull.sh`, `tests/smoke/install-sh-macos.sh`.
- Plist placeholders unified to `__INSTALL_DIR__` / `__MEMORY_DIR__` /
  `__HOME__`. Old plists hardcoded `__HOME__/claude-memory-server/...`,
  breaking clones with any other directory name.

### Changed
- `docker-compose.yml` `mcp` service is back (was removed in v12.3 when
  HTTP transport didn't exist). Same image as `dashboard`/`reflection`,
  `command: ["mcp"]`.
- `scheduler` compose service now runs in the main image
  (`command: ["scheduler"]`) instead of `mcuadros/ofelia` — no more
  Docker socket mount in the stack.
- `install.sh` Step 5 (old `dashboard-service.sh install` call) removed
  — it created a `com.claude-total-memory.dashboard` plist that fought
  the canonical `com.total-agent-memory.dashboard` plist for port 37737.

### Fixed
- `bash install.sh` under any clone name now writes valid plist/unit
  paths instead of references to a non-existent `~/claude-memory-server/`.
- `bash install.sh --uninstall` cleans up the new dashboard plist too.

## [12.3.0] — never tagged — folded into 12.4.0

The 1.1.0 npm wrapper, `docker pull`, and certain `git clone` layouts all
left users with a non-functional web dashboard despite the module being
present in the package. v12.3 closes those gaps end-to-end.

### Added
- `docker/tam_supervisor.py` — single-container supervisor (~150 LOC,
  stdlib only) that runs dashboard + reflection together with prefix
  logging, exponential-backoff restarts, and clean SIGTERM propagation.
  Makes `docker run ghcr.io/.../total-agent-memory` give a working
  dashboard with one command.
- `tam-entrypoint` shim in the image — routes `docker run … mcp|dashboard|reflection|all`
  through the supervisor so compose services and single-image use share
  a single CMD shape.
- `/healthz` endpoint on the dashboard — DB-independent liveness probe
  suitable for Docker `HEALTHCHECK` and Kubernetes `livenessProbe`. Fresh
  volumes no longer report unhealthy until the first save.
- Auto-initialise `memory.db` on first boot in the supervisor.
- `launchagents/com.total-agent-memory.dashboard.plist` — keeps the
  dashboard running and restarts on crash. Installed automatically by
  `install.sh` on macOS.
- `.github/workflows/docker.yml` — build + push multi-arch
  (`amd64`, `arm64`) ghcr images on every git tag, with `/healthz` +
  `/api/stats` + Docker healthcheck smoke before publish. Tags produced:
  `:X.Y.Z`, `:X.Y`, `:X`, `:latest`.
- `.github/workflows/smoke.yml` — runs install-matrix smoke on every PR.
- `tests/smoke/install-docker-pull.sh` + `tests/smoke/install-sh-macos.sh`
  — black-box end-to-end install verifiers.

### Changed
- `install.sh` Step 5 (the old `dashboard-service.sh install` call)
  removed — it created a `com.claude-total-memory.dashboard` plist that
  fought the canonical `com.total-agent-memory.dashboard` plist for
  port 37737. One always ended up in crash-loop.
- Plist placeholders unified to `__INSTALL_DIR__` / `__MEMORY_DIR__` /
  `__HOME__`. Old plists hardcoded `__HOME__/claude-memory-server/...`,
  breaking any clone whose directory name wasn't `claude-memory-server`
  — the common case after the rebrand.
- Dockerfile `CMD` switched to the supervisor; `EXPOSE` narrowed to
  `37737` (the MCP HTTP port 3737 was never functional — see Known Issues).
- `docker-compose.yml` `mcp` service removed — `MCP_TRANSPORT=http` was
  silently ignored by `src/server.py` (stdio-only). For MCP, use
  `docker/run-mcp.sh` (stdio bridge). HTTP transport lands in v12.4.

### Fixed
- `bash install.sh` under any clone name now writes valid plist/unit
  paths instead of references to a non-existent `~/claude-memory-server/`.
- `bash install.sh --uninstall` cleans up the new dashboard plist too.
- New regression test `test_launchagents_substitute_install_dir_and_memory_dir`
  guards against placeholder leakage.

### Known issues
- MCP server is still stdio-only inside the Docker image — connect via
  `docker/run-mcp.sh`. Streamable HTTP transport is tracked for v12.4.

## [12.2.0] — 2026-05-24 — v11 W3 dispatch fix + Codex env alignment

Bugfix release that restores four previously-broken v11 W3 MCP tools and
aligns the Codex installer with the `.tam` memory layout.

### Fixed

- **`server._do` v11 W3 dispatch** — `memory_recall_iterative`,
  `memory_temporal_query`, `memory_entity_resolve`, and
  `memory_consolidate_status` forwarded the symbol `args`, which is not
  in scope inside `_do(name, a)`. The resulting `NameError` was caught
  by `call_tool`'s blanket `except` clause and returned to the client as
  `"Error: name 'args' is not defined"`, silently disabling all four
  tools on v12.0.0–v12.1.0. They now receive the correct per-call args.

### Added

- **`tests/test_v11_dispatch_args.py`** — regression coverage that
  exercises the four v11 W3 branches through `server.call_tool()` with
  monkey-patched handlers, asserting the args object reaches each
  handler unchanged.

### Changed

- **Codex installer env (`install.sh --ide codex`)** — writes the
  canonical `TAM_MEMORY_DIR`, retains `CLAUDE_MEMORY_DIR` as a
  compatibility alias, and defaults `MEMORY_MODE=fast` so freshly
  installed Codex clients land on the same hot-path config as
  Claude Code.
- **`hooks/lib/common.sh` and `hooks/user-prompt-submit.sh`** — resolve
  the memory dir via `TAM_MEMORY_DIR` → `CLAUDE_MEMORY_DIR` → `~/.tam`
  fallback. `hook_project_name` now honours the `MEMORY_PROJECT` env so
  hooks reporting from a wrapper project (Codex, sub-shells) can pin
  the project name explicitly.
- **`tests/test_install_ide_flag.py`** — asserts the new Codex env
  entries and sandboxes installer runs from the real user `systemctl`
  / `launchctl` / `XDG_CONFIG_HOME` via a fake `systemctl` shim, so
  contributors no longer touch their host services when running the
  suite.

### Compatibility

- No schema or wire-protocol changes. Pure server-side fix plus
  installer hardening.
- Existing installs continue to work — `CLAUDE_MEMORY_DIR` still
  resolves through the alias and `~/.claude-memory/` symlinking
  remains intact.

### Notes

- Codex hook registration was intentionally **not** included. The
  shell-style `<system-reminder>` payloads the existing hooks emit
  don't match Codex's structured JSON hook protocol; a Codex-native
  hook bridge will land in a follow-up once that protocol is
  exercised end-to-end.

## [12.1.0] — 2026-05-20 — Claude Code v2.1.145 subagent lineage

Adds first-class support for the **agent lineage** signals that Claude Code
v2.1.139+ began emitting (`x-claude-code-agent-id` / `x-claude-code-parent-agent-id`
HTTP headers; same as `agent_id` / `parent_agent_id` attributes on
`claude_code.tool` and `claude_code.llm_request` OTEL spans). Recall can now
answer "what did subagent X (dispatched from parent Y) produce".

### Added

- **Migration `028_agent_lineage.sql`** — adds nullable `agent_id` and
  `parent_agent_id` columns to `knowledge` plus two partial indexes
  (`WHERE … IS NOT NULL`) so lineage filters cost the same as a regular
  recall.
- **`memory_save` / `memory_save_fast` MCP tool inputs** — two new optional
  fields:
  - `agent_id` — subagent ID as carried by Claude Code (header / OTEL attribute).
  - `parent_agent_id` — the dispatching agent / parent span.
  Schemas are reflected by `/mcp` after a reconnect; pre-v2.1.145 callers
  see no behaviour change.
- **`save_knowledge()` Python signature** gains the same two keyword
  arguments. They flow into the outbox payload so a mid-save crash replay
  re-applies the lineage instead of dropping it.
- **`extract_transcript.py` ingests lineage from session `.jsonl`s.**
  - Reads `agent_id` / `agentId` and `parent_agent_id` / `parentAgentId`
    when Claude Code populates them (forward-compatible — naming is not
    yet contracted upstream).
  - Falls back to the **`isSidechain=true`** signal as a proxy: any session
    with at least one sidechain entry gets `agent_id = "session-<id>"` on
    its auto-extracted rows and a `has-subagent-work` tag, so recall can
    surface "sessions where a subagent ran" even before Anthropic writes
    the explicit fields.
  - Graceful schema check (`PRAGMA table_info`) keeps the script working
    on installs where migration 028 has not been applied.
- **Automatic `spawned_by` KG fact.** When `_save_knowledge_impl()` is
  called with both `agent_id` and `parent_agent_id`, it now records
  `TemporalKG.add_fact(agent, "spawned_by", parent, source="agent-lineage",
  invalidate_previous=False)`. Idempotent on (subject, predicate, object).
  `kg_at(timestamp)` and `kg_timeline()` can now reconstruct the subagent
  lineage tree at any past moment.

### Changed

- `INSERT INTO knowledge` in `server.py` now writes 15 → 17 columns,
  carrying the lineage on every save. The base `CREATE TABLE IF NOT EXISTS`
  is unchanged — the columns appear via the idempotent `_migrate()` step
  so fresh installs and upgrades converge to the same schema.

### Compatibility

- Both columns are nullable. Existing rows, callers that don't pass the
  ids, outbox replays of old intents, and consumers of `memory_recall`
  are unaffected.
- The new MCP tool fields are optional. Pre-v2.1.145 Claude Code does not
  pass them; behaviour matches v12.0.x.
- A reconnect of the MCP `memory` server is required for clients to see
  the updated `inputSchema` for `memory_save` / `memory_save_fast`.

### Companion changes outside this repo

- `~/.claude/hooks/on-stop.sh` — reads `background_tasks` and `session_crons`
  from the Stop hook payload (v2.1.145+) and appends a section to the
  recovery file. Also fixes a stale path: `~/claude-memory-server` →
  `~/total-agent-memory/.venv/bin/python`.
- `claude-statusline 1.2.0` — picks up the new `.github.repo` /
  `.github.pr.*` fields Claude Code injects in v2.1.145+ and renders a PR
  badge on line 2.

## [11.1.0] — 2026-05-14 — Graph dedup + proactive save nudges

Two production bug-fixes from a client report (2026-05-14): "graph
accumulates orphan nodes & duplicates" and "Claude ~never calls
`memory_save` on its own (~1 of 30 sessions)". Both fixed end-to-end.

### Fixed — orphan + duplicate `graph_nodes` (bug #1)

Root causes (4): no UNIQUE on `(name, type)`, case-sensitive lookup in
`add_node`, different extractors classifying the same entity under
different types (e.g. `vue/concept` vs `vue/technology` → two rows),
non-atomic `add_node`+`add_edge` (failed edge left orphan nodes).

- **Migration `026_graph_nodes_dedup.sql`** — adds `name_norm`
  (case-folded), backfill, triggers that keep it in sync, non-UNIQUE
  indexes (UNIQUE installed only post-cleanup).
- **`src/graph/store.py`** — `add_node` rewritten as case-insensitive
  UPSERT with type-collision detection (existing node of any type for
  the same `name_norm` is reused instead of forking). Race-safe via
  IntegrityError catch + re-select.
- **`GraphStore.link_pair(src_name, src_type, dst_name, dst_type, rel)`**
  — atomic create-or-reuse two nodes + edge. On edge failure deletes
  only the nodes this call freshly created. Eliminates the orphan
  pattern at the source.
- **`src/tools/merge_duplicate_nodes.py`** — one-shot cleanup tool.
  `--dry-run` default, `--apply` to mutate, `--case-only` to skip
  cross-type merges, `--add-unique` installs the final
  `UNIQUE(name_norm, type)` constraint. Repoints edges and
  knowledge_node links, dedupes collisions, drops self-loops.
- **Test coverage**: +24 tests across `test_graph.py` and the new
  `test_merge_duplicate_nodes.py`. Run on a real 8304-node production
  DB: merged 102 duplicates + 1472 stale edges, DB 118.5 → 108.8 MB.

### Fixed — model never calls `memory_save` on its own (bug #2)

Smaller models (Sonnet, Haiku) skip the priority-10 "save what matters"
rule unless reminded mid-session. The SessionStart hint fades long
before meaningful work happens. v11.1 adds in-session nudges that
Claude sees as system context on the next turn.

- **`hooks/lib/memory-nudge.sh`** — per-session counter in
  `~/.claude-memory/state/nudge-<session>.json` tracking
  `writes / edits / bashes / memory_saves`.
- **`hooks/post-tool-use.sh`** — was opt-in capture only; now always
  emits a stdout nudge when significant-writes-without-saves crosses a
  threshold. Soft nudge at 3 edits, hard at 7. Throttled to avoid spam;
  hard escalation bypasses throttle once. A `memory_save` releases
  pressure for the next 2×STEP edits.
- **`hooks/on-stop.sh`** — final `MEMORY_FINAL_WARNING` when the
  session is about to close with 0 saves but ≥3 edits.
- **`hooks/post-tool-use.ps1` + `on-stop.ps1`** — feature-parity on
  Windows.
- **New priority-10 behavioural rule**: "`MEMORY_NUDGE` in stdout is
  an immediate action signal, not information — call `memory_save`
  before the next significant edit". Installed via
  `self_rules(add_manual=...)`.
- **Tunables**: `MEMORY_NUDGE_DISABLE=1` to silence;
  `MEMORY_NUDGE_SOFT` / `_HARD` / `_STEP` to retune.
- **Test coverage**: 12 new tests (`test_memory_nudge_hook.py`)
  covering counters, threshold transitions, save-silences-pressure,
  hard escalation, summary emission.

### Operational notes

- Migration 026 is applied automatically by `_apply_sql_migrations()`
  on first MCP-server boot after upgrade — no manual SQL needed.
- `merge_duplicate_nodes.py` refuses to install the UNIQUE index if
  any duplicate row still exists — safe-by-default ordering.
- Nudge state directory is auto-pruned by `on-stop` (files >7 days).
- Hooks remain non-blocking; the inline Python reads only the cached
  temp file (no DB, no network, <50 ms typical).

### Migration

No manual steps required for users on v11.0:

```bash
git pull && pip install -e .   # or your usual upgrade path
# Next MCP-server start applies migration 026 automatically.
# (optional) clean up duplicates accumulated before the upgrade:
.venv/bin/python src/tools/merge_duplicate_nodes.py --dry-run
.venv/bin/python src/tools/merge_duplicate_nodes.py --apply --add-unique
```

## [11.0.0] — 2026-04-28 — Production Memory Engine + LoCoMo SOTA

**Headline:** LoCoMo benchmark **0.705 overall** (1986 QA, gpt-4o gen + gpt-4o-mini judge). Position **#5** on the public leaderboard — above Mem0 (0.669) and v9-ensemble3 internal best (0.696). Temporal **0.654** (+39pp vs v9 paper-method baseline). R@5 (no-adv) **0.673**. See `docs/v11/RELEASE-FINAL-2026-04-28.md` for full breakdown.

### Architecture (Wave 1 + Wave 2, ~5k LOC, ~460 new tests)

**Memory Core** (`src/memory_core/*`) — deterministic facades, no LLM in hot path:

- `episodes/{schema,extractor,retriever}.py` — Episode layer (when, who, where, what, why, outcome). Wired as **Tier 6** in `Recall._search_impl()` RRF fusion.
- `temporal/{allen,normalizer,arithmetic}.py` — Allen's 13 interval relations, 65 composition pairs, en+ru date normalisation, calendar-aware arithmetic.
- `entity_resolver.py` — Cross-session canonical entity resolution (NFKD unicode, multilingual pronoun guard, embedding fallback).
- `idk_router.py` — Threshold-based answer routing (Protocol-typed, layer-wall-safe).
- `negative_retrieval.py` — Active contradiction search for adversarial questions.
- `calibration.py` + `answer_router.py` — Platt-scaled retrieval scores, per-category routing, ECE 0.046 on validation fixture.

**AI Layer** (`src/ai_layer/*`) — LLM-touching modules, hot-path forbidden:

- `iterative_retriever.py` — IRCoT-style iterative retrieval on top of `query_rewriter`.
- `answerability.py` — Permissive answerability classifier (Haiku, JSON contract).
- `verifier.py` — Local NLI verifier (mDeBERTa-v3-base-xnli-multilingual, 270MB, p95 11.9ms on MPS). **Calibrated thresholds** (`p_entail=0.65, p_contradict=0.40`) loaded from `~/.claude-memory/nli_calibration.json`.

**Workers** (`src/workers/*`) — out-of-band consolidation:

- `consolidation_daemon.py` + `bin/consolidation-daemon` + macOS launchd plist — 24/7 idle-project consolidation. Picks oldest idle project, advisory TTL lock, pauses when project becomes active. Per-project budget 600s.

### Bench (`benchmarks/locomo_bench_llm.py`)

- `--v11-pipeline` — Post-processor: NLI veto + answerability + calibrated routing.
- `--v11-skip-nli` — Skip NLI model load (270MB).
- `--ce-rerank` now supports `V9_RERANKER_BACKEND=bge-v2-m3` (BAAI/bge-reranker-v2-m3, multilingual).
- Recall metric fix — case-insensitive dialog-id matching (`canonical_tags` lowercases tags; gold evidence keeps case).

### Existing v11 hot-path features (carried over from earlier 11.0 work)

- 4 modes: `ultrafast` / `fast` / `balanced` / `deep`. Default `fast` — zero LLM/network in save/search/recall hot path.
- Silent Ollama fallback in `Store.embed` GATED — `MEMORY_ALLOW_OLLAMA_IN_HOT_PATH=true` to re-enable.
- Multi-embedding-space contract: every vector row records `embedding_provider/model/dimension/space/content_type/language`. Spaces: `text` / `code` / `log` / `config`.

### New MCP tools

- `memory_recall_iterative` — IRCoT search.
- `memory_temporal_query` — Allen relations + duration arithmetic + date normalisation.
- `memory_entity_resolve` — Cross-session canonical lookup.
- `memory_consolidate_status` — Daemon state + recent activity.
- (carried) `memory_save_fast`, `memory_search_fast`, `memory_explain_search`, `memory_warmup`, `memory_perf_report`, `memory_rebuild_fts`, `memory_rebuild_embeddings`, `memory_eval_*`.

### Embeddings

- OpenAI `text-embedding-3-large` (3072d, l2-normalised) supported via `MEMORY_EMBED_PROVIDER=openai MEMORY_EMBED_MODEL=text-embedding-3-large`. `scripts/reembed.py --provider openai --model text-embedding-3-large` re-encodes with cost preview.

### Few-shot

- `benchmarks/data/locomo_few_shot_v3.json` — 75 LoCoMo-derived pairs (15×5 categories), deterministic seed=42, md5=`29258307fdfb6a33c2668298a515fe65`.

### Migrations

- `023_episodes.sql` — `episodes_v11` + `episode_facts` + FTS mirror.
- `024_entity_aliases.sql` — `canonical_entities` + `entity_aliases` (UNIQUE on `(project, type, name_norm)`).
- `025_consolidation_state.sql` — `project_activity` + `consolidation_state`.

All idempotent.

### Layer separation

`tests/test_v11_layer_separation.py` — AST walks `src/memory_core/`; fails on any `import ai_layer`. Communication is via Protocol structural typing and dataclass parameters.

### Configuration knobs

| Env var | Default | Purpose |
|---|---|---|
| `MEMORY_MODE` | `fast` | `ultrafast`/`fast`/`balanced`/`deep`. |
| `MEMORY_EPISODE_TIER` | `true` | Enable episode-tier in Recall RRF. |
| `MEMORY_EMBED_PROVIDER` | `fastembed` | Set to `openai` for text-embedding-3-large. |
| `MEMORY_EMBED_MODEL` | depends | OpenAI model name when provider is OpenAI. |
| `V9_RERANKER_BACKEND` | `ce-marco` | `bge-v2-m3` selects multilingual BGE reranker. |
| `V11_PIPELINE` | `0` | When `1`, bench applies post-processor (NLI veto + router). |
| `V11_SKIP_NLI` | `0` | When `1`, skip NLI verifier in v11 pipeline. |

### Known regressions vs v9 paper-method baseline

- **multi-hop −10.4pp** in the unenhanced v11 stack (W4 v2). Disappears with the D146 (text-embedding-3-large + BGE + few-shot v3) stack — multi-hop returns to **+15.7pp** vs v9-ensemble3.
- **single-hop −5pp** vs v9-ensemble3 with `--query-rewrite` enabled (qrw favours retrieval over precision on single-hop). Trade-off with overall +0.9pp gain.

### Hot-path performance (warm, in-memory SQLite, M-series)

| metric              |   p50 |   p95 |   p99 |
|---------------------|------:|------:|------:|
| save_fast           |  6.2  |  8.9  | 11.4  |
| save_fast (cached)  |  0.3  |  0.4  |  1.4  |
| search_fast         |  3.4  |  4.7  |  6.0  |
| cached_search       |  3.1  |  3.4  |  3.6  |

`llm_calls = 0`, `network_calls = 0` in the deterministic hot path.

**Migration**: see [`docs/v11/MIGRATION-FROM-V10.md`](docs/v11/MIGRATION-FROM-V10.md). Architecture audit: [`docs/v11/audit.md`](docs/v11/audit.md). LoCoMo final report: [`docs/v11/RELEASE-FINAL-2026-04-28.md`](docs/v11/RELEASE-FINAL-2026-04-28.md). NLI calibration: [`benchmarks/results/nli-calibration-report.md`](benchmarks/results/nli-calibration-report.md). Failure analysis: [`benchmarks/results/baseline-failure-analysis.md`](benchmarks/results/baseline-failure-analysis.md).

## [10.5.0] — 2026-04-27

Universal skill, 9-IDE installer, cross-platform hardening, sub-agent protocol, and a fresh latency benchmark proving the v10.1 async worker delivers an **80× p95 reduction** on `memory_save`.

### Added
- **Universal `memory-protocol` skill** (`skills/memory-protocol/`) — single SKILL.md (`v10.5.0`) + 4 references (`tool-cheatsheet.md` covering all 60+ MCP tools, `workflow-recipes.md` with 15 production-tested recipes, `hooks-explained.md`, `ide-setup.md`, `subagent-protocol.md`) + 4 templates (`claude-code-settings.json`, `codex-config.toml`, `cursor-rules.mdc`, `cline-rules.md`, `codex-AGENTS-block.md`). Same canonical content for every IDE; only the wiring differs.
- **`install.sh --ide` extended from 5 to 9 IDEs**: claude-code, codex, cursor, **cline**, **continue**, **aider**, **windsurf**, gemini-cli, opencode. New helpers: `register_mcp_cline`, `register_mcp_continue`, `register_mcp_aider`, `register_mcp_windsurf`, plus `_json_merge_mcp_nested` for the dotted-key case (`cline.mcpServers`).
- **Auto-install of `skills/memory-protocol/`** on every `install.sh --ide <X>` run that targets an IDE with a skill API (claude-code / codex / opencode); IDEs without a skill API get a rules-file copy via their respective register function.
- **Sub-agent memory protocol** — universal header for any sub-agent (`php-pro`, `golang-pro`, `vue-expert`, `code-reviewer`, etc.). Documented in `skills/memory-protocol/references/subagent-protocol.md`.
- **`benchmarks/v10_5_latency.py`** — apples-to-apples sync vs async micro-bench. `--rounds N`, `--with-llm`, JSON output to `benchmarks/results/v10_5_latency.json`. Markdown report `benchmarks/v10_5_results.md`.

### Performance
- `memory_save` p95 with LLM stages on: **2150 ms (sync) → 27 ms (async)**, **80× reduction**.
- `memory_save` p99: **2179 ms → 27 ms**.
- `memory_save` mean: **348 ms → 23 ms** (15×).
- `memory_recall` p50 steady state: **3-5 ms** in both modes.
- On WSL2 with slow Ollama the same shape holds — sync p95 of 30-40 s becomes async p95 of ~300-1000 ms.

### Fixed
- **`update.sh` bash 3.2 incompatibility** — `${var,,}` (lowercase parameter expansion) replaced with `tr '[:upper:]' '[:lower:]'`. macOS default shell now parses cleanly.
- **Cross-platform shellcheck pass** — all production `.sh` scripts (`install*.sh`, `update.sh`, `setup.sh`, `hooks/*.sh`, `ollama/*.sh`) syntax-check under `/bin/bash 3.2.x` (macOS), `bash 5.x` (Linux / WSL2). Zero blocker findings; only style/info notes remain.

### Changed
- README badges: `version 10.5.0`, `tests 1153 passing`, `IDEs 9 supported`.
- README: new **IDE matrix** table after Install, updated **Performance Tuning** numbers, new **v10.5 Roadmap** entry.
- `install.sh` USAGE: documents all 9 IDEs.
- `pyproject.toml`: bumped to `10.5.0`.

### Test suite
- 1153 passing (no new tests this version — the additions are docs / installer / bench code that is exercised by smoke runs in the bench tool).

## [10.1.0] — 2026-04-27

Inbox/outbox async pipeline, two production bugfixes, and dashboard observability for the worker. Backwards compatible: all new behaviour is opt-in.

### Added
- **Async enrichment worker** (`src/enrichment_worker.py`, migration `020_async_enrichment.sql`). Opt-in via `MEMORY_ASYNC_ENRICHMENT=true`. Moves the heavy LLM-bound stages of `save_knowledge` (quality gate, entity-dedup audit, contradiction detector, episodic event linking, wiki refresh) to a background daemon thread that consumes `enrichment_queue`. Drops `memory_save` p99 latency from ~2.5 s to ~460 ms on macOS, and from 30–40 s to ~300–1000 ms on WSL2 with a slow Ollama. Soft-drop semantic: a `quality_gate` `drop` verdict marks the row `status='quality_dropped'` after the INSERT (instead of blocking it).
- **Stale-processing recovery** in `enrichment_worker`. Rows stuck in `status='processing'` longer than `MEMORY_ENRICH_STALE_AFTER_SEC` (default 60 s) flip back to `pending` automatically. Covers worker process kills mid-stage.
- **Dashboard panel `⚡ v10.1 enrichment worker`** — depth, throughput per minute, p50/p95 ms per task, oldest pending age (color-coded by SLO band), and last 5 failures with their error message. New endpoint `GET /api/v10/enrichment-queue`.
- **5 new env knobs** for the worker: `MEMORY_ASYNC_ENRICHMENT`, `MEMORY_ENRICH_TICK_SEC`, `MEMORY_ENRICH_BATCH`, `MEMORY_ENRICH_MAX_ATTEMPTS`, `MEMORY_ENRICH_STALE_AFTER_SEC`.
- **`Performance tuning` README section** with sync-vs-async benchmark and tuning guidance for slow-LLM hosts.
- **17 regression tests**: 15 for the worker (enqueue/claim/idempotency/retry/soft-drop/daemon/stale-recovery), 4 for `_binary_search` edge cases, 1 for coref RU→EN guard.

### Fixed
- **`Store._binary_search` `ValueError: kth(=N) out of bounds (N)` on small candidate pools.** `np.argpartition` requires `kth STRICTLY < N`; tiny test projects (≤ 50 active embeddings) used to silently break `contradiction_log` because the save-path swallowed the exception in a generic `except`. Hot path now takes the whole pool when `n_candidates >= len(pool)`.
- **`coref_resolver` translating Russian → English.** `qwen2.5-coder:7b` (and Llama 3.x) interpreted the rewrite prompt as an instruction to switch language. Prompt now pins output language explicitly (`Do NOT translate. Do NOT switch language even partially.`) and tests assert the guard remains in the prompt.
- **`embed_provider` test fixtures** rejecting the `context=` kwarg passed by certifi-aware production callers (Python 3.13 macOS). Fixture `_capture_urlopen.fake()` now accepts forward-compatible kwargs.

### Changed
- `sqlite3.connect(check_same_thread=False)` for the `Store` connection so the enrichment worker thread can share it. Safe under WAL + busy_timeout=5000.
- Test suite: 1124 → 1153 passing (+29).
- Bumped version to `10.1.0`; `pyproject.toml` aligned.

## [10.0.0] — 2026-04-27

Beever-Atlas-inspired feature wave: 10 new pipeline stages, 5 new migrations, 153 new tests.

### Added
- **Quality gate (Beever 6-Month Test)** — synchronous LLM scorer (specificity / actionability / verifiability) with threshold 0.5, fail-open. Blocks low-signal records before INSERT. `MEMORY_QUALITY_GATE_ENABLED`, `MEMORY_QUALITY_THRESHOLD`.
- **Importance boost** — `critical / high / medium / low` field on knowledge rows; multiplies recall RRF score (×1.5 / ×1.2 / ×1.0 / ×0.8). Reserved for migration-blocking decisions and security incidents.
- **Canonical tag vocabulary** — 86 topics in `vocabularies/canonical_topics.txt`, normalised on save via embedding cosine + Levenshtein. Aliases under length 3 ignored.
- **Coref resolver** (opt-in via `MEMORY_COREF_ENABLED=true` or `coref=True` per save). Expands pronouns/deictics using last 20 records from the same session before INSERT.
- **Auto contradiction detector** — same-type/project semantic neighbours scored by LLM; `≥0.8` confidence → automatic supersession, `0.5–0.8` → flagged. Audit trail in `contradiction_log`.
- **Outbox / write-intent journal** — every `save_knowledge` call writes an intent row before any side-effects, allowing crash recovery on restart. `_reconcile_outbox_at_startup` replays committed intents.
- **Embedding-based entity dedup** — non-canonical tags get a second-chance lookup against active `graph_nodes` via cosine ≥ 0.85. Audit log in `entity_dedup_log`.
- **Episodic save events** — every save spawns an `event` node in `graph_nodes` with `MENTIONED_IN` edges to entity nodes. Enables queries like "show me saves where Postgres and Bob were mentioned together".
- **Smart query router** — bilingual (EN+RU) heuristic classifier on `memory_recall`; relational queries (wh-words / connectors / multiple entities) get a graph_search pass with a +1.3× RRF boost.
- **Per-project Markdown wiki digest** (`memory_wiki_generate(project)`) — Top Decisions / Active Solutions / Conventions / Recent Changes. Files land in `<MEMORY_DIR>/wikis/<project>.md`.

### Migrations
- `015_quality_importance.sql`, `016_contradictions.sql`, `017_outbox.sql`, `018_entity_dedup.sql`, `019_episodic_links.sql` — applied automatically by `_apply_sql_migrations` at startup.

### Changed
- `save_knowledge` returns a 5-tuple `(rid, was_dedup, was_redacted, private_sections, quality_meta)` — was 4-tuple in v9.

## [8.0.0] — 2026-04-19

Major feature wave: task workflow phases, structured decisions, cloud providers, activeContext live-doc, and many other quality-of-life improvements.

### Added
- **Cloud LLM providers** — `MEMORY_LLM_PROVIDER=openai|anthropic|ollama`. OpenAI-compat for OpenRouter, Together, Groq, DeepSeek, LM Studio, llama.cpp server. Per-phase routing: `MEMORY_TRIPLE_PROVIDER`, `MEMORY_ENRICH_PROVIDER`, `MEMORY_REPR_PROVIDER` with independent models.
- **Cloud embeddings** — `MEMORY_EMBED_PROVIDER=fastembed|openai|cohere` with dimension-mismatch safety gate that blocks catastrophic re-embed accidents.
- **`<private>...</private>` inline-tag** for automatic secret redaction in `save_knowledge`.
- **Session auto-compression** — `session_end(auto_compress=True)` generates summary/next_steps/pitfalls via LLM provider.
- **Progressive disclosure 3-layer workflow** — `memory_recall(mode="index")` returns compact ID+title+score, `memory_get(ids=[...])` batched full-content fetch. ~83% token saving vs default full recall.
- **Task complexity classifier** — `classify_task(description)` returns {level: 1-4, suggested_phases, estimated_tokens}.
- **Task phases state machine** — `task_create` / `phase_transition` / `task_phases_list` / `complete_task` with L1-L4 routing (van→plan→creative→build→reflect→archive). Migration 012_task_phases.sql.
- **Structured `save_decision`** — title + options + criteria_matrix + selected + rationale + discarded + auto multi-representation indexing. `memory_recall(decisions_only=True)` filter.
- **`activeContext.md` live-doc** — Obsidian markdown projection of session_init/end for human-readable session state. `MEMORY_ACTIVECONTEXT_VAULT` env override.
- **Phase-scoped rules** — `self_rules_context(project, phase="build")` with `phase:X` tag filter (zero migration). `rule_set_phase` MCP tool.
- **HTTP citation endpoints** — `/api/knowledge/{id}`, `/api/session/{id}` with related-graph expansion. HTML views at `/knowledge/{id}` and `/session/{id}`.
- **UserPromptSubmit hook** — captures user prompts into `intents` table (migration 013). `save_intent` / `list_intents` / `search_intents` MCP tools.
- **PostToolUse capture hook** — opt-in (`MEMORY_POST_TOOL_CAPTURE=1`) tool observation capture via deferred reflection queue.
- **Unified installer** — `install.sh --ide {claude-code|cursor|gemini-cli|opencode|codex}`. `install-codex.sh` is now a 3-line backward-compat shim.
- **15+ new MCP tools** — total count now 60+.
- **9 new src modules** — privacy_filter, llm_provider, embed_provider, task_classifier, task_phases, decisions, active_context, intents, recall_modes.
- **3 new migrations** — 011 privacy_counters, 012 task_phases, 013 intents.
- **2 new hooks** — user-prompt-submit.sh, post-tool-use.sh.
- **Donation link** updated to PayPal.Me/vbcherepanov.

### Fixed
- **Regression restore** — commit 2976ca1 ("docs(v7.0): sync README, install.sh, src refresh", 2026-04-17) accidentally reverted merged PR #5 (timeout config functions). Restored `get_triple_timeout_sec`, `get_enrich_timeout_sec`, `get_repr_timeout_sec`, `get_triple_max_predict` and related callers.
- **`has_llm()` phase-aware** — now consults provider.available() for cloud providers instead of only probing local Ollama. Previously `MEMORY_LLM_PROVIDER=openai` with Ollama offline would early-return False from all callers.

### Changed
- **Test suite** — 501 → 749 passing tests (+248).
- **Dashboard bind** — already on 127.0.0.1 (no change, maintaining security baseline).

## [7.0.0] — 2026-04-15

See git history for previous releases.
