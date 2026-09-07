# total-agent-memory

<!-- mcp-name: io.github.vbcherepanov/total-agent-memory -->

> **The only memory layer that learns _how_ you work — not just _what_ you said.**
> Persistent, local memory for AI coding agents: Claude Code, Codex CLI, Cursor, any MCP client.
> Temporal knowledge graph · procedural memory · AST codebase ingest · cross-project analogy · 3D WebGL visualization.

[![Version](https://img.shields.io/badge/version-13.0.4-8ad.svg)](https://pypi.org/project/total-agent-memory/)
[![Tests](https://img.shields.io/badge/tests-1881%20passing-4a9.svg)]()
[![IDEs](https://img.shields.io/badge/IDEs-9%20supported-4a9.svg)]()
[![LongMemEval R@5](https://img.shields.io/badge/LongMemEval%20R@5-95.1%25-4a9.svg)](evals/longmemeval-2026-08-27-v13-store.json)
[![LoCoMo R@5](https://img.shields.io/badge/LoCoMo%20R@5-0.607-4a9.svg)](benchmarks/results/v13-locomo-retrieval.json)
[![BEAM R@5](https://img.shields.io/badge/BEAM%201M%20R@5-0.448-4a9.svg)](benchmarks/results/v13-beam-1M.json)
[![vs Supermemory](https://img.shields.io/badge/vs%20Supermemory-%2B9.7pp-4a9.svg)](docs/vs-competitors.md)
[![p50 latency](https://img.shields.io/badge/p50%20warm-0.065ms-4a9.svg)](evals/results-2026-04-17.json)
[![Local-First](https://img.shields.io/badge/100%25-local-4a9.svg)]()
[![License](https://img.shields.io/badge/license-MIT-fa4.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-2026--07--28-blue.svg)](https://modelcontextprotocol.io)
[![npm](https://img.shields.io/badge/npm-total--agent--memory-cb3837.svg)](https://www.npmjs.com/package/total-agent-memory)
[![PyPI](https://img.shields.io/badge/PyPI-total--agent--memory-3776AB.svg)](https://pypi.org/project/total-agent-memory/)
[![Docker GHCR](https://img.shields.io/badge/docker-ghcr.io-2496ED.svg)](https://github.com/vbcherepanov/total-agent-memory/pkgs/container/total-agent-memory)
[![Homebrew](https://img.shields.io/badge/brew-vbcherepanov%2Ftap-FBB040.svg)](https://github.com/vbcherepanov/homebrew-tap)
[![Donate](https://img.shields.io/badge/PayPal-Donate-00457C.svg?logo=paypal&logoColor=white)](https://PayPal.Me/vbcherepanov)

**Why this, not mem0 / Letta / Zep / Supermemory / Cognee?** → [docs/vs-competitors.md](docs/vs-competitors.md)

---

## v13.0.0 — MCP 2026-07-28, and honest benchmarks (2026-08-27)

> **Upgrade if you installed after the MCP Python SDK went 2.0.** The 2.x line
> dropped the `@Server.list_tools()` / `@Server.call_tool()` decorators this
> server was built on, and the dependency was floored at `mcp[cli]>=1.0.0` — so
> every fresh `pip` / `uvx` / `npx` / `brew` / `docker` install resolved 2.x and
> died at import. Existing installs kept working only because their pinned 1.x
> never moved.

**Protocol.** Tools now register through whichever API the installed SDK
exposes, and the server serves both protocol eras from one process: the
stateless **2026-07-28** revision — `tools/list`, `server/discover` and
`tools/call` with no `initialize` handshake, protocol metadata per request —
alongside the legacy handshake for clients on older SDKs. JSON-answering tools
return `structuredContent`, so clients stop re-parsing strings, and all 74
tools carry `readOnlyHint` / `destructiveHint` / `idempotentHint` annotations
that clients use to decide what runs without a confirmation prompt.

**Claude Code plugin.** The MCP server, the `memory-protocol` skill and the
seven capture hooks now install in one step:

```bash
/plugin marketplace add vbcherepanov/total-agent-memory
/plugin install total-agent-memory@vbcherepanov
```

**LongMemEval now measures the product.** The runner had its own
self-contained BM25 / RRF / MMR / CrossEncoder stack, so the published 96.2%
described an algorithm rather than this software. A new `--modes store` — now
the default — ingests each haystack into a real `Store` and queries
`Recall.search`. Re-measured: **95.1% R@5**, 27.6 ms per query.

**Benchmarks that no longer measure themselves.** `Recall.search` bumps
`recall_count` on every row it returns, and the scorer adds
`recall_boost = min(0.3, recall_count * 0.05)`. Spaced repetition is wanted in
normal use and fatal for measurement: successive runs against one database
scored 0.547 → 0.565 → 0.588 → 0.607 R@5 without a line of retrieval code
changing. Both runners now pass `record_usage=False`, a clean run and a re-run
are byte-identical, and every number below was re-measured on that basis. The
LoCoMo runner had also been printing categories 2 and 3 under each other's
labels.

**[BEAM](https://github.com/mohammadtavakoli78/BEAM) (ICLR 2026)** is now part
of the suite — retrieval across its ten memory abilities at the 100K / 500K /
1M scales, graded against each probe's `source_chat_ids` with no LLM in the
loop.

**And ~3 GB it could not reach (13.0.2).** The base install resolved
`sentence-transformers`, `transformers`, `FlagEmbedding` and `peft`, each of
which resolves torch, which on Linux resolves the entire `nvidia-cu*` set: 147
packages and ~3,108 MB of wheels against 97 and ~113 MB without them. The
[Glama](https://glama.ai) build sandbox simply ran out of disk unpacking
`nvidia-cudnn-cu13`. Yet the default configuration cannot touch any of it —
`MEMORY_MODE=fast` disables the reranker, and the same mode's
`MEMORY_ALLOW_OLLAMA_IN_HOT_PATH=false` is the flag that gates the
`SentenceTransformer` fall-through in `Recall._compute`. The stack now lives in
a `rerank` extra, and the mirror of the dependency-drift test keeps it there.
Every installer was also warming `all-MiniLM-L6-v2` — the fallback model, not
the one the server embeds with — into a cache nothing reads.

**The server was carrying ~450 MB it never used.** `chromadb` and
`sentence_transformers` were imported at module scope, both are fallback paths,
and the second pulls in torch — so every user paid for a stack that fastembed
made unnecessary. Deferring them took `import server` from 558 MB to **116 MB**
and a serving process from 1367 MB to **909 MB**. Reported by d.snezhinskiy.
A failed fastembed init also stops being a single log line: it now names the
cache and the memory cost, because a macOS-purged model cache is the usual
reason a memory server suddenly wants 1.5 GB.

**Bugs worth naming — all of the "works in a checkout, silently dead when
installed" kind.** `tree-sitter-language-pack` was in no requirements file, so
"AST codebase ingest, 9 languages" degraded to whole-file chunks for everyone.
`vocabularies/` and `filters/` never made it into the wheel or the image, so
canonical tag normalisation ran against an empty vocabulary and every
`memory_save(filter=…)` was a no-op. The enrichment worker shared the Store's
sqlite connection — safe for reads, not for writes — and long ingests died on
`cannot start a transaction within a transaction`. Migration 028 failed on every fresh
database and could never record itself, so it retried on every startup forever
(root cause spotted by @juicetin in #12: two owners for one schema change). And
`ai_layer/verifier.py` looked for NLI calibrations at the pre-`.tam` path.

Full notes in [`CHANGELOG.md`](CHANGELOG.md#1300--2026-08-27--mcp-2026-07-28-and-honest-benchmarks).
Earlier releases: [v12.4.0](CHANGELOG.md#1240--2026-05-26--100-functional-through-every-install-path) ·
[v12.0.0](CHANGELOG.md#1200--2026-05-16) ·
[v11.0](CHANGELOG.md#1100).

---

## Table of contents

- [v13.0.0 — what changed](#v1300--mcp-2026-07-28-and-honest-benchmarks-2026-08-27)
- [The problem it solves](#the-problem-it-solves)
- [60-second demo](#60-second-demo)
- [Benchmarks — how it compares](#benchmarks--how-it-compares)
  - [LoCoMo](#locomo--snap-researchlocomo) · [BEAM](#beam--beyond-a-million-tokens-iclr-2026) · [LongMemEval](#longmemeval--xiaowu0162longmemeval-cleaned)
- [Competitor comparison](#competitor-comparison)
- [What you get](#what-you-get)
- [Architecture](#architecture)
- [Install](#install)
- [Quick start](#quick-start)
- [CLI: `lookup-memory` for sub-agents](#cli-lookup-memory-for-sub-agents)
- [MCP tools reference](#mcp-tools-reference-74-tools)
- [TypeScript SDK](#typescript-sdk)
- [Dashboard](#dashboard-localhost37737)
- [Update](#update)
- [Upgrading from v8.x to v9.0](#upgrading-from-v8x-to-v90)
- [Upgrading from v7.x to v8.0](#upgrading-from-v7x-to-v80)
- [Ollama setup](#ollama-setup-optional-but-recommended)
- [Configuration](#configuration)
- [Performance tuning](#performance-tuning)
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
- **Searchable** — 6-stage hybrid retrieval (BM25 + dense + graph + CrossEncoder + MMR + RRF fusion), **95.1% R@5 on public LongMemEval**
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

Everything below is **retrieval**: does the memory surface the passage that
contains the answer, in the top-K? That is the part this project owns —
answer quality is bounded above by it, and it can be graded with no LLM in
the loop, which makes the numbers deterministic, free, and reproducible on
your machine.

Two things to read them honestly:

- These are the **default `fast` profile** — FastEmbed, no reranker, no LLM
  anywhere in the path. That is what you get after `install.sh`, not a tuned
  configuration.
- Every runner passes `record_usage=False`. `Recall.search` normally bumps
  `recall_count`, and the scorer adds `recall_boost = min(0.3, recall_count ×
  0.05)` — so before v13, each re-run against the same database scored higher
  than the last, partly measuring its own history. A clean run and a re-run
  are now byte-identical.

### LoCoMo — [snap-research/locomo](https://github.com/snap-research/locomo)

1,536 gradable questions across 10 long-running conversations (5,882 turns
ingested), plus 446 adversarial questions scored separately.

| Category | N | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| single-hop | 282 | 0.202 | 0.500 | 0.638 | 0.332 |
| temporal | 321 | 0.411 | **0.689** | 0.735 | 0.524 |
| multi-hop | 92 | 0.163 | 0.413 | 0.435 | 0.256 |
| open-domain | 841 | 0.363 | 0.633 | 0.712 | 0.479 |
| **overall** | **1,536** | **0.331** | **0.607** | **0.687** | **0.448** |

Latency p50 **18.2 ms**, p95 55.4 ms. Temporal is the strongest category —
the bi-temporal knowledge graph earns its keep. Multi-hop is the weakest and
is the v13.1 target.

Reproduce: `python benchmarks/locomo_bench.py --wipe` →
[`benchmarks/results/v13-locomo-retrieval.json`](benchmarks/results/v13-locomo-retrieval.json)

### BEAM — [Beyond a Million Tokens](https://github.com/mohammadtavakoli78/BEAM), ICLR 2026

BEAM is the benchmark that starts where context windows stop: conversations of
100K / 500K / 1M tokens (a separate 10M set goes further), probed across ten
distinct memory abilities. Scored here against each probe's `source_chat_ids`.

**Scale 100K** — 20 conversations, 5,732 messages, 355 gradable probes:

| Ability | N | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| contradiction_resolution | 40 | 0.700 | **1.000** | 1.000 | 0.824 |
| temporal_reasoning | 40 | 0.475 | **0.975** | 1.000 | 0.689 |
| knowledge_update | 40 | 0.550 | **0.925** | 0.950 | 0.719 |
| multi_session_reasoning | 40 | 0.375 | 0.675 | 0.850 | 0.486 |
| information_extraction | 40 | 0.400 | 0.625 | 0.725 | 0.503 |
| summarization | 36 | 0.167 | 0.444 | 0.556 | 0.267 |
| preference_following | 39 | 0.077 | 0.282 | 0.410 | 0.169 |
| event_ordering | 40 | 0.025 | 0.150 | 0.200 | 0.074 |
| instruction_following | 40 | 0.025 | 0.075 | 0.150 | 0.054 |
| **overall** | **355** | **0.313** | **0.575** | **0.651** | **0.423** |

Latency p50 **17.7 ms**. The shape is the useful part: contradiction
resolution, temporal reasoning and knowledge update are effectively solved,
while `instruction_following` and `event_ordering` are near-zero — those probes
ask *whether a stated instruction was followed* or *in what order things
happened*, and semantic similarity to the question does not find the message
where the instruction was given. Retrieval is the wrong primitive there, and
that is the roadmap item.

**Scale 500K** — 35 conversations, 38,058 messages, 629 gradable probes:

| Ability | N | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| contradiction_resolution | 70 | 0.714 | **0.943** | 0.971 | 0.828 |
| knowledge_update | 69 | 0.464 | **0.855** | 0.899 | 0.617 |
| temporal_reasoning | 70 | 0.500 | **0.786** | 0.871 | 0.625 |
| multi_session_reasoning | 70 | 0.357 | 0.614 | 0.729 | 0.470 |
| information_extraction | 70 | 0.271 | 0.443 | 0.571 | 0.354 |
| preference_following | 70 | 0.071 | 0.300 | 0.471 | 0.168 |
| summarization | 70 | 0.100 | 0.286 | 0.414 | 0.174 |
| instruction_following | 70 | 0.029 | 0.157 | 0.257 | 0.086 |
| event_ordering | 70 | 0.014 | 0.029 | 0.186 | 0.042 |
| **overall** | **629** | **0.280** | **0.490** | **0.596** | **0.373** |

**Scale 1M** — 35 conversations, 74,630 messages, 625 gradable probes:

| Ability | N | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| knowledge_update | 70 | 0.529 | **0.886** | 0.929 | 0.677 |
| contradiction_resolution | 70 | 0.686 | **0.871** | 0.914 | 0.772 |
| temporal_reasoning | 70 | 0.371 | 0.686 | 0.800 | 0.508 |
| multi_session_reasoning | 70 | 0.214 | 0.429 | 0.600 | 0.315 |
| information_extraction | 70 | 0.157 | 0.371 | 0.500 | 0.250 |
| summarization | 66 | 0.015 | 0.288 | 0.515 | 0.147 |
| preference_following | 69 | 0.029 | 0.246 | 0.406 | 0.134 |
| event_ordering | 70 | 0.000 | 0.157 | 0.329 | 0.069 |
| instruction_following | 70 | 0.029 | 0.086 | 0.200 | 0.061 |
| **overall** | **625** | **0.227** | **0.448** | **0.578** | **0.327** |

### How it scales, and what that exposed

| Scale | Messages | R@5 | search p50 | ingest |
|---|---:|---:|---:|---:|
| 100K | 5,732 | 0.575 | 17.7 ms | 25.6 msg/s |
| 500K | 38,058 | 0.490 | 58.5 ms | 10.8 msg/s |
| 1M | 74,630 | **0.448** | **411.5 ms** | **5.0 msg/s** |

Recall decays gracefully — 13× the haystack costs 12.7 points of R@5, and the
abilities that hold up (knowledge update, contradiction resolution) hold up at
every scale. The two curves that do *not* decay gracefully are the interesting
part, and they have separate causes.

**Ingest — found and fixed.** Throughput fell 5× across the three scales on
identical code. The cause was ours: `graph/auto_link.py` runs on every save and
constructed a fresh `ConceptExtractor` each time. The node-name cache lives on
the instance, so it was thrown away immediately and the whole `graph_nodes`
table was re-read per write — 1,000 saves triggered 1,000 full table reads
(~139 million rows at the 139k nodes this ingest reaches). Fixed in v13.0.1;
counting reads rather than timing makes the check load-independent, and it is
now **1** read per 1,000 saves. **The ingest column above was measured before
that fix** and is kept as the record of the problem.

**Search — open.** p50 grew 7× between 500K and 1M for 2× the data.
`Store._binary_search` loads the binary vectors of every active record into
numpy on each query, so search is linear in store size. That is a different
problem from the ingest one and is not fixed; an ANN index over the binary
vectors is the obvious answer and has not been built yet. Stated rather than
buried, because 411 ms is a real number a user would feel.

Reproduce: `python benchmarks/beam_bench.py --scale 100K --wipe` →
[`v13-beam-100K.json`](benchmarks/results/v13-beam-100K.json) ·
[`v13-beam-500K.json`](benchmarks/results/v13-beam-500K.json) ·
[`v13-beam-1M.json`](benchmarks/results/v13-beam-1M.json)

### LongMemEval — [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

470 questions across six question types, re-measured for v13 **through the
product**: each question's haystack is ingested into a real `Store` and queried
with `Recall.search`, the same path an agent takes.

| Question type | Count | R@5 (recall_any) |
|---|---:|---:|
| knowledge-update | 72 | **100.0%** |
| multi-session | 121 | **98.3%** |
| single-session-user | 64 | 95.3% |
| single-session-assistant | 56 | 94.6% |
| temporal-reasoning | 127 | 92.9% |
| single-session-preference | 30 | 80.0% |
| **total** | **470** | **95.1%** |

Also `recall_all@5` 85.7% (every required fragment, not just one), NDCG@5
88.9%, **27.6 ms** per query.

> **This replaces the 96.2% we published before**, and the difference matters
> more than the 1.1 points. Until v13 this runner used its own self-contained
> BM25 / RRF / MMR / CrossEncoder stack, so the number described *an
> algorithm*, not this software. `--modes store` drives the shipping path and
> is now the default. The old modes remain for ablations.
>
> For reference on the same set, Mastra "Observational" reports 95.0% and
> Supermemory 85.4% — both cloud services.

Reproduce: `python benchmarks/longmemeval_bench.py --modes store` →
[`evals/longmemeval-2026-08-27-v13-store.json`](evals/longmemeval-2026-08-27-v13-store.json)

### On end-to-end accuracy numbers

Systems in this space usually publish LoCoMo **accuracy** — a generator answers
from the retrieved context and an LLM judges it. We publish it too, with the
two caveats that make it meaningful.

**One LLM-judged run is a sample, not a measurement.** Temperature 0 does not
make the API deterministic and OpenAI documents `seed` as best-effort, so the
runner takes `--seed` and we report three runs:

| Category | N | mean | min | max | spread |
|---|---:|---:|---:|---:|---:|
| single-hop | 282 | 0.366 | 0.358 | 0.372 | 0.014 |
| temporal | 321 | 0.426 | 0.424 | 0.427 | 0.003 |
| multi-hop | 96 | 0.292 | 0.281 | 0.302 | **0.021** |
| open-domain | 841 | 0.570 | 0.567 | 0.573 | 0.006 |
| adversarial | 446 | **0.904** | 0.899 | 0.908 | 0.009 |
| **overall (no adversarial)** | 1,540 | **0.486 ± 0.002** | 0.484 | 0.488 | 0.005 |
| **overall (all)** | 1,986 | **0.579 ± 0.002** | 0.578 | 0.582 | 0.004 |

gpt-4o generator, gpt-4o-mini judge, seeds 1/2/3. Retrieval was **byte-identical
across all three** — only generation and judging vary.

**The judge needed two guards, and they point opposite ways.**

*Refusals scored as correct answers.* On ~100 of the 1,540 non-adversarial
questions per run, the judge answered YES to *"Not mentioned in the
conversation."* against golds like `Sweden`, `June 2023`, `Single` — F1 exactly
0.00. Almost certainly the adversarial rule bleeding across, since the judge is
told to accept a refusal when the gold also indicates no information. Per
category the inflation runs **3.2 pp (open-domain) to 14.3 pp (temporal)**.

*Hallucinations scored as correct abstentions.* **99.6% of LoCoMo's adversarial
golds are the empty string.** The judge accepts almost any fluent answer against
an empty reference, so 27–30 invented answers per run scored correct —
inflating the one category we used to lead on.

Both are rules rather than judgements — on categories 1–4 the gold *is* a fact,
so a refusal cannot be right; with an empty gold, only a refusal can be — so
both now run deterministically at judging time. **Effect: no-adv 0.551 → 0.486,
adversarial 0.966 → 0.904, all 0.645 → 0.579. The table above is corrected.**

How noisy is the rest? Aligning all 1,986 questions across the three seeds:

| | share |
|---|---:|
| generator's answer differed between seeds | 12.5% |
| judge's verdict differed | 5.1% |
| **judge flipped on an identical answer** | **2.7%** |

The aggregate holds within ±0.005 because those flips roughly cancel, not
because the instrument is precise. Quoting one run to three decimals — as we
did before — is not supported by the data.

Not comparable to the 90%+ figures some competitors publish: different
generators, judges, prompts and question subsets. And on this evidence, an
unguarded LLM judge can be worth six points on its own. The retrieval numbers
above remain our primary metric because they are checkable without an API key.

[`benchmarks/results/v13-locomo-llm-3seeds.json`](benchmarks/results/v13-locomo-llm-3seeds.json) ·
Runner: [`benchmarks/locomo_bench_llm.py`](benchmarks/locomo_bench_llm.py)

### Do the retrieval numbers mean anything? — negative controls

A retrieval score with no floor under it is not a claim. Every LoCoMo run now
scores three degenerate baselines on the same questions:

| Baseline | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| random — ten turns from the same conversation | 0.001 | 0.012 | 0.023 |
| first — the ten earliest turns | 0.000 | 0.023 | 0.039 |
| recency — the ten most recent turns | 0.001 | 0.003 | 0.011 |
| **the pipeline** | **0.331** | **0.607** | **0.687** |

**27× the best degenerate baseline.** The controls run in the same pass as the
metric, so the floor ships with the number rather than living in a script
somebody stops running.

### Latency profile

```
  p50 (warm)   ▌ 0.065 ms
  p95 (warm)   ▌▌ 2.97 ms
  LoCoMo       ▌▌▌ 18.2 ms/query    ← full hybrid retrieval over 5,882 records
  BEAM 100K    ▌▌▌ 17.7 ms/query    ← over 5,732 messages
  LongMemEval  ▌▌▌▌▌ 38.8 ms/query  ← includes embedding + CrossEncoder rerank
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
| MCP-native | via SDK | ❌ | 🟡 Graphiti | 🟡 | ❌ | ❌ | **✅ 74 tools, MCP 2026-07-28** |
| Knowledge graph | 🔒 $249/mo | ❌ | ✅ | ✅ | ✅ | ❌ | **✅** |
| **Temporal facts** (`kg_at`) | ❌ | ❌ | ✅ | ❌ | 🟡 | ❌ | **✅** |
| **Procedural memory** | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | **✅ `workflow_predict`** |
| **Cross-project analogy** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ `analogize`** |
| **Self-improving rules** | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | **✅ `learn_error`** |
| **AST codebase ingest** | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | **✅ tree-sitter 9 lang** |
| **Pre-edit risk warnings** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ `file_context`** |
| 3D WebGL graph viewer | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | **✅** |
| Price for graph features | $249/mo | free | cloud | usage | free | free | **free** |

**On competitors' benchmark numbers.** mem0 now publishes 92.5 on LoCoMo and
94.4 on LongMemEval. Those are end-to-end accuracy with their own generator,
judge and prompts — not comparable to the retrieval numbers above, and not
independently reproducible without their stack. We publish retrieval because
the runner, the corpus and the gold labels are all public and you can re-run
them on your laptop without an API key. Where a project has not published on a
benchmark, we write "—" rather than inventing a number.

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

- **6-stage hybrid retrieval** (BM25 + dense + fuzzy + graph + CrossEncoder + MMR, RRF fusion) — 95.1% R@5 public
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
                                         │ 74 tools
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

### Quickstart — pick one

| Channel | Command | What it does |
|---|---|---|
| **npx** (Node) | `npx -y total-agent-memory connect claude-code` | Zero-install. Bootstraps a Python venv in `~/.tam/.venv` via uv (or python3 fallback), pulls the PyPI server, wires the MCP entry into your IDE. Replace `claude-code` with `codex` / `cursor` / `cline` / `continue` / `aider` / `windsurf` / `gemini-cli` / `opencode`. |
| **uvx** (Python via uv) | `uvx total-agent-memory` | One-off run with no install. Best for trying without commitment. |
| **pipx** (Python isolated) | `pipx install total-agent-memory` | Installs the `total-agent-memory`, `tam`, `tam-lookup`, `lookup-memory` binaries on PATH in an isolated venv. |
| **brew** (macOS / Linuxbrew) | `brew install vbcherepanov/tap/total-memory` | Bottle-style install with `tam` and legacy `claude-total-memory` symlinks. |
| **Docker** (multi-arch) | `docker run -p 37737:37737 -v ~/.tam:/data ghcr.io/vbcherepanov/total-agent-memory:13.0.4` | Containerized (linux/amd64 + linux/arm64). Dashboard on `:37737`. |
| **Claude Code plugin** | `/plugin marketplace add vbcherepanov/total-agent-memory`<br>`/plugin install total-agent-memory@vbcherepanov` | Installs the MCP server, the `memory-protocol` skill and all seven capture hooks in one step, from inside Claude Code. The bootstrap reuses an existing install if it finds one, so nothing is downloaded twice. |
| **Manual clone** | `git clone https://github.com/vbcherepanov/total-agent-memory ~/total-agent-memory && cd ~/total-agent-memory && ./install.sh --ide claude-code` | Full control. Lets you hack on the server, run benchmarks, and pick which background services to enable. Detailed walkthrough below. |

All seven channels land at the same MCP server. The `npx` and `./install.sh` paths
additionally configure IDE-specific MCP entries and hooks. Other channels start
the server bare — you wire the IDE afterwards (see [`docs/installation.md`](docs/installation.md)).

**The reranker is an extra, not a dependency.** A base install is 97 packages
and ~113 MB of wheels: fastembed runs the embeddings through ONNX and no torch
is resolved anywhere. The CrossEncoder / BGE reranker needs the torch stack,
which on Linux drags in the whole `nvidia-cu*` set — 147 packages and ~3.1 GB —
so it ships separately, and the default `MEMORY_MODE=fast` does not use it. Turn
it on with `MEMORY_MODE=deep` (or `MEMORY_RERANK_ENABLED=true`) and install it:

```bash
pip install "total-agent-memory[rerank]"      # pip / uvx / pipx
pip install -r requirements-rerank.txt        # clone / Docker
```

**Upgrade from v11.x?** Whatever channel you pick will auto-migrate
`~/.claude-memory/` → `~/.tam/` on first run and keep a symlink for backward
compat. No manual data move required.

---

### Detailed paths (manual / Docker / per-IDE)

Two manual paths. Same 74 tools, same dashboard, different deployment shapes.

### IDE matrix (v10.5)

The same MCP server, same tools, same protocol — different installation
locations and hook wiring per IDE. The installer (`install.sh --ide <name>`)
automates all of it.

| IDE | Skill API | Hook API | Sub-agents | Install command |
|---|:-:|:-:|:-:|---|
| Claude Code | ✅ | ✅ full | ✅ | `./install.sh --ide claude-code` |
| Codex CLI | ✅ | ✅ | ❌ | `./install.sh --ide codex` |
| Cursor | rules-pane | ❌ | composer | `./install.sh --ide cursor` |
| Cline (VS Code) | `.clinerules/` | ❌ | ❌ | `./install.sh --ide cline` |
| Continue | rules file | ❌ | ❌ | `./install.sh --ide continue` |
| Aider | `.aider.conf.yml` read | ❌ ¹ | ❌ | `./install.sh --ide aider` |
| Windsurf | `.windsurfrules` | ❌ | cascade | `./install.sh --ide windsurf` |
| Gemini CLI | `.gemini/rules/` | ⚠️ partial | ❌ | `./install.sh --ide gemini-cli` |
| OpenCode | `.opencode/skills/` | ✅ | custom | `./install.sh --ide opencode` |

¹ Aider has no MCP yet — the bridge is via `lookup_memory.sh` /
`save_memory.sh` shell scripts.

Full per-IDE setup, manual fallbacks, and template snippets:
[`skills/memory-protocol/references/ide-setup.md`](skills/memory-protocol/references/ide-setup.md).

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
git clone https://github.com/vbcherepanov/total-agent-memory.git ~/total-agent-memory
cd ~/total-agent-memory
bash install.sh --ide claude-code   # or: cursor | gemini-cli | opencode | codex
```

The installer:

1. Clones + creates `~/total-agent-memory/.venv/`
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

Restart Claude Code → `/mcp` → `memory` should show **Connected** with 74 tools.

### Path A — native (Windows 10/11)

```powershell
git clone https://github.com/vbcherepanov/total-agent-memory.git $HOME\total-agent-memory
cd $HOME\total-agent-memory
powershell -ExecutionPolicy Bypass -File install.ps1 -Ide claude-code
```

Same 9 steps as Unix, but:

- MCP config path is `%USERPROFILE%\.claude\settings.json` (or `.cursor\mcp.json`, etc.)
- Hooks copied to `%USERPROFILE%\.claude\hooks\` — `.ps1` versions (auto-capture, memory-trigger, user-prompt-submit, post-tool-use, pre-edit, on-bash-error, session-start/end, on-stop, codex-notify)
- Background services via **Task Scheduler**:
  - `total-agent-memory-reflection` — every 5 min (no native FileSystemWatcher equivalent)
  - `total-agent-memory-orphan-backfill` — daily 00:00 + 6h repetition
  - `total-agent-memory-check-updates` — weekly Mon 09:00
  - `TotalAgentMemoryDashboard` — AtLogon

### Uninstall

All installers preserve `~/.tam/memory.db` (legacy installs: `~/.claude-memory/memory.db`) and your config files; only services + hook registrations are removed.

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
git clone https://github.com/vbcherepanov/total-agent-memory.git
cd total-agent-memory
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

> **v11 default is `MEMORY_MODE=fast`.** No LLM, no Ollama, no network in the save/search/recall hot path. To restore v10.5 synchronous-LLM behaviour set `export MEMORY_MODE=deep`. Mode switching: [`LAUNCH.md` § Tuning](LAUNCH.md#tuning-v110).

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
tam-lookup "Caroline researched"             # short canonical alias
ctm-lookup "Caroline researched"             # legacy alias (v11.x and earlier)

lookup-memory --project myproj --limit 5 "auth flow"
lookup-memory --type solution --tag reusable "fix bug"
lookup-memory --json "claude code hooks"     # structured stdout for piping
```

**How it works:** opens the same `$TAM_MEMORY_DIR/memory.db` (legacy: `$CLAUDE_MEMORY_DIR/memory.db`) the running MCP server uses → BM25 ranking via FTS5 → falls back to LIKE on older DBs. **Zero deps beyond the package.** No Ollama, no rag_chat.py, no ChromaDB required for the CLI path. Works on macOS, Linux, Windows.

```text
$ lookup-memory --project locomo_0 --limit 2 "adoption"
1. [synthesized_fact|locomo_0] Caroline is researching adoption agencies.
2. [synthesized_fact|locomo_0] Melanie congratulates Caroline on her adoption.
```

**Why three names?** `lookup-memory` matches the legacy bash script that older docs and sub-agent prompts reference (`~/claude-memory-server/ollama/lookup_memory.sh`, legacy install path). `tam-lookup` is the new project-prefixed canonical form (v12+). `ctm-lookup` is the v11.x prefixed name, kept as a legacy alias. All three call into `total_agent_memory.lookup:main` (v11.x and earlier: `claude_total_memory.lookup:main`, still importable via deprecation shim).

**Migration note:** v7/v8 docs that pointed at `~/claude-memory-server/ollama/lookup_memory.sh` should be updated — the bash version still works for users with a manual install, but `./install.sh` / `./update.sh` clients on v9+ now get `lookup-memory` (and `tam-lookup`) on PATH directly via the package's `[project.scripts]` entry.

---

## MCP tools reference (74 tools)

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

Total: **74 tools.** Each is documented below with input schema and example.

Every tool carries MCP behaviour annotations — 38 are marked `readOnlyHint`,
and `memory_delete` / `memory_forget` / `memory_update` / `kg_invalidate_fact`
plus the two rebuild tools are marked `destructiveHint`. Clients use these to
decide what may run without a confirmation prompt. Tools that answer in JSON
also return it as `structuredContent`, so you do not have to parse the text.

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

Full JSON schemas: `python -m total_agent_memory.cli tools --json` or open the dashboard at `localhost:37737/tools`.

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

Screenshots → the dashboard is at `http://localhost:37737` once installed.

---

## Update

```bash
cd ~/total-agent-memory   # legacy clones: ~/claude-memory-server
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
cd ~/total-agent-memory   # legacy clones: ~/claude-memory-server
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
cd ~/total-agent-memory && ./update.sh   # legacy clones: ~/claude-memory-server
# pulls v9 src, installs new entry-points (tam, tam-lookup, lookup-memory; legacy: ctm-lookup),
# keeps existing memory.db untouched.
```

After upgrade, verify the new CLI is on PATH:

```bash
lookup-memory --limit 1 "any-query-from-your-history"
```

### What's new (no action required)

- **`lookup-memory` / `tam-lookup` / `ctm-lookup` (legacy)** CLI now installed alongside `total-agent-memory` MCP server (registered as `[project.scripts]` so `./install.sh` and `./update.sh` put them on PATH automatically). Sub-agent prompts that reference the legacy `~/claude-memory-server/ollama/lookup_memory.sh` script keep working; new prompts should prefer the package-installed name.
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
cd ~/total-agent-memory && ./update.sh   # legacy clones: ~/claude-memory-server
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

Quick reference — see full docs in [MCP tools reference](#mcp-tools-reference-74-tools):

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

### v11.0 — Memory mode + multi-embedding-space

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_MODE` | `fast` | `ultrafast\|fast\|balanced\|deep`. Selects hot-path profile. See [Performance tuning](#performance-tuning). |
| `MEMORY_USE_LLM_IN_HOT_PATH` | `false` | Master switch for sync LLM stages in `save_knowledge` / `Recall.search`. `MEMORY_MODE=deep` flips this to `true`. |
| `MEMORY_ALLOW_OLLAMA_IN_HOT_PATH` | `false` | Re-enables the silent FastEmbed → Ollama fallback ladder when FastEmbed is unavailable. |
| `MEMORY_RERANK_ENABLED` | `false` | Honour caller's `rerank=true`. When `false`, CrossEncoder rerank is hard-disabled even if a tool call requests it. |
| `MEMORY_ENRICHMENT_ENABLED` | `false` | Run the async enrichment worker. Default-ON in `balanced` / `deep`. |
| `MEMORY_TEXT_EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model for `embedding_space=text`. |
| `MEMORY_CODE_EMBED_MODEL` | _empty → falls back to TEXT model_ | Model for `embedding_space=code`. The row still records `space=code` so a future swap is config-only. |
| `MEMORY_LOG_EMBED_MODEL` | _empty → TEXT_ | Model for `embedding_space=log`. |
| `MEMORY_CONFIG_EMBED_MODEL` | _empty → TEXT_ | Model for `embedding_space=config`. |
| `MEMORY_DEFAULT_EMBEDDING_SPACE` | `text` | Space for unclassified content. |

### v10 + earlier

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_DB` | `~/.tam/memory.db` (legacy installs: `~/.claude-memory/memory.db`) | SQLite location |
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
| `MEMORY_ASYNC_ENRICHMENT` | `false` | **v10.1** — move quality gate / contradiction / entity dedup / episodic / wiki to a background worker. See [Performance tuning](#performance-tuning) |
| `MEMORY_ENRICH_TICK_SEC` | `0.1` | Worker tick interval (clamp `0.01..5`) |
| `MEMORY_ENRICH_BATCH` | `5` | Rows claimed per tick (clamp `1..50`) |
| `MEMORY_ENRICH_MAX_ATTEMPTS` | `3` | Retries before flipping a row to `failed` |
| `MEMORY_ENRICH_STALE_AFTER_SEC` | `60` | Seconds before a `processing` row is reclaimed (worker crash recovery) |

> CPU-only / WSL hosts: if Ollama keeps timing out, lower `MEMORY_TRIPLE_MAX_PREDICT` before raising timeouts. `install-codex.sh` writes conservative defaults automatically. **For 30-40s save latency on WSL2 → set `MEMORY_ASYNC_ENRICHMENT=true`** — see below.

Full config: see `total_agent_memory/config.py`.

---

## Performance tuning

### v11.0 fast-mode hot path (default)

When `MEMORY_MODE=fast` (default):

| metric              |   p50 |   p95 |   p99 |
|---------------------|------:|------:|------:|
| `save_fast`         |  6.2  |  8.9  | 11.4  |
| `save_fast` cached  |  0.3  |  0.4  |  1.4  |
| `search_fast`       |  3.4  |  4.7  |  6.0  |
| `cached_search`     |  3.1  |  3.4  |  3.6  |

`llm_calls=0`, `network_calls=0`. Reproduce: `./bin/memory-bench`. Regression gate: `./bin/memory-perf-gate`. Architecture rationale and per-stage audit: [`docs/v11/audit.md`](docs/v11/audit.md). Raw bench artifact: [`docs/v11/benchmark.md`](docs/v11/benchmark.md).

If your numbers do not match the table, run `./bin/memory-bench --warmup` first — cold FastEmbed import dominates the first call.

### Legacy: v10.5 deep-mode `memory_save` latency

The synchronous v10 hot path runs five LLM-bound stages inline so a `drop` verdict can block the INSERT and a contradiction supersede commits in the same transaction. On macOS with a warm Ollama that's ~340 ms median; on a WSL2 box without GPU/CoreML each LLM round-trip can stretch the same call into 30–40 seconds.

v10.1 ships an opt-in **inbox/outbox worker** that moves the heavy stages out of band:

```
sync   : privacy → canonical_tags → INSERT → embed → enqueue → return
worker : quality_gate → entity_dedup_audit → contradiction → episodic → wiki
```

Enable it in your env:

```bash
export MEMORY_ASYNC_ENRICHMENT=true
# Optional knobs (defaults shown):
export MEMORY_ENRICH_TICK_SEC=0.1
export MEMORY_ENRICH_BATCH=5
export MEMORY_ENRICH_MAX_ATTEMPTS=3
export MEMORY_ENRICH_STALE_AFTER_SEC=60
```

Restart the MCP server. A background daemon thread now consumes `enrichment_queue`; you can watch it on the dashboard panel **⚡ v10.1 enrichment worker**.

### Bench v10.5 (10-record corpus × 2 rounds, with LLM stages on)

`memory_save` latency:

| | min | p50 | **p95** | **p99** | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| **sync** (default) | 17.5 ms | 25.3 ms | **2150.5 ms** | **2179.0 ms** | 2186.1 ms | 348.0 ms |
| **async** (`MEMORY_ASYNC_ENRICHMENT=true`) | 18.1 ms | 22.3 ms | **26.7 ms** | **27.4 ms** | 27.5 ms | 22.7 ms |

`memory_recall` latency: p50 ≈ 3-5 ms in both modes (steady state),
with cold-cache p95 outliers on the first warmup hit.

**p95 collapses 80×** with async (`2150 ms → 27 ms`). On WSL2 with a
slow Ollama, the same shape holds — sync p95 of 30-40 s becomes
async p95 of ~300-1000 ms (LLM moves out of the hot path entirely).

Reproduce: `./.venv/bin/python benchmarks/v10_5_latency.py --rounds 2 --with-llm`.
Full report: [`benchmarks/v10_5_results.md`](benchmarks/v10_5_results.md).

### Trade-off — soft drop semantic

When async is on, a `quality_gate` `drop` no longer prevents the INSERT (we already committed in the sync path). Instead the row is marked `status='quality_dropped'` after the worker scores it. `memory_recall` ignores that status (`idx_knowledge_status_quality` is added in migration 020). Audit history stays in `quality_gate_log` so nothing is lost.

If you need strict pre-INSERT gating (e.g. compliance), keep the default sync path.

### Crash recovery

Rows stuck in `processing` longer than `MEMORY_ENRICH_STALE_AFTER_SEC` (default 60 s) are flipped back to `pending` automatically — covers worker process kills mid-stage. The pre-existing `write_intents` outbox still covers a crash *before* INSERT.

---

## Roadmap

### Shipped in v13.0.0 (2026-08-27)
- ✅ **MCP SDK 2.x compatibility** — the blocker: every install created after
  `mcp` 2.0 shipped was dead on arrival. Tools register through either SDK era;
  dependency bounded `>=1.9,<3`.
- ✅ **Protocol revision 2026-07-28** — stateless era served end-to-end
  (`tools/list` / `server/discover` / `tools/call` with no handshake), legacy
  handshake era from the same process, `structuredContent` on JSON-answering
  tools, behaviour annotations on all 74.
- ✅ **Claude Code plugin** — `/plugin install total-agent-memory@vbcherepanov`
  wires the MCP server, the skill and seven hooks in one step.
- ✅ **Reproducible benchmarks** — `record_usage=False` stops runs from
  measuring their own history; category labels in the LoCoMo runner corrected.
- ✅ **BEAM (ICLR 2026)** added to the suite at 100K / 500K / 1M.
- ✅ **`tree-sitter-language-pack` is now an actual dependency** — AST ingest
  had been silently degrading to whole-file chunks for every user.
- ✅ **Enrichment worker owns its sqlite connection** — long ingests no longer
  die on `cannot start a transaction within a transaction`.

### Shipped in v11.0 (2026-04-27) — production memory engine
- ✅ **Default `MEMORY_MODE=fast`** — zero LLM, zero Ollama, zero network in save/search/recall hot path. Set `MEMORY_MODE=deep` to restore v10.5 behaviour.
- ✅ **Memory Core / AI Layer split** — `src/memory_core/*` is deterministic; `src/ai_layer/*` owns every LLM-bound code path. Enforced by `tests/test_no_llm_hot_path.py`.
- ✅ **4 modes**: `ultrafast` / `fast` / `balanced` / `deep`. Single env flag.
- ✅ **Multi-embedding-space contract** — every vector row records provider / model / dimension / space / content_type / language. Spaces: `text` / `code` / `log` / `config`. Single Chroma backend; per-space model swap is config-only.
- ✅ **Embed fallback ladder gated** — silent Ollama fallback in `Store.embed` requires `MEMORY_ALLOW_OLLAMA_IN_HOT_PATH=true`.
- ✅ **New MCP tools**: `memory_save_fast`, `memory_search_fast`, `memory_explain_search`, `memory_warmup`, `memory_perf_report`, `memory_rebuild_fts`, `memory_rebuild_embeddings`, `memory_eval_locomo`, `memory_eval_recall`, `memory_eval_temporal`, `memory_eval_entity_consistency`, `memory_eval_contradictions`, `memory_eval_long_context`.
- ✅ **Migrations 021 (embedding_spaces) + 022 (embedding_cache_v11)** — idempotent on next start.
- ✅ **Benchmark suite**: `bin/memory-bench` (artifact `docs/v11/benchmark.md`) + `bin/memory-perf-gate` for CI.

### Shipped in v10.5 (2026-04-27)
- ✅ **Universal `memory-protocol` skill** — single canonical SKILL.md + 4 references (tool cheatsheet for all MCP tools, workflow recipes for 15 common situations, hooks reference, per-IDE setup) + 4 templates (Claude Code settings.json, Codex config.toml, Cursor `.mdc`, Cline `.md`). Same content for every IDE; only the wiring differs.
- ✅ **`install.sh --ide` extended to 9 IDEs**: claude-code, codex, cursor, **cline**, **continue**, **aider**, **windsurf**, gemini-cli, opencode. New helpers: `register_mcp_cline / continue / aider / windsurf` + `_json_merge_mcp_nested` for the dotted-key case (`cline.mcpServers`).
- ✅ **Cross-platform hardening** — all bash scripts pass `bash -n` under macOS bash 3.2 (default). Replaced `${var,,}` lowercase bashism in `update.sh` with `tr '[:upper:]' '[:lower:]'`. Verified with shellcheck.
- ✅ **Sub-agent memory protocol** — universal header for any sub-agent (`php-pro`, `golang-pro`, `vue-expert`, etc.) with mandatory `memory_recall` before / `memory_save` after. Full template in `skills/memory-protocol/references/subagent-protocol.md`.
- ✅ **v10.5 latency benchmark** — `benchmarks/v10_5_latency.py` with apples-to-apples sync vs async comparison. Demonstrates **80× p95 reduction** (`2150 ms → 27 ms`) when async is enabled with LLM stages on.

### Shipped in v10.1 (2026-04-27)
- ✅ **Async enrichment worker** — opt-in `MEMORY_ASYNC_ENRICHMENT=true` moves quality gate / entity dedup / contradiction detector / episodic linking / wiki refresh to a background thread. Drops max save latency 5.4× on macOS, 60–100× on WSL2. See [Performance tuning](#performance-tuning).
- ✅ **`enrichment_queue` table** with stale-processing recovery (rows stuck >60 s in `processing` flip back to `pending`).
- ✅ **Dashboard panel** for worker health: depth, throughput/min, p50/p95 ms per task, oldest pending age, recent failures.
- ✅ **`_binary_search` ValueError fix** — `np.argpartition` requires `kth STRICTLY < N`; tiny test projects (pool ≤ 50) used to silently break `contradiction_log`.
- ✅ **`coref_resolver` RU→EN translation fix** — prompt explicitly pins output language (`Do NOT translate`).

### Shipped in v10.0 (2026-04-27)
- ✅ **10 Beever-Atlas-inspired features in one push**: quality gate (Beever 6-Month Test), canonical tag vocabulary, importance boost in recall, opt-in coref resolution, contradiction auto-detection with supersede, write-intent outbox + reconciler, embedding-based entity dedup, episodic save events in the graph, smart query router (relational vs lexical), per-project Markdown wiki digest.
- ✅ 5 SQLite migrations (`015–019`) applied automatically on restart.
- ✅ 11 new env knobs, all with safe fail-open defaults.
- ✅ Tests: 971 → 1124 (+153).

### Shipped in v9.0 (2026-04-25)
- ✅ **`lookup-memory` / `tam-lookup` / `ctm-lookup` (legacy) CLI** — bash entry-point for sub-agents, registered as `[project.scripts]` and installed by `./install.sh` / `./update.sh` (replaces manual `~/claude-memory-server/ollama/lookup_memory.sh`)
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

### Next — what the v13 numbers say to fix

The benchmarks point at specific gaps rather than a general "make retrieval
better", so the roadmap names them:

- **`instruction_following` R@5 = 0.075, `event_ordering` = 0.150 (BEAM).**
  These probes ask *whether a stated instruction was followed* or *in what
  order things happened*. Semantic similarity to the question does not find the
  message where the instruction was given — retrieval is the wrong primitive.
  Needs a directive index (statements of the form "always/never/from now on")
  and ordering-aware traversal over the episodic graph.
- **`multi-hop` R@5 = 0.413 (LoCoMo).** Weakest category, and the one where
  the leaders win. Query decomposition without putting an LLM back in the hot
  path is the open design question.
- **`single_session_preference` R@5 = 0.80 (LongMemEval), `preference_following`
  = 0.282 (BEAM).** The same weakness from two directions: preferences are
  stated once, in passing, and never restated.
- **BEAM-10M.** The 1M scale runs today; 10M is the interesting claim.
- **Search is linear in store size.** BEAM 1M measured p50 411 ms against 58 ms
  at 500K — `Store._binary_search` loads every active record's binary vector
  into numpy per query. An ANN index over those vectors is the obvious answer.
  This is the largest open performance item.
- ~~Profile the write path~~ — done in v13.0.1: `auto_link` constructed a
  `ConceptExtractor` per save and threw away its node cache, re-reading the
  whole `graph_nodes` table on every write.

### Planned
- GitHub Actions: install smoke tests + a nightly retrieval gate, so a
  regression in R@5 fails CI the way `bin/memory-perf-gate` already fails on
  latency.
- `has_llm()` per-phase provider caching.

### Under research
- "Endless mode" — continuous session without hard boundaries (virtual sessions by idle >N hours)
- MLX local LLM integration
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
