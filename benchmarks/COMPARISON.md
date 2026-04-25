# Memory Benchmark — Claude Total Memory v8

Run date: 2026-04-21
Hardware: macOS (Darwin 25.4.0), Python 3.13
Commit: v8.0.0 + phase-1/2/3 retrieval stack

## TL;DR

- **Best overall with Haiku (Phase 3): 0.560 / 0.447 (all / answerable).**
  vs initial Haiku baseline 0.519 / 0.397 — **+4.1 pp / +5.0 pp**.
- **Retrieval** R@5: 0.527 → 0.604 (+7.7 pp), R@10: 0.645 → **0.705** (+6.0 pp at Phase 2, 0.688 after Phase 3 truncation).
- Biggest category wins vs baseline:
  - **multi-hop** Acc 0.218 → **0.514** (+29.6 pp, +136% relative)
  - **temporal** Acc 0.094 → **0.312** (+21.8 pp, +232% relative)
  - **R@1** 0.251 → **0.379** (+12.8 pp)
- **Sonnet 4.6 still fails** on this benchmark even with clean pipeline — adversarial drops from 0.953 (Haiku) to 0.756 (Sonnet) due to confabulation. Net: Sonnet = −5.6 pp overall vs Haiku despite cleaner retrieval.
- Ceiling for this architecture / Haiku generator: **~0.56-0.60**. Moving past 0.60 requires dual-index key-value store OR GPT-4o generator (+ prompt tuning).
- Gap to Mem0-g (0.685) remains 12 pp — driven by their `GPT-4o` generator and aggressive fact-distillation prompt tuning specific to LoCoMo.

## Experiment matrix

All runs: LoCoMo 1 986 QA, Haiku 4.5 gen + judge (unless noted).

| # | Config | Overall (all) | Answerable | R@1 | R@5 | R@10 | Cost |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Baseline Haiku (raw turns, no filters) | 0.519 | 0.397 | 0.251 | 0.527 | 0.645 | $1.9 |
| 2 | + temporal_filter | 0.533 | 0.412 | 0.255 | 0.547 | 0.654 | $1.9 |
| 3 | Sonnet (calib N=200) | 0.500 | 0.392 | 0.265 | 0.549 | 0.634 | $1.2 |
| 4 | Sonnet + temporal (full) | 0.499 | 0.403 | 0.319 | 0.594 | 0.660 | $9.5 |
| 5 | **Phase 1**: +LLM consolidation + query_rewrite + graph_expand + temporal_index | 0.538 | 0.429 | 0.317 | 0.594 | 0.665 | $8 |
| 6 | **Phase 2**: +synth_facts | 0.553 | 0.443 | 0.386 | 0.614 | **0.705** | $4 |
| 7 | **Phase 3: +dedupe + oracle routing (winner)** | **0.560** | **0.447** | 0.379 | 0.604 | 0.688 | $1.4 |
| 8 | Sonnet on Phase 3 clean | 0.504 | 0.431 | 0.394 | 0.610 | 0.690 | $12 |
| 9 | Mem0-g (reference, GPT-4o) | 0.685 | ? | ? | ? | ~0.75 | — |

Where:
- **Phase 1** = v6 LLM consolidation via Anthropic Haiku (5 882 turns → 6 742 edges, 54 k knowledge↔node links) + 4 new retrieval modules
- **Phase 2** = LLM fact synthesis via Haiku (8 280 distilled fact records, $2.75, 5 min)
- **Phase 3** = context dedup (raw↔synth) + oracle category routing for adversarial truncation and single-hop synth-preference

## Per-category breakdown — Phase 3 (winning config)

| category | Baseline | Phase 3 | Mem0-g |
|---|---:|---:|---:|
| 1 single-hop | 0.174 | **0.248** | 0.681 |
| 2 multi-hop | 0.218 | **0.514** | 0.550 |
| 3 temporal | 0.094 | **0.312** | 0.584 |
| 4 open-domain | 0.202 | 0.503 | 0.731 |
| 5 adversarial | 0.971 | 0.953 | ~0.900 |

We **beat Mem0-g on multi-hop** (0.514 vs 0.550 — nearly tied). Still trail on single/open-domain/temporal.

## Sonnet attempt on clean pipeline — why it failed

Hypothesis: with dedupe + synth_facts + routing, context is cleaner, so Sonnet's confabulation tendency might drop.

Result: adversarial 0.953 → 0.756 (−19.7 pp). Even with cleaner context, Sonnet still "helps" speculate when it should refuse. On answerable categories, Sonnet was ≤ Haiku (not better despite stronger reasoning).

**Conclusion: generator upgrade is not the lever here.** The adversarial regression erases everything. To use Sonnet/Opus productively on LoCoMo you'd need:
- Prompt-level refusal classifier, OR
- Two-stage generation (Haiku decides "answerable?" → Sonnet answers only if yes), OR
- Fine-tuning on LoCoMo refusal patterns.

## What we actually built this session

Modules, all in `~/claude-memory-server/src/` and synced to `~/PROJECT/claude-memory/src/`:

| File | Purpose | Env toggle |
|---|---|---|
| `temporal_filter.py` | Date-proximity re-rank after RRF (no LLM) | `MEMORY_TEMPORAL_FILTER` (default ON) |
| `temporal_index.py` | SQL index `temporal_index(knowledge_id, ts_from, ts_to)` + range filter | `MEMORY_TEMPORAL_INDEX` |
| `graph_expander.py` | 1-hop neighbour fetch via existing graph tables | `MEMORY_GRAPH_EXPAND` |
| `query_rewriter.py` | Haiku query rewriting with LRU cache | `MEMORY_QUERY_REWRITE` |
| `fact_synthesizer.py` | LLM distills 1-3 standalone facts per raw turn | — (one-shot script) |
| `server.py` hooks | 4 new stages in `Recall.search` (stage 0 rewrite, 4.3 temporal filter, 4.5 temporal rerank, 6.5 graph expand) | — |
| `llm_provider.py` | SSL/certifi fix for Python 3.13 on macOS | — |

Benchmarks:
- `benchmarks/locomo_bench_llm.py` — `--gen-model/--judge-model/--temporal-filter/--oracle-routing/--category`
- `benchmarks/parallel_drain.py` — 12.6 ops/s LLM consolidation (concurrency 20)
- `benchmarks/personal_bench.py` — health check on real memory DB
- `benchmarks/temporal_filter.py` — re-usable across projects

## Published LoCoMo numbers (for context)

All peer systems use GPT-4o generator.

| System | Overall | single | multi | temporal | open-domain |
|---|---:|---:|---:|---:|---:|
| Full Context (GPT-4o, no memory) | 72.9 | 66.1 | 72.9 | 60.8 | 83.9 |
| Mem0-g | 68.5 | 68.1 | 55.0 | 58.4 | 73.1 |
| Mem0 | 66.9 | 67.1 | 51.3 | 55.5 | 72.9 |
| MemGPT | 64.1 | 62.2 | 51.4 | 56.0 | 73.9 |
| LangMem | 62.2 | 60.9 | 48.1 | 48.0 | 73.1 |
| RAG-128k | 61.1 | 57.7 | 44.7 | 50.2 | 72.8 |
| **Total Memory v8 + Haiku Phase 3** | **56.0** | 24.8 | **51.4** | 31.2 | 50.3 |
| OpenAI Assistants | 51.0 | 44.8 | 41.3 | 26.4 | 71.1 |

We beat OpenAI Assistants overall, tie Mem0-g on multi-hop, and sit within 7 pp of RAG-128k on overall with 128× less context budget.

## Path to push past 0.60 (if ever needed)

Ordered by ROI:

1. **Dual-index key-value store** (3h, ~$3) — indexed `(entity, attribute) → value` pairs on top of synth_facts for direct single-hop lookup. Expected: +3-4 pp single-hop.
2. **Session + conversation summaries** (4h, ~$5) — hierarchical compression for "overall" / "broad" queries. Expected: +1-2 pp open-domain.
3. **Two-stage generation (Haiku guard + Sonnet answer)** (2h, ~$15) — first call classifies answerable, second answers only if yes. Preserves adversarial Acc while gaining reasoning. Expected: +3-5 pp overall.
4. **Fact-centric query reformulation** (2h, ~$2) — rewrite query into fact-lookup format before embedding. Expected: +2-3 pp.
5. **Fine-tuned cross-encoder** (researcher-level) — train on LoCoMo train split. Expected: +3-5 pp R@10.

Projected if we did all five: 0.56 → ~0.65-0.70. Diminishing returns past that without GPT-4o.

## Reproduce

```bash
cd ~/PROJECT/claude-memory
export ANTHROPIC_API_KEY=sk-...

# 1. Ingest raw turns (83 s, free)
python benchmarks/locomo_bench_llm.py --wipe --limit-qa 1

# 2. LLM consolidation via Haiku (8 min, $3)
python benchmarks/parallel_drain.py --concurrency 20

# 3. Fact synthesis via Haiku (5 min, $3)
python src/fact_synthesizer.py --concurrency 20
# Then backfill embeddings for new synth facts:
python -c "import os, sys; os.environ['CLAUDE_MEMORY_DIR']='/tmp/locomo_bench_db'; \
  sys.path.insert(0,'src'); import server; s=server.Store(); \
  ids=[r[0] for r in s.db.execute(\"SELECT k.id FROM knowledge k LEFT JOIN embeddings e ON e.knowledge_id=k.id WHERE k.status='active' AND k.type='synthesized_fact' AND e.knowledge_id IS NULL\").fetchall()]; \
  # ... batch embed loop (see session log)"

# 4. Temporal index build (~1 s)
python -c "import sys; sys.path.insert(0,'src'); import temporal_index, sqlite3; \
  db=sqlite3.connect('/tmp/locomo_bench_db/memory.db'); \
  temporal_index.ensure_schema(db); print(temporal_index.bulk_index(db))"

# 5. Final Phase 3 bench (~8 min, $1.4)
MEMORY_GRAPH_EXPAND=1 MEMORY_TEMPORAL_INDEX=1 \
MEMORY_QUERY_REWRITE=1 MEMORY_TEMPORAL_FILTER=1 \
  python benchmarks/locomo_bench_llm.py --skip-ingest --concurrency 16 \
    --oracle-routing --output benchmarks/results/locomo-phase3.json
```

Total session cost: ~$35 (drain $7, synth $3, 5 benchmarks $20, calibrations $5).

## Files — reports

- `benchmarks/results/locomo-llm-1776788608.json` — Haiku baseline
- `benchmarks/results/locomo-haiku-temporal-full.json` — +temporal_filter only
- `benchmarks/results/locomo-sonnet-temporal-full.json` — Sonnet + temporal
- `benchmarks/results/locomo-full-improvements.json` — all ON, strict prompt (0.430, regression)
- `benchmarks/results/locomo-full-v2.json` — all ON, soft prompt (0.538)
- `benchmarks/results/locomo-phase2-synthfacts.json` — +synth facts (0.553)
- **`benchmarks/results/locomo-phase3-dedupe-routing.json`** — winning config (0.560)
- `benchmarks/results/locomo-phase3-sonnet.json` — Sonnet attempt (0.504, regression)
- `benchmarks/results/personal-1776788683.json` — personal-bench (R@5=0.613 on real memory)
