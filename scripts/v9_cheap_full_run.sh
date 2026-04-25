#!/usr/bin/env bash
# v9.0 LoCoMo cheap full run — gpt-4o-mini everywhere, est. ~$5 total.
#
# Prereqs:
#   1) export OPENAI_API_KEY=sk-...
#   2) pip install sentence-transformers 'FlagEmbedding>=1.2' torch
#   3) cd /Users/vitalii-macpro/PROJECT/claude-memory
#
# Usage:
#   bash scripts/v9_cheap_full_run.sh [smoke|full]
#     smoke = 100 QA, ~$0.30, sanity check
#     full  = 1986 QA, ~$3-5, leaderboard report
#
# IMPORTANT after run:
#   - Revoke key on platform.openai.com/api-keys (precaution).
#   - Final results: benchmarks/results/v9_cheap_full_*.json

set -euo pipefail

MODE="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[v9] ERROR: OPENAI_API_KEY not set."
    exit 1
fi

# ---- pre-flight: ML deps ----
deps_ok() {
    python3 - <<'PY'
import importlib, sys
for mod in ("sentence_transformers", "FlagEmbedding", "torch", "numpy"):
    try:
        importlib.import_module(mod)
    except ImportError:
        sys.exit(1)
PY
}

if ! deps_ok; then
    echo "[v9] Installing missing ML deps (one-time, ~600MB) ..."
    python3 -m pip install --quiet \
        'sentence-transformers>=2.7' 'FlagEmbedding>=1.2.10' torch numpy
fi

DB=/tmp/locomo_bench_db
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="$ROOT/benchmarks/results"
mkdir -p "$RESULTS_DIR"

# ---- env wiring ----
export MEMORY_EMBED_API_KEY="$OPENAI_API_KEY"
export MEMORY_EMBED_PROVIDER=openai
export MEMORY_EMBED_API_BASE="${MEMORY_EMBED_API_BASE:-https://api.openai.com/v1}"
export MEMORY_LLM_ENABLED=false   # ingest stays cheap; synthesis runs as separate step
export V9_EMBED_BACKEND=openai-3-large
export V9_RERANKER_BACKEND=bge-v2-m3
export V9_RERANKER_FP16=1

echo "[v9] mode=$MODE  db=$DB  embed=$V9_EMBED_BACKEND  reranker=$V9_RERANKER_BACKEND"

# ---- 1. ingest (~$0.01, runs 1 dummy QA to warm up) ----
if [[ ! -d "$DB" ]]; then
    echo "[v9] step 1/8: ingest LoCoMo turns + 1 warm-up QA (~$0.01)"
    python3 benchmarks/locomo_bench_llm.py --wipe \
        --db-path "$DB" \
        --gen-model gpt-4o-mini --judge-model gpt-4o-mini \
        --concurrency 8 --limit-qa 1
else
    echo "[v9] step 1/8: db already exists, skipping ingest"
fi

# ---- 2. re-embed на text-embedding-3-large (~$0.10) ----
echo "[v9] step 2/8: re-embed knowledge with text-embedding-3-large"
python3 scripts/reembed.py --db "$DB/memory.db" --backend openai-3-large --batch 64 --confirm

# ---- 3. synth_facts с LoCoMo v2 prompt на gpt-4o-mini (~$0.80) ----
echo "[v9] step 3/8: synth_facts (gpt-4o-mini + LoCoMo v2 prompt)"
python3 src/fact_synthesizer.py --db-path "$DB" \
    --provider openai --model gpt-4o-mini \
    --prompt-version v2 --concurrency 12 --reset

# ---- 4. extract_triples с canonical predicates (~$0.30) ----
echo "[v9] step 4/8: extract_triples (gpt-4o-mini + v2 canonical predicates)"
python3 scripts/extract_triples_openai.py --db-path "$DB" \
    --provider openai --model gpt-4o-mini \
    --prompt-version v2 --concurrency 12 --reset

# ---- 5. build temporal index (~$0.10) ----
if [[ -f scripts/build_temporal_index.py ]]; then
    echo "[v9] step 5/8: temporal index"
    python3 scripts/build_temporal_index.py --db-path "$DB" --concurrency 8 || true
else
    echo "[v9] step 5/8: build_temporal_index.py missing, skipping"
fi

# ---- 6. session summaries (~$0.10) ----
if [[ -f scripts/build_session_summaries.py ]]; then
    echo "[v9] step 6/8: session summaries"
    python3 scripts/build_session_summaries.py --db-path "$DB" --provider openai \
        --model gpt-4o-mini --concurrency 8 || true
fi

# ---- 7. mine few-shot pairs (free) ----
echo "[v9] step 7/8: mine LoCoMo few-shot pairs from train conv 0..6 (free)"
python3 scripts/mine_locomo_fewshot.py \
    --train-conv-ids 0,1,2,3,4,5,6 \
    --n-per-category 20 \
    --output benchmarks/data/locomo_few_shot_v2.json

# ---- 8. final bench run (~$3) ----
EVAL_SAMPLES_FLAG=""
RUN_LIMIT=""
if [[ "$MODE" == "smoke" ]]; then
    RUN_LIMIT="--limit-qa 100"
fi
RESULT_JSON="$RESULTS_DIR/v9_cheap_${MODE}_${TIMESTAMP}.json"
echo "[v9] step 8/8: bench run → $RESULT_JSON"
python3 benchmarks/locomo_bench_llm.py --skip-ingest \
    --db-path "$DB" \
    --reranker bge-v2-m3 \
    --few-shot-pairs benchmarks/data/locomo_few_shot_v2.json \
    --per-cat-prompts \
    --fact-index --semantic-fact-index \
    --query-rewrite --hyde --entity-boost \
    --temporal-filter \
    --ce-rerank \
    --ensemble 3 --ensemble-mode judge \
    --gen-model gpt-4o-mini --judge-model gpt-4o-mini \
    --concurrency 8 \
    $RUN_LIMIT \
    --output "$RESULT_JSON"

echo
echo "[v9] DONE. Results: $RESULT_JSON"
echo "[v9] Reminder: revoke OpenAI key on platform.openai.com/api-keys"
