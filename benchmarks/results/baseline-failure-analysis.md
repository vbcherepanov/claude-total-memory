# LoCoMo Baseline Failure Analysis

Date: 2026-04-28
Tool: `benchmarks/analyze_failures.py`

## Real baseline (paper-methodology-gpt4o, 1986 QA)

| Category | n | Accuracy | retrieval_miss | over_cautious | hallucination |
|----------|---|----------|----------------|---------------|---------------|
| single-hop | 282 | 0.440 | 100 (63%) | 35 (22%) | 22 (14%) |
| multi-hop | 321 | 0.573 | 61 (45%) | 51 (37%) | 24 (18%) |
| temporal | 96 | **0.260** | 46 (65%) | 25 (35%) | 0 |
| open-domain | 841 | 0.564 | 201 (55%) | 113 (31%) | 52 (14%) |
| adversarial | 446 | 0.969 | 5 | 0 | 9 |
| **OVERALL** | 1986 | 0.624 | **413 (55%)** | **224 (30%)** | 107 (14%) |

## Internal best per experiment (across 17 v9 runs)

| Run | Overall | single | multi | temp | open | adv |
|-----|---------|--------|-------|------|------|-----|
| **ensemble3** | **0.696** | 0.493 | 0.611 | 0.458 | 0.665 | 0.996 |
| fewshot-graph | 0.695 | **0.518** | 0.608 | 0.448 | 0.656 | 0.996 |
| mega-all | 0.694 | 0.493 | 0.604 | 0.427 | **0.667** | 0.993 |
| semantic-fact | 0.678 | 0.472 | **0.713** | 0.365 | 0.623 | 0.955 |
| paper-methodology | 0.624 | 0.440 | 0.573 | 0.260 | 0.564 | 0.969 |
| twostage-gpt4o | 0.514 | 0.255 | 0.474 | 0.125 | 0.411 | 0.984 |

## Key findings

1. **retrieval_miss is #1 root cause (55% of all errors)**
   - Episode layer (W1-A), IRCoT (W1-B), Entity Resolver (W1-F) are correctly aimed.
2. **over_cautious is #2 (30%)** — model refuses to answer even when evidence is retrieved.
   - W1-D (answerability) must lean PERMISSIVE: encourage answering when evidence exists.
   - Quick-win: review gen-prompt in `locomo_bench_llm.py` for over-cautious phrasing.
3. **hallucination only 14%** — NLI Verifier (W1-E) helps but is not the bottleneck.
4. **Temporal hard cap ~0.46** across all 17 experiments → without Allen's algebra (W1-C) no breakthrough.
5. **Best-per-category routing → soft upper bound 0.715** (already achievable with current per-category prompts; W2-I targets this).
6. **two-stage gating destroys performance** (-18pp). Avoid hard gate; use IDK as last resort, not first.

## Memory drift detected
- Memory #3944: claimed SOTA `0.687`. Actual peak in archive = `0.696` (ensemble3). Memory record stale — needs update.
- Memory #3936: claimed v11 release-ready, repo not git, version.py was at 10.5.0 (now synced to 11.0.0).

## Implications for prompt design
The over_cautious failure mode means the answer-pipeline default is "abstain". The new architecture must invert this:
- Answerability classifier (W1-D) → output "answer" when ANY evidence supports gold (not "all evidence must agree").
- IDK router → only return IDK when:
  a) `retrieval` ∈ ∅ AND `iters_done` == max, OR
  b) NLI verifier (W1-E) flags CONTRADICT against the candidate answer.
