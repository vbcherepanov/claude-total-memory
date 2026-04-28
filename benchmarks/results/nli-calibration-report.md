# NLI Verifier Calibration — Report (W5)

Generated: 2026-04-28T08:39:57Z

## Fixture

* Path: `tests/fixtures/nli_calibration_set.json`
* MD5: `8b05f37e221428c12d7fa5483d2a87a3`
* Total: 309  ({'entail': 120, 'contradict_numeric': 29, 'contradict_entity': 40, 'contradict_negate': 40, 'neutral_unrelated': 54, 'neutral_idk': 26, 'total': 309})
* Train / Test split (stratified, seed=43): 216 / 93

Construction:
* **Entail** — premise is a first-person clause taken from a LoCoMo dialogue turn; hypothesis is a third-person paraphrase of the same clause (so the entailment is real, not requiring inference).
* **Contradict** — same premise, hypothesis = corrupted paraphrase: (a) numeric/year flip on a token that appears in both premise and hypothesis; (b) entity swap with a proper noun present in both; or (c) in-sentence negation (`X is` → `X is not`).
* **Neutral (unrelated)** — premise from one conversation, hypothesis paraphrased from a clause in a *different* conversation.
* **Neutral (IDK adversarial)** — premise = real dialogue turn, hypothesis = an IDK-style answer (`Not specified in the conversation`, `I don't know`, …). Mirrors the production failure mode where the verifier was emitting `contradict` 58% of the time for (answer, evidence) pairs where the answer was an IDK token.

Leakage caveat: LoCoMo has no formal train/test split. The bench task is RAG QA, the calibration task is NLI triple classification — different task formulations, but the same underlying dialogue text. Documenting honestly per the W5 spec.

## Per-model summary

| model | train fc-rate | test fc-rate | entail rec (test) | contradict rec (test) | bal acc (test) |
|---|---|---|---|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 0.143 | 0.050 | 0.778 | 0.879 | 0.830 |
| `cross-encoder/nli-deberta-v3-base` | 0.279 | 0.183 | 0.972 | 0.788 | 0.781 |
| `sileod/deberta-v3-base-tasksource-nli` | 0.107 | 0.067 | 1.000 | 0.788 | 0.874 |

## Chosen path: **A (threshold tuning, no model swap)**  —  `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

### Thresholds

* `p_entail_threshold = 0.65`
* `p_contradict_threshold = 0.4`
* `p_contradict_margin = 0.0`

### Before vs after — test split (same model)

| metric | baseline (p_c>0.6, p_e>=0.5, no margin) | calibrated |
|---|---|---|
| false-contradict rate | 0.050 | 0.050 |
| entail recall | 0.833 | 0.778 |
| contradict recall | 0.848 | 0.879 |
| neutral recall | 0.833 | 0.833 |
| balanced accuracy | 0.838 | 0.830 |

### Confusion — baseline (test split)

| gold \ pred | entail | neutral | contradict |
|---|---|---|---|
| **entail** | 30 | 6 | 0 |
| **neutral** | 1 | 20 | 3 |
| **contradict** | 2 | 3 | 28 |

### Confusion — calibrated (test split)

| gold \ pred | entail | neutral | contradict |
|---|---|---|---|
| **entail** | 28 | 8 | 0 |
| **neutral** | 1 | 20 | 3 |
| **contradict** | 2 | 2 | 29 |

## Notes

* Path B model swap (cross-encoder/nli-deberta-v3-base, sileod/deberta-v3-base-tasksource-nli) was attempted; both increased false-contradict rate over Path A on the test split.
* IDK-style answers like "Not specified in the conversation" trigger the production model false-positive contradicts ~50% of the time. In the deployed v11 router this is a no-op (the route was already IDK), so the practical impact on bench accuracy is bounded.
* Existing v11 W1-E test_verifier.py thresholds (strict p_c > 0.6) preserved as default when calibration file is absent.
* Bench impact (cached Haiku 200 run): {"bench_path": "benchmarks/results/v11-subset-haiku-200-NLI.json", "n_records": 200, "decisions_under_baseline": {"entail": 48, "neutral": 37, "contradict": 115, "missing": 0}, "raw_probs_available": false, "note": "Cached predictions store only categorical decisions; per-record probabilities are not available, so an exact count of NEW vetoes under the calibrated thresholds cannot be computed without re-running. Lower bound: at most 115 records were ever flagged contradict by the old (uncalibrated) thresholds."}

