"""v9.0 D5 — finetune_embedding helpers (parse, dataset iter, IO).

Heavy ML paths (mining, training) require sentence-transformers and are
exercised manually via `scripts/finetune_embedding.py`. This suite covers
the pure-python pieces that CI can run without GPU/CPU heavy deps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import finetune_embedding as fte  # noqa: E402


def _make_dataset(tmp_path: Path) -> Path:
    payload = [
        {
            "sample_id": "alpha",
            "conversation": {
                "session_1_date_time": "12:00 pm on 1 Jan, 2024",
                "session_1": [
                    {"speaker": "A", "dia_id": "D1:1", "text": "hello world"},
                    {"speaker": "B", "dia_id": "D1:2", "text": "i went to paris"},
                    {"speaker": "A", "dia_id": "D1:3", "text": "what was the weather"},
                ],
                "session_2_date_time": "12:00 pm on 2 Jan, 2024",
                "session_2": [
                    {"speaker": "B", "dia_id": "D2:1", "text": "it was raining hard"},
                ],
            },
            "qa": [
                {
                    "question": "Where did B go?",
                    "answer": "Paris",
                    "evidence": ["D1:2"],
                    "category": 1,
                },
                {
                    "question": "What was the weather?",
                    "answer": "raining",
                    "evidence": ["D2:1"],
                    "category": 2,
                },
                {
                    "question": "Open question with no evidence.",
                    "answer": "n/a",
                    "evidence": [],
                    "category": 4,
                },
            ],
        },
        {
            "sample_id": "beta",
            "conversation": {
                "session_1_date_time": "12:00 pm",
                "session_1": [
                    {"speaker": "X", "dia_id": "D1:1", "text": "filler turn"},
                ],
            },
            "qa": [],
        },
    ]
    p = tmp_path / "loco.json"
    p.write_text(json.dumps(payload))
    return p


def test_iter_samples_collects_turns_and_qa(tmp_path):
    p = _make_dataset(tmp_path)
    samples = fte._iter_samples(p)
    assert len(samples) == 2
    s0 = samples[0]
    assert s0.sample_id == "alpha"
    assert s0.sample_idx == 0
    assert len(s0.turns) == 4
    assert s0.turns[0].dia_id == "D1:1"
    assert s0.turns[3].session == 2
    assert len(s0.qa) == 3


def test_parse_conv_ids_default_uses_70_percent():
    assert fte._parse_conv_ids(None, 10) == [0, 1, 2, 3, 4, 5, 6]
    assert fte._parse_conv_ids(None, 1) == [0]
    assert fte._parse_conv_ids("", 10) == [0, 1, 2, 3, 4, 5, 6]


def test_parse_conv_ids_explicit():
    assert fte._parse_conv_ids("0, 2 ,5", 10) == [0, 2, 5]


def test_gather_positive_text_joins_evidence_turns(tmp_path):
    samples = fte._iter_samples(_make_dataset(tmp_path))
    s0 = samples[0]
    pos = fte._gather_positive_text(s0, ["D1:2", "D2:1"])
    assert "i went to paris" in pos
    assert "it was raining hard" in pos
    assert "B:" in pos  # speaker prefix preserved


def test_gather_positive_text_skips_unknown_ids(tmp_path):
    samples = fte._iter_samples(_make_dataset(tmp_path))
    s0 = samples[0]
    pos = fte._gather_positive_text(s0, ["D9:9"])
    assert pos == ""


def test_format_turn_includes_speaker():
    t = fte.Turn(dia_id="D1:1", speaker="Alice", text="hi", session=1)
    assert fte._format_turn(t) == "Alice: hi"


def test_triplet_jsonl_roundtrip(tmp_path):
    rows = [
        {"anchor": "q1", "positive": "p1", "negative": "n1", "category": 1, "sample_idx": 0},
        {"anchor": "q2", "positive": "p2", "negative": "n2", "category": 2, "sample_idx": 0},
    ]
    p = tmp_path / "triplets.jsonl"
    fte.write_triplets(rows, p)
    assert p.exists()
    out = fte.read_triplets(p)
    assert out == rows


def test_write_meta_records_train_ids(tmp_path):
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    triplets = tmp_path / "triplets.jsonl"
    triplets.write_text('{"anchor":"q","positive":"p","negative":"n"}\n')
    fte.write_meta(
        output_dir=out_dir,
        base_model_name="some-model",
        train_conv_ids=[0, 1, 2],
        triplet_path=triplets,
        n_triplets=1,
        hard_neg_per_pos=4,
    )
    meta = json.loads((out_dir / "locomo_meta.json").read_text())
    assert meta["base_model"] == "some-model"
    assert meta["train_conv_ids"] == [0, 1, 2]
    assert meta["n_triplets"] == 1
    assert meta["hard_neg_per_pos"] == 4
    assert len(meta["triplet_sha256_short"]) == 16
    assert meta["loss"] == "MultipleNegativesRankingLoss"


def test_train_aborts_when_no_triplets(tmp_path):
    with pytest.raises(SystemExit):
        fte.train(
            triplets=[],
            base_model_name="x",
            output_dir=tmp_path,
            epochs=1,
            batch_size=4,
        )
