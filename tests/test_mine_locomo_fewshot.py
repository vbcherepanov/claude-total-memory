"""v9.0 D6 — few-shot pair mining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import mine_locomo_fewshot as mine  # noqa: E402


def _toy_dataset(tmp_path: Path) -> Path:
    payload = []
    for i in range(4):
        qa = []
        for cat in (1, 2, 3, 4, 5):
            for k in range(5):
                qa.append({
                    "question": f"Question_{i}_c{cat}_n{k}?",
                    "answer": "x" * (k + 1),
                    "category": cat,
                    "evidence": [],
                })
        payload.append({
            "sample_id": f"s{i}",
            "conversation": {},
            "qa": qa,
        })
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(payload))
    return p


def test_parse_conv_ids_default():
    assert mine._parse_conv_ids(None, 10) == [0, 1, 2, 3, 4, 5, 6]
    assert mine._parse_conv_ids(None, 4) == [0, 1]  # 70% rounded down → 2
    assert mine._parse_conv_ids("1,3", 10) == [1, 3]


def test_short_answer_score_prefers_short():
    short = mine._short_answer_score("yes")
    long_ = mine._short_answer_score("a much longer answer with many many many words")
    assert short < long_


def test_dedupe_by_question_prefix_keeps_diverse():
    pairs = [
        {"q": "Where did Alice go?", "a": "Paris"},
        {"q": "Where did Alice live?", "a": "London"},  # same prefix → drop
        {"q": "When did Bob meet Carol?", "a": "May"},
        {"q": "What does Alice prefer?", "a": "Tea"},
    ]
    out = mine._dedupe_by_question_prefix(pairs, k=3)
    qs = [p["q"] for p in out]
    assert "Where did Alice go?" in qs
    assert "Where did Alice live?" not in qs
    assert "When did Bob meet Carol?" in qs
    assert len(out) == 3


def test_dedupe_tops_up_when_filter_too_aggressive():
    """If we ask for more than diversity allows, we still get k items."""
    pairs = [
        {"q": "Where Alice", "a": "p1"},
        {"q": "Where Alice", "a": "p2"},
        {"q": "Where Alice", "a": "p3"},
    ]
    out = mine._dedupe_by_question_prefix(pairs, k=2)
    assert len(out) == 2


def test_mine_returns_per_category_buckets(tmp_path):
    ds = _toy_dataset(tmp_path)
    result = mine.mine(ds, train_conv_ids=[0, 1], n_per_category=3)
    assert set(result.keys()) >= {"1", "2", "3", "4", "5", "_meta"}
    for cat in ("1", "2", "3", "4", "5"):
        assert len(result[cat]) <= 3
    assert result["_meta"]["n_per_category"] == 3
    assert result["_meta"]["train_conv_ids"] == [0, 1]


def test_mine_skips_held_out_conversations(tmp_path):
    ds = _toy_dataset(tmp_path)
    result = mine.mine(ds, train_conv_ids=[0], n_per_category=20)
    # Only conv 0 has 5 cats × 5 pairs = 25 questions; but mining picks per-cat.
    for cat in ("1", "2", "3", "4", "5"):
        # Question prefix of conv 0 starts with "Question_0_..."
        for p in result[cat]:
            assert p["q"].startswith("Question_0_")


def test_mine_skips_qa_without_required_fields(tmp_path):
    payload = [{
        "sample_id": "x",
        "conversation": {},
        "qa": [
            {"question": "ok?", "answer": "y", "category": 1},
            {"question": "", "answer": "no question", "category": 1},
            {"question": "no answer?", "answer": "", "category": 2},
        ],
    }]
    ds = tmp_path / "tiny.json"
    ds.write_text(json.dumps(payload))
    result = mine.mine(ds, train_conv_ids=[0], n_per_category=10)
    # cat 1 should have exactly the one valid pair, cat 2 should be empty.
    assert len(result["1"]) == 1
    assert result["2"] == []
