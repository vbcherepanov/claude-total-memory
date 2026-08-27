"""Unit tests for the BEAM retrieval harness.

The corpus itself is gitignored (CC BY-SA 4.0, downloaded on demand), so the
tests target the parsing and scoring logic — which is where a silent mistake
would quietly invalidate a published number. The one test that needs the
corpus skips when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import beam_bench as bb  # noqa: E402


# ── source_chat_ids comes in several shapes ─────────────────────────


def test_gold_ids_reads_a_plain_list():
    assert bb.gold_ids({"source_chat_ids": [28]}) == {28}
    assert bb.gold_ids({"source_chat_ids": [3, 7, 7]}) == {3, 7}


def test_gold_ids_flattens_the_keyed_form():
    """Temporal questions use {'first_event': [2], 'second_event': [0]}."""
    probe = {"source_chat_ids": {"first_event": [2], "second_event": [0, 5]}}
    assert bb.gold_ids(probe) == {0, 2, 5}


def test_gold_ids_accepts_numeric_strings_and_scalars():
    assert bb.gold_ids({"source_chat_ids": "12"}) == {12}
    assert bb.gold_ids({"source_chat_ids": 9}) == {9}


def test_gold_ids_is_empty_when_there_is_no_evidence():
    """Abstention probes carry none — they must not be scored as recall."""
    assert bb.gold_ids({}) == set()
    assert bb.gold_ids({"source_chat_ids": None}) == set()
    assert bb.gold_ids({"source_chat_ids": []}) == set()
    assert bb.gold_ids({"source_chat_ids": "not-an-id"}) == set()


# ── tag round-trip ──────────────────────────────────────────────────


def test_msg_tag_survives_the_store_lowercasing_tags():
    tag = bb.msg_tag("7", 42)
    entry = {"tags": [tag.upper(), "user", "s3"]}
    assert bb.extract_msg_tags(entry) == [tag.lower()]


def test_extract_msg_tags_parses_json_encoded_tags():
    entry = {"tags": '["beam3:11", "assistant"]'}
    assert bb.extract_msg_tags(entry) == ["beam3:11"]


def test_extract_msg_tags_ignores_unrelated_and_malformed_tags():
    assert bb.extract_msg_tags({"tags": ["user", "s0", "beam-no-colon"]}) == []
    assert bb.extract_msg_tags({"tags": "{not json"}) == []
    assert bb.extract_msg_tags({}) == []


# ── report shape ────────────────────────────────────────────────────


def test_format_report_renders_every_ability_and_the_total():
    stats = {
        "per_ability": {
            "information_extraction": {
                "n": 4, "R@1": 0.75, "R@5": 1.0, "R@10": 1.0, "MRR": 0.875,
            },
            "temporal_reasoning": {
                "n": 4, "R@1": 0.5, "R@5": 1.0, "R@10": 1.0, "MRR": 0.613,
            },
        },
        "overall": {"n": 8, "R@1": 0.625, "R@5": 1.0, "R@10": 1.0, "MRR": 0.744},
        "latency": {"p50_ms": 26.0, "p95_ms": 47.3, "mean_ms": 27.7, "queries": 10},
        "abstention": {"n": 2, "mean_top_score": 1.9, "note": ""},
    }
    report = bb.format_report("100K", {"chats": 2, "saved": 388}, stats)
    assert "information_extraction" in report
    assert "temporal_reasoning" in report
    assert "OVERALL" in report
    assert "100K" in report


def test_missing_dataset_fails_with_a_download_hint(monkeypatch, tmp_path):
    """An operator who has not fetched the corpus needs the curl line."""
    monkeypatch.setattr(bb, "DATA_DIR", tmp_path / "empty")
    with pytest.raises(SystemExit) as exc:
        bb.load_dataset("100K")
    message = str(exc.value)
    assert "huggingface.co/datasets/Mohammadta/BEAM" in message
    assert "100K-00000-of-00001.parquet" in message


# ── the one test that needs the corpus ──────────────────────────────


@pytest.mark.skipif(
    not bb.dataset_path("100K").is_file(),
    reason="BEAM corpus absent (benchmarks/data is gitignored)",
)
def test_the_real_corpus_parses_into_the_expected_shape():
    data = bb.load_dataset("100K")
    assert len(data["conversation_id"]) == 20

    probes = bb.parse_probes(data["probing_questions"][0])
    assert bb.NO_EVIDENCE_ABILITY in probes
    graded = {a for a in probes if a != bb.NO_EVIDENCE_ABILITY}
    assert len(graded) == 9, f"expected 9 graded abilities, got {sorted(graded)}"

    for ability in graded:
        for probe in probes[ability]:
            assert probe["question"].strip()
            assert bb.gold_ids(probe), f"{ability} probe has no gold evidence"
