"""Tests for TOML-based content filter pipeline (rtk-style, with whitelist safety)."""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────
# Stage 1 — strip_ansi
# ──────────────────────────────────────────────


def test_strip_ansi_removes_color_codes():
    from content_filter import strip_ansi

    text = "\x1b[31mERROR\x1b[0m: something \x1b[32mgreen\x1b[0m"
    assert strip_ansi(text) == "ERROR: something green"


def test_strip_ansi_handles_no_codes():
    from content_filter import strip_ansi

    assert strip_ansi("plain text") == "plain text"


# ──────────────────────────────────────────────
# Stage 2 — replace
# ──────────────────────────────────────────────


def test_replace_multiple_patterns():
    from content_filter import apply_replace

    text = "foo=1 bar=2 baz=3"
    rules = [{"pattern": r"\w+=\d+", "replacement": "KV"}]
    assert apply_replace(text, rules) == "KV KV KV"


def test_replace_with_backrefs():
    from content_filter import apply_replace

    text = "file /tmp/log.txt:123:error"
    rules = [{"pattern": r"(\S+):(\d+):(\w+)", "replacement": r"\1 (line \2)"}]
    assert apply_replace(text, rules) == "file /tmp/log.txt (line 123)"


# ──────────────────────────────────────────────
# Stage 3-5 — line-level filters
# ──────────────────────────────────────────────


def test_keep_lines_retains_matches_only():
    from content_filter import keep_lines

    text = "pass: test1\nfail: test2\npass: test3\n"
    assert keep_lines(text, [r"^fail"]) == "fail: test2"


def test_strip_lines_removes_matches():
    from content_filter import strip_lines

    text = "pass: test1\nfail: test2\npass: test3\n"
    assert strip_lines(text, [r"^pass"]) == "fail: test2"


def test_strip_and_keep_combined():
    from content_filter import run_pipeline

    text = "info: hello\nerror: boom\ndebug: trace\nwarn: oh"
    out = run_pipeline(text, {"strip_lines": [r"^debug"], "keep_lines": [r"^(error|warn)"]})
    assert "error: boom" in out
    assert "warn: oh" in out
    assert "debug" not in out
    assert "info: hello" not in out


# ──────────────────────────────────────────────
# Stage 6-7 — truncate / head / tail / max_lines
# ──────────────────────────────────────────────


def test_truncate_long_lines():
    from content_filter import truncate_lines

    text = "short\n" + "x" * 500 + "\nok"
    out = truncate_lines(text, max_chars=50)
    lines = out.splitlines()
    assert len(lines[0]) <= 50
    assert len(lines[1]) <= 50
    assert lines[1].endswith("...") or len(lines[1]) == 50


def test_head_takes_first_n():
    from content_filter import head_lines

    text = "\n".join(f"line {i}" for i in range(10))
    assert head_lines(text, 3) == "line 0\nline 1\nline 2"


def test_tail_takes_last_n():
    from content_filter import tail_lines

    text = "\n".join(f"line {i}" for i in range(10))
    assert tail_lines(text, 3) == "line 7\nline 8\nline 9"


def test_max_lines_caps_total():
    from content_filter import run_pipeline

    text = "\n".join(f"x{i}" for i in range(100))
    out = run_pipeline(text, {"max_lines": 10})
    assert len(out.splitlines()) == 10


# ──────────────────────────────────────────────
# Stage 8 — on_empty fallback
# ──────────────────────────────────────────────


def test_on_empty_returns_placeholder():
    from content_filter import run_pipeline

    text = "noise"
    out = run_pipeline(text, {"strip_lines": [r"."], "on_empty": "(all output filtered)"})
    assert out == "(all output filtered)"


# ──────────────────────────────────────────────
# Whitelist safety (CRITICAL)
# ──────────────────────────────────────────────


def test_whitelist_preserves_urls_through_aggressive_filter():
    """Even if a filter would delete a line, URLs must survive."""
    from content_filter import run_pipeline

    text = "fluff text https://critical.example/api other fluff"
    # Aggressive strip: remove everything
    config = {"strip_lines": [r".*"], "on_empty": "(empty)"}
    out = run_pipeline(text, config, safety="strict")
    # URL preserved somewhere in output (either as-is or via placeholder restoration)
    assert "https://critical.example/api" in out


def test_whitelist_preserves_absolute_paths():
    from content_filter import run_pipeline

    text = "error in /Users/me/project/src/main.py at line 42"
    out = run_pipeline(text, {"max_lines": 0}, safety="strict")  # 0 = keep none
    assert "/Users/me/project/src/main.py" in out


def test_whitelist_preserves_inline_code():
    from content_filter import run_pipeline

    text = "call `save_knowledge()` with `ktype=fact`"
    out = run_pipeline(text, {"strip_lines": [r".*"]}, safety="strict")
    assert "`save_knowledge()`" in out


def test_safety_off_allows_full_loss():
    """When safety='off', whitelist is NOT enforced (user opted in)."""
    from content_filter import run_pipeline

    text = "see https://x.y/z for details"
    out = run_pipeline(text, {"strip_lines": [r".*"]}, safety="off")
    assert "https://" not in out


# ──────────────────────────────────────────────
# Token-saving metrics
# ──────────────────────────────────────────────


def test_filter_reports_reduction():
    from content_filter import filter_with_stats

    text = "\n".join(f"noise line {i}" for i in range(100))
    out, stats = filter_with_stats(
        text, {"max_lines": 10}, safety="strict"
    )
    assert stats["input_chars"] > stats["output_chars"]
    assert stats["input_lines"] == 100
    assert stats["output_lines"] <= 10
    assert 0 < stats["reduction_pct"] < 100


# ──────────────────────────────────────────────
# TOML config loading
# ──────────────────────────────────────────────


def test_load_toml_filter(tmp_path):
    from content_filter import load_filter_config

    toml = tmp_path / "test.toml"
    toml.write_text(
        """
name = "pytest_short"
description = "Keep only fail/error lines"
safety = "strict"

[stages]
strip_ansi = true
keep_lines = ["FAIL", "ERROR", "passed"]
max_lines = 20
on_empty = "(all passed)"
"""
    )
    cfg = load_filter_config(toml)
    assert cfg["name"] == "pytest_short"
    assert cfg["safety"] == "strict"
    assert cfg["stages"]["max_lines"] == 20
    assert "FAIL" in cfg["stages"]["keep_lines"]


def test_pipeline_end_to_end_with_all_stages():
    from content_filter import run_pipeline

    text = (
        "\x1b[31mFAIL\x1b[0m: test_alpha https://docs.example/test\n"
        "PASS: test_beta\n"
        "PASS: test_gamma\n"
        "PASS: test_delta\n"
        "FAIL: test_epsilon (see /tmp/report.txt)\n"
        "DEBUG: internal stuff\n"
    )
    config = {
        "strip_ansi": True,
        "keep_lines": [r"^FAIL"],
        "max_lines": 20,
        "on_empty": "(all tests passed)",
    }
    out = run_pipeline(text, config, safety="strict")
    assert "FAIL: test_alpha" in out
    assert "FAIL: test_epsilon" in out
    assert "PASS" not in out
    assert "DEBUG" not in out
    # Whitelist preserved both URL and path
    assert "https://docs.example/test" in out
    assert "/tmp/report.txt" in out
    # ANSI stripped
    assert "\x1b" not in out
