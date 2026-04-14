"""Tests for verbosity.analyze_query_complexity — adaptive detail level."""

from __future__ import annotations

import pytest


def test_single_short_word_is_compact():
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("go") == "compact"
    assert analyze_query_complexity("auth") == "compact"


def test_bare_keyword_tuple_is_compact():
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("jwt auth") == "compact"


def test_medium_question_is_summary():
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("how to set up jwt auth") == "summary"


def test_long_query_is_full():
    from verbosity import analyze_query_complexity
    q = "how should I configure jwt auth with refresh tokens and redis session store in a multi-service setup"
    assert analyze_query_complexity(q) == "full"


def test_query_with_file_path_is_full():
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("fix bug in /src/auth.py") == "full"


def test_query_with_line_number_is_full():
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("error at line 42") == "full"


def test_query_with_url_is_full():
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("https://docs.example.com/api") == "full"


def test_query_with_code_symbol_is_full():
    """Identifiers with :: or () indicate code-specific queries."""
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("what does save_knowledge() do") == "full"
    assert analyze_query_complexity("std::vector::push_back usage") == "full"


def test_empty_query_is_compact():
    from verbosity import analyze_query_complexity
    assert analyze_query_complexity("") == "compact"
    assert analyze_query_complexity("   ") == "compact"


def test_multiple_clauses_with_and_is_full():
    from verbosity import analyze_query_complexity
    q = "auth setup and redis caching and migration strategy"
    assert analyze_query_complexity(q) == "full"
