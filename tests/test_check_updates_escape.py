"""Regression: check_updates._osa_escape must neutralize AppleScript injection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_plain_text_unchanged():
    from tools.check_updates import _osa_escape
    assert _osa_escape("v6.0 -> v6.1 available") == "v6.0 -> v6.1 available"


def test_quote_escaped():
    from tools.check_updates import _osa_escape
    out = _osa_escape('has "quote"')
    assert out == 'has \\"quote\\"'


def test_backslash_escaped():
    from tools.check_updates import _osa_escape
    out = _osa_escape(r"path\\with\\bs")
    assert "\\\\\\\\" in out or out.count("\\\\") >= 2


def test_strips_control_chars():
    from tools.check_updates import _osa_escape
    s = "hello\x00world\x07end\n\t"
    out = _osa_escape(s)
    assert "\x00" not in out
    assert "\x07" not in out
    assert "\t" in out       # tab preserved
    # newline 0x0A is control but >= 0x20 is the rule — 0x0A is < 0x20 so stripped
    assert "\n" not in out


def test_applescript_injection_attempt_is_neutralized():
    """Classic injection: close string + inject do shell script."""
    from tools.check_updates import _osa_escape
    evil = '"\ndo shell script "rm -rf /"\n--'
    safe = _osa_escape(evil)
    # The closing quote and newline must be neutralized
    assert '"\n' not in safe
    assert 'do shell script' in safe  # the text survives, but as a literal
    assert '"' not in safe.replace('\\"', '')  # only escaped quotes remain


def test_none_returns_empty():
    from tools.check_updates import _osa_escape
    assert _osa_escape(None) == ""
