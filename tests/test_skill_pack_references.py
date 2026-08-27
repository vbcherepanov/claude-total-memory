"""The shipped skill pack must not tell users to run things that don't exist.

`skills/memory-protocol/` is copied verbatim into every IDE a user wires up,
and its templates are meant to be pasted into real config files. Until v13 they
referenced `python3 -m claude_total_memory.cli serve --mode stdio` — a module
that has never existed since the rebrand — and the pre-rebrand
`~/claude-memory-server/` install path. Both produce a broken setup, silently,
for anyone following the manual instructions.

Nothing catches that at runtime, so it is checked here.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills" / "memory-protocol"

DOC_SUFFIXES = {".md", ".mdc", ".toml", ".json", ".txt"}
DOCS = sorted(p for p in SKILLS.rglob("*") if p.suffix in DOC_SUFFIXES)


def test_the_skill_pack_is_actually_there():
    assert (SKILLS / "SKILL.md").is_file()
    assert len(DOCS) >= 8, "skill pack looks truncated"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(SKILLS)))
def test_no_reference_to_the_nonexistent_cli_module(doc: Path):
    """`claude_total_memory.cli` is not importable and never was."""
    assert "claude_total_memory.cli" not in doc.read_text(), (
        "use the `total-agent-memory` console script instead"
    )


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(SKILLS)))
def test_no_reference_to_the_pre_rebrand_install_path(doc: Path):
    """The documented checkout is ~/total-agent-memory."""
    text = doc.read_text()
    assert "claude-memory-server" not in text
    assert "claude_memory_server" not in text


def test_the_entry_point_the_templates_name_is_declared():
    """`total-agent-memory` must exist as a console script."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = pyproject["project"]["scripts"]
    assert "total-agent-memory" in scripts
    assert "lookup-memory" in scripts


def test_claude_code_template_is_valid_and_points_at_the_entry_point():
    cfg = json.loads((SKILLS / "templates" / "claude-code-settings.json").read_text())
    assert cfg["mcpServers"]["memory"]["command"] == "total-agent-memory"
    for event, blocks in cfg["hooks"].items():
        for block in blocks:
            for hook in block["hooks"]:
                assert hook["type"] == "command", event
                assert hook["command"].startswith("$HOME/.claude/hooks/"), event


def test_codex_template_is_valid_toml_and_points_at_the_entry_point():
    cfg = tomllib.loads((SKILLS / "templates" / "codex-config.toml").read_text())
    assert cfg["mcp"]["servers"]["memory"]["command"] == "total-agent-memory"
    assert cfg["mcp"]["servers"]["memory"]["args"] == []


def test_lookup_helper_is_referenced_by_its_installed_name():
    """Sub-agents are told to call `lookup-memory`, not a path into a checkout."""
    text = (SKILLS / "references" / "subagent-protocol.md").read_text()
    assert "lookup-memory" in text
