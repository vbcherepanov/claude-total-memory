"""The Claude Code plugin must stay installable.

Covers the two ways this breaks silently: a manifest that drifts out of sync
with the release (wrong version, dangling file references), and a bootstrap
that cannot find a runnable server.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bin"))

PLUGIN = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
MARKETPLACE = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())


# ── Manifest ────────────────────────────────────────────────────────


def test_plugin_manifest_has_the_required_fields():
    assert PLUGIN["name"] == "total-agent-memory"
    assert PLUGIN["description"].strip()
    # `repository` must be a string — Claude Code rejects the npm-style object.
    assert isinstance(PLUGIN["repository"], str)


def test_plugin_version_tracks_the_package_version():
    """A stale plugin version means users never get the update."""
    from version import VERSION

    assert PLUGIN["version"] == VERSION, (
        f"plugin.json says {PLUGIN['version']}, src/version.py says {VERSION}"
    )


def test_marketplace_lists_this_plugin():
    names = {p["name"] for p in MARKETPLACE["plugins"]}
    assert PLUGIN["name"] in names
    assert MARKETPLACE["owner"]["name"]


# ── Wiring ──────────────────────────────────────────────────────────


def test_mcp_config_points_at_an_existing_bootstrap():
    cfg = json.loads((ROOT / ".mcp.json").read_text())
    server = cfg["mcpServers"]["memory"]
    assert server["command"] == "python3"
    (arg,) = server["args"]
    assert arg.startswith("${CLAUDE_PLUGIN_ROOT}/")
    target = ROOT / arg.replace("${CLAUDE_PLUGIN_ROOT}/", "")
    assert target.is_file(), f"{target} referenced by .mcp.json does not exist"


def test_every_hook_command_exists_and_is_executable():
    cfg = json.loads((ROOT / "hooks" / "hooks.json").read_text())
    seen = 0
    for event, matchers in cfg["hooks"].items():
        for matcher in matchers:
            for hook in matcher["hooks"]:
                command = hook["command"]
                assert command.startswith("${CLAUDE_PLUGIN_ROOT}/"), event
                path = ROOT / command.replace("${CLAUDE_PLUGIN_ROOT}/", "")
                assert path.is_file(), f"{event}: missing {path}"
                assert os.stat(path).st_mode & stat.S_IXUSR, f"{event}: {path} not +x"
                seen += 1
    assert seen >= 7, "expected the full capture hook set"


def test_hook_events_are_real_claude_code_events():
    cfg = json.loads((ROOT / "hooks" / "hooks.json").read_text())
    known = {
        "SessionStart", "SessionEnd", "UserPromptSubmit", "PreToolUse",
        "PostToolUse", "PostToolUseFailure", "Stop", "SubagentStop",
        "PreCompact", "PostCompact", "Notification", "MessageDisplay",
    }
    unknown = set(cfg["hooks"]) - known
    assert not unknown, f"unknown hook events: {sorted(unknown)}"


def test_the_skill_ships_with_the_plugin():
    skill = ROOT / "skills" / "memory-protocol" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text()
    assert text.startswith("---"), "SKILL.md needs YAML frontmatter"
    assert "description:" in text.split("---")[1]


# ── Bootstrap resolution ────────────────────────────────────────────


@pytest.fixture
def bootstrap():
    import tam_plugin_bootstrap

    return tam_plugin_bootstrap


def test_bootstrap_prefers_an_existing_install(bootstrap, tmp_path, monkeypatch):
    venv = tmp_path / ".venv"
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True)
    entry = bindir / ("total-agent-memory.exe" if os.name == "nt" else "total-agent-memory")
    entry.write_text("#!/bin/sh\n")
    monkeypatch.setenv("TAM_MEMORY_DIR", str(tmp_path))

    assert bootstrap._from_existing_venv() == [str(entry)]


def test_bootstrap_reports_no_install_when_there_is_none(bootstrap, tmp_path, monkeypatch):
    monkeypatch.setenv("TAM_MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(bootstrap.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(bootstrap, "PLUGIN_ROOT", tmp_path)

    assert bootstrap._from_existing_venv() is None


def test_bootstrap_honours_the_legacy_memory_dir(bootstrap, tmp_path, monkeypatch):
    monkeypatch.delenv("TAM_MEMORY_DIR", raising=False)
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path / "legacy"))

    assert bootstrap._memory_home() == tmp_path / "legacy"


def test_bootstrap_falls_back_to_a_runner(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: (
        "/usr/local/bin/uvx" if name == "uvx" else None
    ))
    assert bootstrap._from_runner() == ["/usr/local/bin/uvx", "total-agent-memory"]

    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: (
        "/usr/local/bin/npx" if name == "npx" else None
    ))
    assert bootstrap._from_runner() == ["/usr/local/bin/npx", "-y", "total-agent-memory"]

    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: None)
    assert bootstrap._from_runner() is None
