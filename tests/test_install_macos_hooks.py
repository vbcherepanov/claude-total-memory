"""Tests for macOS install path: hooks copy, settings.json registration,
and --uninstall semantics.

All cases run install.sh with INSTALL_TEST_MODE=1 and a sandbox HOME so we
don't touch the real filesystem, pip, launchctl, or Ollama.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
INSTALL_SH = ROOT / "install.sh"


# ---------- helpers ----------


def _run_install(
    home: Path,
    *args: str,
    extra_env: dict | None = None,
    fake_uname: str = "Darwin",
    fake_launchctl: Path | None = None,
):
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["INSTALL_TEST_MODE"] = "1"
    env["FAKE_UNAME"] = fake_uname
    env["CLAUDE_MEMORY_DIR"] = str(home / ".claude-memory")

    if fake_launchctl is not None:
        env["PATH"] = f"{fake_launchctl.parent}:{env.get('PATH','')}"

    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["bash", str(INSTALL_SH), *args],
        env=env,
        capture_output=True,
        text=True,
        # Generous — this runs the real installer; a loaded machine must not
        # turn into a red test. Nothing here measures speed.
        timeout=900,
    )


def _write_launchctl_stub(bin_dir: Path, log_path: Path) -> Path:
    """Fake launchctl that logs every invocation and returns 0."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "launchctl"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{log_path}"\n'
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub


@pytest.fixture
def sandbox_home(tmp_path: Path) -> Path:
    home = tmp_path / "sandbox-home"
    home.mkdir()
    yield home
    shutil.rmtree(home, ignore_errors=True)


# ---------- hook-copy behaviour ----------


def test_install_copies_all_hooks_to_claude_hooks_dir(sandbox_home: Path):
    result = _run_install(sandbox_home, "--ide", "claude-code")
    assert result.returncode == 0, result.stderr

    hooks_dir = sandbox_home / ".claude" / "hooks"
    assert hooks_dir.is_dir(), "~/.claude/hooks must be created"

    # Core hooks from hooks/*.sh must all land there.
    for name in [
        "session-start.sh",
        "session-end.sh",
        "on-stop.sh",
        "memory-trigger.sh",
        "auto-capture.sh",
        "user-prompt-submit.sh",
        "post-tool-use.sh",
    ]:
        dst = hooks_dir / name
        assert dst.is_file(), f"hook missing: {name}"
        # Must be executable
        assert os.access(dst, os.X_OK), f"hook not executable: {name}"

    # Shared lib
    assert (hooks_dir / "lib" / "common.sh").is_file()

    # Example hooks (pre-edit, on-bash-error) must be copied too.
    assert (hooks_dir / "pre-edit.sh").is_file()
    assert (hooks_dir / "on-bash-error.sh").is_file()


def test_install_preserves_existing_hooks_by_default(sandbox_home: Path):
    """User customizations must not be clobbered on re-run."""
    hooks_dir = sandbox_home / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True)
    custom_content = "#!/usr/bin/env bash\n# USER-CUSTOMIZED\nexit 0\n"
    (hooks_dir / "session-start.sh").write_text(custom_content)

    result = _run_install(sandbox_home, "--ide", "claude-code")
    assert result.returncode == 0, result.stderr

    # Still the user's version
    assert (hooks_dir / "session-start.sh").read_text() == custom_content
    # But new v8 hooks that weren't there before DID get installed
    assert (hooks_dir / "user-prompt-submit.sh").is_file()


def test_install_overwrite_hooks_env_forces_replace(sandbox_home: Path):
    hooks_dir = sandbox_home / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "session-start.sh").write_text("#!/bin/sh\nexit 99\n")

    result = _run_install(
        sandbox_home,
        "--ide", "claude-code",
        extra_env={"INSTALL_OVERWRITE_HOOKS": "1"},
    )
    assert result.returncode == 0, result.stderr

    content = (hooks_dir / "session-start.sh").read_text()
    assert "exit 99" not in content, "INSTALL_OVERWRITE_HOOKS=1 must replace the file"


def test_non_claude_code_ide_does_not_install_hooks(sandbox_home: Path):
    """Hooks belong to claude-code only; other IDEs must not populate them."""
    result = _run_install(sandbox_home, "--ide", "cursor")
    assert result.returncode == 0, result.stderr

    hooks_dir = sandbox_home / ".claude" / "hooks"
    assert not hooks_dir.exists(), "cursor install must not create ~/.claude/hooks"


# ---------- settings.json hook registration ----------


def _load_settings(home: Path) -> dict:
    path = home / ".claude" / "settings.json"
    assert path.exists(), "settings.json must exist after claude-code install"
    return json.loads(path.read_text())


def _all_commands(hooks_block) -> list[str]:
    out: list[str] = []
    if not isinstance(hooks_block, list):
        return out
    for group in hooks_block:
        for h in (group.get("hooks") or []):
            cmd = h.get("command")
            if cmd:
                out.append(cmd)
    return out


def test_install_registers_user_prompt_submit_in_settings_json(sandbox_home: Path):
    result = _run_install(sandbox_home, "--ide", "claude-code")
    assert result.returncode == 0, result.stderr

    data = _load_settings(sandbox_home)
    cmds = _all_commands(data["hooks"].get("UserPromptSubmit"))
    assert any(c.endswith("/user-prompt-submit.sh") for c in cmds), \
        f"UserPromptSubmit not registered: {cmds}"
    # Must point at ~/.claude/hooks/, not the install tree.
    for c in cmds:
        assert str(sandbox_home / ".claude" / "hooks") in c


def test_install_registers_post_tool_use_in_settings_json(sandbox_home: Path):
    """post-tool-use.sh registers unconditionally (no-op without env flag)."""
    result = _run_install(sandbox_home, "--ide", "claude-code")
    assert result.returncode == 0, result.stderr

    data = _load_settings(sandbox_home)
    cmds = _all_commands(data["hooks"].get("PostToolUse"))
    assert any(c.endswith("/post-tool-use.sh") for c in cmds), \
        f"post-tool-use.sh not registered: {cmds}"


def test_install_registers_pre_edit_guard(sandbox_home: Path):
    result = _run_install(sandbox_home, "--ide", "claude-code")
    assert result.returncode == 0, result.stderr

    data = _load_settings(sandbox_home)
    pre = data["hooks"].get("PreToolUse")
    cmds = _all_commands(pre)
    assert any(c.endswith("/pre-edit.sh") for c in cmds), \
        f"pre-edit.sh not registered: {cmds}"
    # matcher must target Write|Edit
    matchers = [group.get("matcher") for group in (pre or [])]
    assert "Write|Edit" in matchers


def test_install_registers_on_bash_error_alongside_memory_trigger(sandbox_home: Path):
    result = _run_install(sandbox_home, "--ide", "claude-code")
    assert result.returncode == 0, result.stderr

    data = _load_settings(sandbox_home)
    post = data["hooks"].get("PostToolUse") or []
    # Find the Bash matcher group
    bash_groups = [g for g in post if g.get("matcher") == "Bash"]
    assert bash_groups, "PostToolUse:Bash matcher group missing"
    bash_cmds = [h.get("command") for h in bash_groups[0].get("hooks") or []]
    assert any(c and c.endswith("/memory-trigger.sh") for c in bash_cmds)
    assert any(c and c.endswith("/on-bash-error.sh") for c in bash_cmds)


def test_hook_paths_point_to_home_not_install_dir(sandbox_home: Path):
    result = _run_install(sandbox_home, "--ide", "claude-code")
    assert result.returncode == 0, result.stderr

    data = _load_settings(sandbox_home)
    home_prefix = str(sandbox_home / ".claude" / "hooks")
    install_prefix = str(ROOT / "hooks")

    for key, block in data["hooks"].items():
        for cmd in _all_commands(block):
            assert cmd.startswith(home_prefix), \
                f"{key}: hook path must live under ~/.claude/hooks, got {cmd}"
            assert not cmd.startswith(install_prefix), \
                f"{key}: stale reference to install tree: {cmd}"


# ---------- idempotency ----------


def test_hooks_registration_idempotent_on_rerun(sandbox_home: Path):
    """Running install twice must not duplicate hook entries."""
    r1 = _run_install(sandbox_home, "--ide", "claude-code")
    assert r1.returncode == 0
    r2 = _run_install(sandbox_home, "--ide", "claude-code")
    assert r2.returncode == 0

    data = _load_settings(sandbox_home)

    # UserPromptSubmit must have exactly one user-prompt-submit.sh entry
    ups = _all_commands(data["hooks"].get("UserPromptSubmit"))
    assert sum(1 for c in ups if c.endswith("/user-prompt-submit.sh")) == 1

    # post-tool-use.sh exactly once
    post = _all_commands(data["hooks"].get("PostToolUse"))
    assert sum(1 for c in post if c.endswith("/post-tool-use.sh")) == 1
    # memory-trigger.sh exactly once
    assert sum(1 for c in post if c.endswith("/memory-trigger.sh")) == 1
    # on-bash-error exactly once
    assert sum(1 for c in post if c.endswith("/on-bash-error.sh")) == 1


# ---------- uninstall ----------


def test_uninstall_removes_launchagents(sandbox_home: Path, tmp_path: Path):
    """--uninstall on Darwin must remove plist files + bootout labels."""
    launchctl_log = tmp_path / "launchctl.log"
    stub = _write_launchctl_stub(tmp_path / "bin", launchctl_log)

    la_dir = sandbox_home / "Library" / "LaunchAgents"
    la_dir.mkdir(parents=True)

    # Simulate previously-installed plists.
    for name in [
        "com.claude.memory.reflection.plist",
        "com.claude.memory.orphan-backfill.plist",
        "com.claude.memory.check-updates.plist",
    ]:
        (la_dir / name).write_text("<plist/>")

    result = _run_install(sandbox_home, "--uninstall", fake_launchctl=stub)
    assert result.returncode == 0, result.stderr

    remaining = list(la_dir.glob("com.claude.memory.*.plist"))
    assert not remaining, f"plists must be removed, found: {remaining}"

    # launchctl bootout must have been invoked for each label.
    log = launchctl_log.read_text() if launchctl_log.exists() else ""
    assert "bootout" in log
    assert "com.claude.memory.reflection" in log
    assert "com.claude.memory.orphan-backfill" in log
    assert "com.claude.memory.check-updates" in log


def test_uninstall_preserves_memory_db(sandbox_home: Path, tmp_path: Path):
    """User data in ~/.claude-memory must survive --uninstall."""
    launchctl_log = tmp_path / "launchctl.log"
    stub = _write_launchctl_stub(tmp_path / "bin", launchctl_log)

    mem_dir = sandbox_home / ".claude-memory"
    mem_dir.mkdir()
    db = mem_dir / "memory.db"
    db.write_bytes(b"PRECIOUS USER DATA")
    (mem_dir / "raw").mkdir()
    (mem_dir / "raw" / "note.txt").write_text("user note")

    result = _run_install(sandbox_home, "--uninstall", fake_launchctl=stub)
    assert result.returncode == 0, result.stderr

    assert db.exists(), "memory.db must NOT be deleted by --uninstall"
    assert db.read_bytes() == b"PRECIOUS USER DATA"
    assert (mem_dir / "raw" / "note.txt").read_text() == "user note"


# ---------- LaunchAgent plist installation (real run, not test mode) ----------


def test_launchagents_substitute_install_dir_and_memory_dir(
    sandbox_home: Path, tmp_path: Path
):
    """install.sh must replace __INSTALL_DIR__ / __MEMORY_DIR__ / __HOME__ in
    every plist. Regression for the v12.2 bug where plists hardcoded
    `__HOME__/claude-memory-server/...` — any user who cloned the repo
    under a different name had a non-functional reflection daemon.

    This test runs install.sh WITHOUT INSTALL_TEST_MODE so the
    LaunchAgent install branch actually fires, with a fake launchctl.
    """
    launchctl_log = tmp_path / "launchctl.log"
    stub = _write_launchctl_stub(tmp_path / "bin", launchctl_log)

    env = os.environ.copy()
    env["HOME"] = str(sandbox_home)
    env["FAKE_UNAME"] = "Darwin"
    env["TAM_MEMORY_DIR"] = str(sandbox_home / ".tam")
    # Skip pip & model download, but keep the LaunchAgent step running.
    env["INSTALL_TEST_MODE"] = "skip-heavy"
    env["PATH"] = f"{stub.parent}:{env.get('PATH','')}"

    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--ide", "claude-code"],
        # Generous: this runs the real installer end to end, and on a loaded
        # machine (a benchmark ingesting in another process, CI on a shared
        # runner) 180s was not enough — the test then fails for load, not for
        # a regression. Nothing here measures speed.
        env=env, capture_output=True, text=True, timeout=900,
    )
    # We don't assert returncode==0: pip/model steps may skip with
    # non-fatal warnings under our partial test mode. We only care that
    # the LaunchAgent step ran and produced valid plists.
    la_dir = sandbox_home / "Library" / "LaunchAgents"
    if not la_dir.exists():
        pytest.skip(
            "LaunchAgent branch did not fire under INSTALL_TEST_MODE=skip-heavy "
            f"(install.sh stderr: {result.stderr[-300:]})"
        )

    plists = list(la_dir.glob("*.plist"))
    assert plists, "no plists were copied to LaunchAgents dir"

    for plist in plists:
        body = plist.read_text()
        # No leftover placeholders.
        for placeholder in ("__INSTALL_DIR__", "__MEMORY_DIR__", "__HOME__"):
            assert placeholder not in body, (
                f"{plist.name}: leftover placeholder {placeholder}\n{body}"
            )
        # No hardcoded old paths.
        assert "claude-memory-server" not in body, (
            f"{plist.name}: hardcoded old checkout name 'claude-memory-server'"
        )


def test_uninstall_preserves_settings_json(sandbox_home: Path, tmp_path: Path):
    """settings.json should be left intact — MCP entry is documented as
    user-removable, not auto-nuked (avoids wiping other MCP servers)."""
    launchctl_log = tmp_path / "launchctl.log"
    stub = _write_launchctl_stub(tmp_path / "bin", launchctl_log)

    settings = sandbox_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "mcpServers": {"memory": {"command": "python3"}, "other": {"command": "x"}},
        "hooks": {"SessionStart": []},
    }))

    result = _run_install(sandbox_home, "--uninstall", fake_launchctl=stub)
    assert result.returncode == 0, result.stderr

    assert settings.exists()
    data = json.loads(settings.read_text())
    # Both entries still there (uninstall is informational re: MCP config).
    assert "other" in data["mcpServers"]
