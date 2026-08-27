#!/usr/bin/env python3
"""Launch the total-agent-memory MCP server from the Claude Code plugin.

The plugin ships source, not a Python environment, so this picks the first
runnable server it can find and `exec`s it — stdio is handed straight to
Claude Code, so this process must be replaced, never wrapped.

Resolution order, cheapest and most predictable first:

1. An existing install's venv (``$TAM_HOME/.venv``, ``~/.tam/.venv``,
   ``~/.claude-memory/.venv``) — the common case for anyone who ran
   ``install.sh``, and the only branch that touches no network.
2. ``total-agent-memory`` already on PATH (pipx, brew, a global pip).
3. The plugin's own checkout, when its dependencies happen to be importable.
4. ``uvx`` / ``npx``, which fetch a pinned release on first run.
5. A venv this script creates under the memory dir and installs into.

Every branch ends in exec, so the MCP client sees one process either way.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = "total-agent-memory"
PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def log(message: str) -> None:
    """Diagnostics go to stderr — stdout is the MCP frame channel."""
    sys.stderr.write(f"[tam-plugin] {message}\n")


def _exe(venv: Path, name: str) -> Path:
    bindir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv / bindir / f"{name}{suffix}"


def _memory_home() -> Path:
    for env in ("TAM_MEMORY_DIR", "CLAUDE_MEMORY_DIR"):
        value = os.environ.get(env)
        if value:
            return Path(value).expanduser()
    tam = Path.home() / ".tam"
    if tam.exists():
        return tam
    legacy = Path.home() / ".claude-memory"
    return legacy if legacy.exists() else tam


def _candidate_venvs() -> list[Path]:
    seen: list[Path] = []
    for path in (
        _memory_home() / ".venv",
        Path.home() / ".tam" / ".venv",
        Path.home() / ".claude-memory" / ".venv",
        PLUGIN_ROOT / ".venv",
    ):
        if path not in seen:
            seen.append(path)
    return seen


def _exec(argv: list[str]) -> None:
    """Replace this process. Returns only if the command cannot start."""
    log(f"starting: {' '.join(argv)}")
    if os.name == "nt":
        # Windows has no exec that preserves the console handles the way MCP
        # needs, so mirror the child's exit code instead.
        raise SystemExit(subprocess.call(argv))
    os.execv(argv[0], argv)


def _from_existing_venv() -> list[str] | None:
    for venv in _candidate_venvs():
        entry = _exe(venv, "total-agent-memory")
        if entry.is_file():
            return [str(entry)]
        python = _exe(venv, "python")
        server = PLUGIN_ROOT / "src" / "server.py"
        if python.is_file() and server.is_file():
            return [str(python), str(server)]
    return None


def _from_path() -> list[str] | None:
    found = shutil.which("total-agent-memory") or shutil.which("tam")
    return [found] if found else None


def _from_plugin_checkout() -> list[str] | None:
    """Use the shipped source when this interpreter can already import mcp."""
    server = PLUGIN_ROOT / "src" / "server.py"
    if not server.is_file():
        return None
    probe = subprocess.run(
        [sys.executable, "-c", "import mcp"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    return [sys.executable, str(server)]


def _from_runner() -> list[str] | None:
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, PACKAGE]
    npx = shutil.which("npx")
    if npx:
        return [npx, "-y", PACKAGE]
    return None


def _bootstrap_venv() -> list[str] | None:
    """Last resort: build a venv under the memory dir and install into it."""
    venv = _memory_home() / ".venv"
    log(f"no runnable install found — creating {venv}")
    try:
        venv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [str(_exe(venv, "python")), "-m", "pip", "install", "--quiet",
             "--upgrade", PACKAGE],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", b"") or b""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        log(f"bootstrap failed: {e}. {detail.strip()[:400]}")
        return None
    entry = _exe(venv, "total-agent-memory")
    return [str(entry)] if entry.is_file() else None


def main() -> int:
    for resolve in (
        _from_existing_venv,
        _from_path,
        _from_plugin_checkout,
        _from_runner,
        _bootstrap_venv,
    ):
        argv = resolve()
        if argv:
            _exec(argv + sys.argv[1:])
            return 0  # only reached if exec failed outright

    log(
        "could not start the memory server. Install it with one of:\n"
        "  pipx install total-agent-memory\n"
        "  uvx total-agent-memory\n"
        "  npx -y total-agent-memory\n"
        "  curl -fsSL https://get.totalmemory.dev | sh"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
