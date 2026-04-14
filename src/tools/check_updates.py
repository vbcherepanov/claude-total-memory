"""Check for updates and (optionally) notify the user.

Three sources, tried in order:
  1. GitHub releases   — if `UPDATE_GH_REPO` set (e.g. "owner/repo")
  2. Local git remote  — if repo has a tracked upstream
  3. Skip silently     — no remote configured

Notifies via macOS `osascript display notification` when a newer version is
available. Designed for cron / LaunchAgent (e.g. weekly).

Usage:
    ~/claude-memory-server/.venv/bin/python src/tools/check_updates.py [--apply]

  --apply       — invoke `update.sh` on its own when a new version is found
                  (default: notify only)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _log(msg: str) -> None:
    sys.stderr.write(f"[check-updates] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
    sys.stderr.flush()


def current_version() -> str:
    try:
        from version import VERSION
        return VERSION
    except Exception:
        return "0.0.0"


def latest_from_github(repo: str) -> str | None:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        tag = (data.get("tag_name") or "").lstrip("v").strip()
        return tag or None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


def latest_from_git_remote() -> str | None:
    if not (ROOT / ".git").is_dir():
        return None
    try:
        subprocess.run(
            ["git", "fetch", "--tags", "--quiet"], cwd=ROOT, check=True, timeout=10
        )
        out = subprocess.run(
            ["git", "tag", "--sort=-version:refname"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip().splitlines()
        return out[0].lstrip("v") if out else None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _vparse(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def is_newer(remote: str, local: str) -> bool:
    return _vparse(remote) > _vparse(local)


def _osa_escape(s: str) -> str:
    """Escape a string for safe injection into AppleScript literal.

    Removes NUL bytes, escapes backslash + double-quote. Also strips any
    control chars < 0x20 (except tab) because AppleScript parses them.
    Paranoia: this input can come from an attacker-controlled GitHub
    release `tag_name` or env var in a shared CI.
    """
    if s is None:
        return ""
    s = s.replace("\0", "").replace("\\", "\\\\").replace('"', '\\"')
    s = "".join(c for c in s if c == "\t" or ord(c) >= 0x20)
    return s


def notify(title: str, message: str) -> None:
    try:
        safe_title = _osa_escape(title)
        safe_msg = _osa_escape(message)
        script = (
            f'display notification "{safe_msg}" '
            f'with title "{safe_title}" sound name "Glass"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            check=False, timeout=5,
        )
    except Exception:
        pass


def main() -> int:
    apply = "--apply" in sys.argv

    cur = current_version()
    _log(f"current: v{cur}")

    remote: str | None = None
    repo = os.environ.get("UPDATE_GH_REPO", "").strip()
    if repo:
        remote = latest_from_github(repo)
        _log(f"github {repo}: latest = {remote or 'unknown'}")
    if remote is None:
        remote = latest_from_git_remote()
        if remote:
            _log(f"git remote tag: {remote}")

    if not remote:
        _log("no remote source configured — skipping")
        return 0

    if not is_newer(remote, cur):
        _log(f"up to date (cur=v{cur}, remote=v{remote})")
        return 0

    msg = f"v{cur} → v{remote} available"
    _log(msg)
    notify("claude-total-memory update", msg)

    if apply:
        _log("running update.sh")
        try:
            subprocess.run(
                ["bash", str(ROOT / "update.sh")],
                cwd=ROOT, check=True, timeout=600,
            )
            _log("update.sh succeeded")
        except subprocess.SubprocessError as e:
            _log(f"update.sh failed: {e}")
            return 1
    else:
        _log("notify-only mode (use --apply to run update.sh)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
