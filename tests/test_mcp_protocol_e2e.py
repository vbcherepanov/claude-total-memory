"""End-to-end MCP transport tests — both protocol eras, real subprocess.

The 2026-07-28 revision made MCP stateless: there is no ``initialize``
handshake and no session id. A client instead stamps protocol metadata into
``params._meta`` on every request. Servers are expected to keep serving the
older handshake era from the same process.

These tests spawn ``src/server.py`` exactly the way Claude Code / Codex /
Cursor spawn it and drive both eras over stdio, so a future SDK bump that
breaks either one fails here rather than in a user's install.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src" / "server.py"

mcp_types = pytest.importorskip("mcp.types")

MODERN_ERA = hasattr(mcp_types, "PROTOCOL_VERSION_META_KEY")


def _env(tmp_path: Path) -> dict[str, str]:
    """Isolate the server onto a throwaway memory dir."""
    env = dict(os.environ)
    env.update(
        TAM_MEMORY_DIR=str(tmp_path),
        CLAUDE_MEMORY_DIR=str(tmp_path),
        MEMORY_MODE="fast",
        MEMORY_ASYNC_ENRICHMENT="false",
    )
    return env


@pytest.mark.skipif(not MODERN_ERA, reason="requires mcp SDK >= 2.0")
def test_stateless_2026_era_serves_without_initialize(tmp_path):
    """No handshake, no session id — just enveloped requests."""
    meta = {
        mcp_types.PROTOCOL_VERSION_META_KEY: mcp_types.LATEST_PROTOCOL_VERSION,
        mcp_types.CLIENT_CAPABILITIES_META_KEY: {},
        mcp_types.CLIENT_INFO_META_KEY: {"name": "tam-tests", "version": "1.0.0"},
    }
    frames = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": meta}},
        {"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {"_meta": meta}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"_meta": meta, "name": "memory_stats", "arguments": {}},
        },
    ]
    replies = _drive(frames, tmp_path)

    assert len(replies[1]["result"]["tools"]) > 50
    assert mcp_types.LATEST_PROTOCOL_VERSION in replies[2]["result"]["supportedVersions"]
    call = replies[3]["result"]
    assert call.get("isError") is not True
    assert json.loads(call["content"][0]["text"])["sessions"] >= 1
    # 2026-07-28 allows any JSON value in structuredContent; JSON-answering
    # tools hand clients the parsed object so they stop re-parsing strings.
    assert call["structuredContent"]["sessions"] >= 1


@pytest.mark.skipif(not MODERN_ERA, reason="requires mcp SDK >= 2.0")
def test_error_results_carry_no_structured_content(tmp_path):
    meta = {
        mcp_types.PROTOCOL_VERSION_META_KEY: mcp_types.LATEST_PROTOCOL_VERSION,
        mcp_types.CLIENT_CAPABILITIES_META_KEY: {},
        mcp_types.CLIENT_INFO_META_KEY: {"name": "tam-tests", "version": "1.0.0"},
    }
    replies = _drive(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"_meta": meta, "name": "no_such_tool", "arguments": {}},
            }
        ],
        tmp_path,
    )
    result = replies[1]["result"]
    assert result["isError"] is True
    assert result.get("structuredContent") is None
    assert result["content"][0]["text"].startswith("Error:")


def test_legacy_handshake_era_still_works(tmp_path):
    """Clients pinned to an older SDK must keep working against this server."""
    frames = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "tam-tests", "version": "1.0.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    replies = _drive(frames, tmp_path, expected=2)

    assert "protocolVersion" in replies[1]["result"]
    assert len(replies[2]["result"]["tools"]) > 50


def _drive(frames: list[dict], tmp_path: Path, expected: int | None = None) -> dict:
    """Send `frames` to a fresh server process, return replies keyed by id."""
    want = expected if expected is not None else sum("id" in f for f in frames)
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_env(tmp_path),
        cwd=str(ROOT),
        text=True,
        bufsize=1,
    )
    try:
        for frame in frames:
            proc.stdin.write(json.dumps(frame) + "\n")
        proc.stdin.flush()

        replies: dict = {}
        while len(replies) < want:
            line = proc.stdout.readline()
            if not line:
                raise AssertionError(
                    f"server closed stdout after {len(replies)}/{want} replies"
                )
            msg = json.loads(line)
            if "id" in msg:
                replies[msg["id"]] = msg
        for rid, msg in replies.items():
            assert "error" not in msg, f"id={rid} returned {msg['error']}"
        return replies
    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=15)
