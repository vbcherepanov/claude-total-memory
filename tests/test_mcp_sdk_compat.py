"""MCP SDK era compatibility — server must import and serve under 1.x and 2.x.

Background: mcp 2.0 (protocol revision 2026-07-28) removed the
``@Server.list_tools()`` / ``@Server.call_tool()`` decorators in favour of
``Server.add_request_handler(method, params_model, handler)``. Because
``requirements.txt`` only floored the dependency, every fresh install started
resolving 2.x and died at import with::

    AttributeError: 'Server' object has no attribute 'list_tools'

These tests pin the contract from both directions: the module must import
against whichever SDK is installed, and the era-specific registration path
must produce working ``tools/list`` and ``tools/call`` handlers.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import server  # noqa: E402


def test_module_imports_and_reports_sdk_era():
    """The import itself is the regression — it used to raise AttributeError."""
    assert server.MCP_SDK_ERA in ("1.x", "2.x")


def test_tools_are_registered_for_the_installed_sdk():
    """Whatever the era, the server must actually answer tools/list + tools/call."""
    if server.MCP_SDK_ERA == "2.x":
        registered = set(server.app._request_handlers)
        assert {"tools/list", "tools/call"} <= registered
    else:
        # 1.x keeps decorator-registered handlers keyed by request type.
        assert server.app.request_handlers, "no request handlers registered"


def test_list_tools_returns_the_full_catalogue():
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert len(tools) > 50, f"expected the full tool catalogue, got {len(tools)}"
    # Spot-check the tools every client contract depends on.
    assert {"memory_recall", "memory_save", "session_init", "session_end"} <= names


def test_every_tool_has_an_object_input_schema():
    """MCP requires inputSchema roots to be objects on every revision."""
    for tool in asyncio.run(server.list_tools()):
        # 1.x exposes `inputSchema`, 2.x renamed the field to `input_schema`
        # and keeps the camelCase form only as a serialization alias.
        schema = getattr(tool, "input_schema", None)
        if schema is None:
            schema = tool.inputSchema
        assert schema.get("type") == "object", tool.name


def test_call_tool_impl_flags_errors_instead_of_raising():
    """Unknown tools come back as isError results, not protocol exceptions."""
    content, is_error = asyncio.run(
        server._call_tool_impl("definitely_not_a_tool", {})
    )
    assert is_error is True
    assert content and content[0].text.startswith("Error:")


@pytest.mark.skipif(
    server.MCP_SDK_ERA != "2.x", reason="2.x-only handler wiring"
)
def test_2x_call_tool_handler_builds_a_call_tool_result():
    from mcp.types import CallToolRequestParams

    handler = server.app._request_handlers["tools/call"].handler
    params = CallToolRequestParams(name="definitely_not_a_tool", arguments={})
    result = asyncio.run(handler(None, params))
    assert result.is_error is True
    assert result.content[0].text.startswith("Error:")


@pytest.mark.skipif(
    server.MCP_SDK_ERA != "2.x", reason="2.x-only handler wiring"
)
def test_2x_list_tools_handler_builds_a_list_tools_result():
    handler = server.app._request_handlers["tools/list"].handler
    result = asyncio.run(handler(None, None))
    assert len(result.tools) > 50


# ────────────────────────────────────────────────────────────────────
# Tool annotations (MCP behaviour hints)
# ────────────────────────────────────────────────────────────────────


def test_every_tool_carries_behaviour_annotations():
    """Clients gate auto-approval on these; an unannotated tool is a gap."""
    for tool in asyncio.run(server.list_tools()):
        ann = tool.annotations
        assert ann is not None, f"{tool.name} has no annotations"
        assert ann.read_only_hint is not None, tool.name
        assert ann.destructive_hint is not None, tool.name
        assert ann.idempotent_hint is not None, tool.name
        # Memory acts only on this machine's own store.
        assert ann.open_world_hint is False, tool.name


def test_read_only_tools_are_never_marked_destructive():
    for tool in asyncio.run(server.list_tools()):
        if tool.annotations.read_only_hint:
            assert tool.annotations.destructive_hint is False, tool.name


def test_classification_tables_match_the_catalogue():
    """A table entry naming a tool that no longer exists is a silent no-op."""
    names = {t.name for t in asyncio.run(server.list_tools())}
    for label, table in (
        ("read-only", server._READ_ONLY_TOOLS),
        ("destructive", server._DESTRUCTIVE_TOOLS),
        ("idempotent", server._IDEMPOTENT_TOOLS),
    ):
        stale = table - names
        assert not stale, f"{label} table names unknown tools: {sorted(stale)}"


def test_the_obviously_destructive_tools_are_flagged():
    by_name = {t.name: t for t in asyncio.run(server.list_tools())}
    for name in ("memory_delete", "memory_forget"):
        ann = by_name[name].annotations
        assert ann.read_only_hint is False, name
        assert ann.destructive_hint is True, name
    for name in ("memory_recall", "memory_get", "memory_stats"):
        assert by_name[name].annotations.read_only_hint is True, name
