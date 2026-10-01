"""
Unit tests for the generic MCP integration mechanics in ToolRegistry --
not the specific OCR/Recall tools themselves (those have their own test
files), but the plumbing that makes ANY MCP server pluggable:
config-driven enable/disable, runtime discovery, graceful failure
isolation between servers, and correct call routing.

Real MCP clients are mocked so these tests run fast, offline, and
deterministically -- they don't spawn real subprocesses or hit real
network services.
"""

import types
from unittest.mock import AsyncMock, patch

import pytest

from tool_registry import ToolRegistry


def _fake_tool(name, description="A fake tool.", schema=None):
    return types.SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _make_client(tools=None, list_tools_side_effect=None):
    """Builds a mock MCP client with async list_tools()/call_tool()."""
    client = AsyncMock()
    if list_tools_side_effect is not None:
        client.list_tools.side_effect = list_tools_side_effect
    else:
        client.list_tools.return_value = tools or []
    return client


@pytest.fixture
def registry_kwargs(tmp_path):
    return {
        "workspace_root": str(tmp_path / "workspace"),
        "receipt_memory_path": str(tmp_path / "memory.json"),
    }


def test_disabled_server_is_never_instantiated(registry_kwargs):
    """Config-driven enable/disable: a disabled server's client class
    should never even be constructed, let alone connected to."""
    with patch("tool_registry.RecallMCPClient") as MockRecall, \
         patch("tool_registry.OCRMCPClient") as MockOCR, \
         patch("tool_registry.QRMCPClient") as MockQR:

        MockOCR.return_value = _make_client(tools=[])
        MockQR.return_value = _make_client(tools=[])

        ToolRegistry(
            mcp_servers={"recall": {"enabled": False}, "ocr": {"enabled": True, "server_script": None}},
            **registry_kwargs,
        )

        MockRecall.assert_not_called()
        MockOCR.assert_called_once()


def test_enabled_server_discovers_tools_at_runtime(registry_kwargs):
    """Tool discovery is dynamic (list_tools() at startup), not a
    hardcoded list -- schemas() should reflect whatever the mocked
    server reports, proving nothing is baked in."""
    fake_tools = [
        _fake_tool("made_up_tool_one", "First fake capability."),
        _fake_tool("made_up_tool_two", "Second fake capability."),
    ]
    with patch("tool_registry.RecallMCPClient") as MockRecall, \
         patch("tool_registry.OCRMCPClient") as MockOCR, \
         patch("tool_registry.QRMCPClient") as MockQR:

        MockRecall.return_value = _make_client(tools=fake_tools)
        MockOCR.return_value = _make_client(tools=[])
        MockQR.return_value = _make_client(tools=[])

        registry = ToolRegistry(
            mcp_servers={"recall": {"enabled": True}, "ocr": {"enabled": False}},
            **registry_kwargs,
        )

        names = [s["function"]["name"] for s in registry.schemas()]
        assert "made_up_tool_one" in names
        assert "made_up_tool_two" in names


def test_unavailable_server_does_not_crash_or_block_the_other(registry_kwargs):
    """If one MCP server fails to connect (network error, missing
    binary, etc.), ToolRegistry must not crash, and the OTHER server's
    tools must still be available -- one failure should never take down
    the whole registry."""
    with patch("tool_registry.RecallMCPClient") as MockRecall, \
         patch("tool_registry.OCRMCPClient") as MockOCR, \
         patch("tool_registry.QRMCPClient") as MockQR:

        MockRecall.return_value = _make_client(
            list_tools_side_effect=ConnectionError("simulated: server unreachable")
        )
        MockOCR.return_value = _make_client(tools=[_fake_tool("working_tool")])
        MockQR.return_value = _make_client(tools=[])

        # Must not raise, even though the recall client's discovery fails.
        registry = ToolRegistry(
            mcp_servers={"recall": {"enabled": True}, "ocr": {"enabled": True, "server_script": None}},
            **registry_kwargs,
        )

        names = [s["function"]["name"] for s in registry.schemas()]
        assert "working_tool" in names
        assert not any("simulated" in n for n in names)


def test_call_routes_to_the_owning_client_only(registry_kwargs):
    """When several MCP servers are active, invoking one server's tool must
    call THAT server's client and no other."""
    recall_client = _make_client(tools=[_fake_tool("recall_only_tool")])
    recall_client.call_tool.return_value = "result from recall"

    ocr_client = _make_client(tools=[_fake_tool("ocr_only_tool")])
    ocr_client.call_tool.return_value = "result from ocr"

    qr_client = _make_client(tools=[_fake_tool("qr_only_tool")])
    qr_client.call_tool.return_value = "result from qr"

    with patch("tool_registry.RecallMCPClient") as MockRecall, \
         patch("tool_registry.OCRMCPClient") as MockOCR, \
         patch("tool_registry.QRMCPClient") as MockQR:
        MockRecall.return_value = recall_client
        MockOCR.return_value = ocr_client
        MockQR.return_value = qr_client

        registry = ToolRegistry(
            mcp_servers={
                "recall": {"enabled": True},
                "ocr": {"enabled": True, "server_script": None},
                "qr": {"enabled": True},
            },
            **registry_kwargs,
        )

        result = registry.call("recall_only_tool", {})

        assert result == "result from recall"
        recall_client.call_tool.assert_called_once_with("recall_only_tool", {})
        ocr_client.call_tool.assert_not_called()
        qr_client.call_tool.assert_not_called()


def test_mcp_tool_call_exception_returns_clean_error_not_a_crash(registry_kwargs):
    """If invoking the tool itself fails (the server crashed mid-call,
    connection dropped, etc.), call() must return a readable error
    string instead of letting the exception propagate up to the ReAct
    loop and kill the turn."""
    failing_client = _make_client(tools=[_fake_tool("flaky_tool")])
    failing_client.call_tool.side_effect = RuntimeError("simulated mid-call failure")

    with patch("tool_registry.RecallMCPClient") as MockRecall, \
         patch("tool_registry.OCRMCPClient") as MockOCR, \
         patch("tool_registry.QRMCPClient") as MockQR:
        MockRecall.return_value = failing_client
        MockOCR.return_value = _make_client(tools=[])
        MockQR.return_value = _make_client(tools=[])

        registry = ToolRegistry(
            mcp_servers={"recall": {"enabled": True}, "ocr": {"enabled": False}},
            **registry_kwargs,
        )

        result = registry.call("flaky_tool", {})

        assert "Error" in result
        assert "flaky_tool" in result or "simulated mid-call failure" in result


def test_unknown_tool_name_returns_error_not_exception(registry_kwargs):
    with patch("tool_registry.RecallMCPClient") as MockRecall, \
         patch("tool_registry.OCRMCPClient") as MockOCR, \
         patch("tool_registry.QRMCPClient") as MockQR:
        MockRecall.return_value = _make_client(tools=[])
        MockOCR.return_value = _make_client(tools=[])
        MockQR.return_value = _make_client(tools=[])

        registry = ToolRegistry(
            mcp_servers={"recall": {"enabled": False}, "ocr": {"enabled": False}},
            **registry_kwargs,
        )

        result = registry.call("this_tool_does_not_exist", {})
        assert "Error" in result
        assert "unknown tool" in result.lower()
