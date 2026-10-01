"""
End-to-end: a scripted LLM drives the real ReAct loop, a real QR MCP server
subprocess, and real ChromaDB memory. No token and no config.json needed, so
this test always runs.

The model is replaced by ScriptedLLM, which returns a fixed sequence of
tool calls and a final answer. Everything below the model is real: the loop,
the ToolRegistry, the QR MCP client/server over stdio, and the long-term
memory store.

Run from the repository root:
    pytest tests/e2e_tests/test_agent_flow.py -v
"""

import pytest

from context import Context
from loop import react_step
from permissions import auto_approve
from main import SYSTEM_PROMPT
from tool_registry import ToolRegistry
from agent.ltm import Memory

zxingcpp = pytest.importorskip(
    "zxingcpp", reason="zxing-cpp is required to decode the generated QR codes"
)
from PIL import Image  # noqa: E402  (imported after the decoder check on purpose)


ASSET_TAG = "Dell XPS 13 - DXP-13-45927"


class ScriptedLLM:
    """Minimal stand-in for LLMClient.

    Returns a preset list of decisions, one per get_next_step() call, and
    records every call so the test can check what tools were offered to the
    model. The decision dicts have the same shape llm_client.py returns.
    """

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls = []

    def get_next_step(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._decisions:
            return {"type": "final_answer", "content": "(script exhausted)"}
        return self._decisions.pop(0)


def test_tool_mcp_memory_flow(tmp_path, shipped_policy):
    """Runs create -> remember_fact -> generate_qr_code (MCP) -> recall_fact
    through the real loop and verifies each effect."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = str(tmp_path / "chroma")

    # No config.json / load_config: the registry is configured directly, and
    # only the QR MCP server is enabled so the test stays offline and needs
    # neither Tesseract nor the CPSC network endpoint.
    memory = Memory(db_path=db_path)
    tools = ToolRegistry(
        workspace_root=str(workspace),
        memory=memory,
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers={
            "recall": {"enabled": False},
            "ocr": {"enabled": False},
            "qr": {"enabled": True},
        },
    )

    llm = ScriptedLLM([
        {"type": "tool_call", "id": "call_1", "tool_name": "create",
         "arguments": {"path": "inventory.md", "type": "file"}},
        {"type": "tool_call", "id": "call_2", "tool_name": "remember_fact",
         "arguments": {"subject": "user", "predicate": "prefers", "value": "metric units"}},
        {"type": "tool_call", "id": "call_3", "tool_name": "generate_qr_code",
         "arguments": {"filename": "asset-tag.png", "payload_type": "text", "text": ASSET_TAG}},
        {"type": "tool_call", "id": "call_4", "tool_name": "recall_fact",
         "arguments": {"query": "units"}},
        {"type": "final_answer", "content": "Created the inventory and generated the asset tag."},
    ])

    context = Context(SYSTEM_PROMPT)
    context.add_user_message(
        "Set up the inventory, remember my unit preference, and make an asset tag QR code."
    )
    answer = react_step(
        llm, context, tools, 10,
        policy=shipped_policy, confirm=auto_approve,
    )

    # Discovery: the MCP tools were discovered at runtime and offered to the model.
    offered = {schema["function"]["name"] for schema in llm.calls[0]["tools"]}
    assert "generate_qr_code" in offered
    assert "remember_fact" in offered

    # Built-in tool: the file was created in the workspace.
    assert (workspace / "inventory.md").is_file()

    # MCP: the QR subprocess wrote a real, decodable PNG in the fixed directory.
    png = workspace / "qr_codes" / "asset-tag.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = zxingcpp.read_barcode(Image.open(png))
    assert decoded is not None and decoded.text == ASSET_TAG

    # Memory: persisted across sessions -- a fresh store on the same path recalls it.
    fresh = Memory(db_path=db_path)
    assert "metric units" in fresh.recall("units")

    # Loop/context integrity: every tool call was answered, in order, and the
    # turn ended on the assistant's final answer.
    assert answer == "Created the inventory and generated the asset tag."
    messages = context.as_list()
    tool_results = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_results] == ["call_1", "call_2", "call_3", "call_4"]
    assert "metric units" in tool_results[3]["content"]
    assert messages[-1]["role"] == "assistant"
