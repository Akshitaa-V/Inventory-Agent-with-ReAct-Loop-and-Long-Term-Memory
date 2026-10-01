"""
End-to-end test: a scripted LLM drives the real ReAct loop to create a
folder and write numbers in words. No token and no config.json needed, so
this test always runs.

The model is replaced by ScriptedLLM, which returns a fixed sequence of
tool calls and a final answer. Everything below the model is real: the
loop, the ToolRegistry, and the workspace filesystem.

Run from the repository root:
    pytest tests/e2e_tests/test_number_in_words.py -v
"""

from context import Context
from loop import react_step
from permissions import auto_approve
from main import SYSTEM_PROMPT
from tool_registry import ToolRegistry


class ScriptedLLM:
    """Minimal stand-in for LLMClient.

    Returns a preset list of decisions, one per get_next_step() call. The
    decision dicts have the same shape llm_client.py returns.
    """

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls = []

    def get_next_step(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._decisions:
            return {"type": "final_answer", "content": "(script exhausted)"}
        return self._decisions.pop(0)


NUMBERS_CONTENT = (
    "1 one\n"
    "2 two\n"
    "3 three\n"
    "4 four\n"
    "5 five\n"
    "6 six\n"
    "7 seven\n"
    "8 eight\n"
    "9 nine\n"
    "10 ten\n"
    "11 eleven\n"
    "12 twelve\n"
    "13 thirteen\n"
    "14 fourteen\n"
    "15 fifteen\n"
    "16 sixteen\n"
    "17 seventeen\n"
    "18 eighteen\n"
    "19 nineteen\n"
    "20 twenty\n"
)


def test_number_in_words(tmp_path, shipped_policy):
    """Agent creates output/numbers.md with numbers 1-20 written in words."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # All MCP servers disabled so the test stays offline (no Tesseract, no
    # CPSC endpoint, no QR subprocess).
    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers={
            "recall": {"enabled": False},
            "ocr": {"enabled": False},
            "qr": {"enabled": False},
        },
    )

    llm = ScriptedLLM([
        {"type": "tool_call", "id": "call_1", "tool_name": "create",
         "arguments": {"path": "output", "type": "folder"}},
        {"type": "tool_call", "id": "call_2", "tool_name": "create",
         "arguments": {"path": "output/numbers.md", "type": "file"}},
        {"type": "tool_call", "id": "call_3", "tool_name": "modify",
         "arguments": {"path": "output/numbers.md", "content": NUMBERS_CONTENT}},
        {"type": "final_answer", "content": "Created output/numbers.md with numbers 1-20 in words."},
    ])

    context = Context(SYSTEM_PROMPT)
    context.add_user_message(
        "Create a folder called output and inside it a file called numbers.md "
        "that contains a list with the numbers from 1 to 20 in words."
    )
    answer = react_step(
        llm, context, tools, 10,
        policy=shipped_policy, confirm=auto_approve,
    )

    # 1. output/ folder exists
    output_dir = workspace / "output"
    assert output_dir.is_dir(), f"output/ folder was not created. Agent said: {answer}"

    # 2. output/numbers.md exists
    numbers_path = output_dir / "numbers.md"
    assert numbers_path.exists(), f"output/numbers.md was not created. Agent said: {answer}"

    # 3. numbers.md contains 'one' and 'twenty'
    content = numbers_path.read_text(encoding="utf-8")
    assert "one" in content, f"numbers.md missing 'one'. Agent said: {answer}"
    assert "twenty" in content, f"numbers.md missing 'twenty'. Agent said: {answer}"

    # 4. every tool call was answered and the turn ended on the final answer
    assert answer == "Created output/numbers.md with numbers 1-20 in words."
    messages = context.as_list()
    tool_results = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_results] == ["call_1", "call_2", "call_3"]
    assert messages[-1]["role"] == "assistant"