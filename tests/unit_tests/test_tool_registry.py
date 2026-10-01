"""
Unit tests for ToolRegistry: verifies the create-overwrite prevention
and workspace path-safety fixes.

Run from the repo root:
    pytest tests/test_tool_registry.py
"""

import shutil

import pytest

from agent.tool_registry import ToolRegistry


@pytest.fixture
def registry(tmp_path):
    reg = ToolRegistry(workspace_root=str(tmp_path))
    yield reg
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_create_file_succeeds(registry):
    result = registry.call("create", {"path": "a.txt", "type": "file"})
    assert "Created file" in result


def test_create_refuses_existing_file(registry):
    registry.call("create", {"path": "a.txt", "type": "file"})
    result = registry.call("create", {"path": "a.txt", "type": "file"})
    assert "already exists" in result


def test_read_blocks_path_outside_workspace(registry):
    result = registry.call("read", {"path": "../../../etc/passwd"})
    assert "escapes the workspace" in result


def test_modify_appends_without_erasing(registry):
    registry.call("create", {"path": "a.txt", "type": "file"})
    registry.call("modify", {"path": "a.txt", "content": "first line"})
    registry.call("modify", {"path": "a.txt", "content": "second line"})
    content = registry.call("read", {"path": "a.txt"})
    assert "first line" in content
    assert "second line" in content


# ── Long-term memory schemas ────────────────────────────────────────
#
# schemas() is what loop.py sends to the LLM: a tool that is dispatched
# but has no schema is unreachable by the model, which is exactly the
# bug recall_summaries had. These assert every memory tool the system
# prompt advertises is actually exposed.


def _schema_names(registry):
    return [s["function"]["name"] for s in registry.schemas()]


def _schema(registry, name):
    return next(s for s in registry.schemas() if s["function"]["name"] == name)


def test_recall_summaries_schema_is_exposed(registry):
    """recall_summaries is dispatched and advertised in the prompt, so it
    must carry a schema or the model can never call it."""
    assert "recall_summaries" in _schema_names(registry)


def test_recall_summaries_takes_a_query(registry):
    params = _schema(registry, "recall_summaries")["function"]["parameters"]
    assert params["required"] == ["query"]
    assert params["properties"]["query"]["type"] == "string"


def test_all_memory_tools_are_exposed(registry):
    """Every memory tool in the dispatch table needs a schema."""
    names = _schema_names(registry)
    for tool in (
        "remember_fact",
        "recall_fact",
        "recall_summaries",
        "recall_all_summaries",
        "forget_fact",
    ):
        assert tool in names, f"{tool} is dispatched but has no schema"
