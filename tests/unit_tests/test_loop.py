from unittest.mock import MagicMock

from context import Context
from loop import ITERATION_LIMIT_MESSAGE, react_step


def test_direct_answer():
    mock_llm = MagicMock()
    mock_registry = MagicMock()
    context = Context("You are a test assistant.")

    mock_llm.get_next_step.return_value = {
        "type": "final_answer",
        "content": "Hello!",
    }
    mock_registry.schemas.return_value = []

    context.add_user_message("Hi")
    result = react_step(mock_llm, context, mock_registry, 5)

    assert result == "Hello!"
    messages = context.as_list()
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_tool_then_answer():
    mock_llm = MagicMock()
    mock_registry = MagicMock()
    context = Context("You are a test assistant.")

    mock_llm.get_next_step.side_effect = [
        {"type": "tool_call", "id": "call_1", "tool_name": "create", "arguments": {"path": "a.txt", "type": "file"}},
        {"type": "final_answer", "content": "Done"},
    ]
    mock_registry.schemas.return_value = [{"type": "function"}]
    mock_registry.call.return_value = "Created file: a.txt"

    context.add_user_message("Create a file")
    result = react_step(mock_llm, context, mock_registry, 5)

    assert result == "Done"
    mock_registry.call.assert_called_once_with("create", {"path": "a.txt", "type": "file"})
    messages = context.as_list()
    assert len(messages) == 5
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "tool"
    assert messages[4]["role"] == "assistant"


def test_iteration_limit_records_assistant_message():
    """The cap must close the turn: the guard message is returned AND appended,
    so the history never ends on a dangling 'tool' role."""
    mock_llm = MagicMock()
    mock_registry = MagicMock()
    context = Context("You are a test assistant.")

    # The model never returns a final answer -- it only ever asks for a tool.
    mock_llm.get_next_step.return_value = {
        "type": "tool_call",
        "id": "call_1",
        "tool_name": "navigate",
        "arguments": {},
    }
    mock_registry.schemas.return_value = []
    mock_registry.call.return_value = "ok"

    context.add_user_message("Go")
    result = react_step(mock_llm, context, mock_registry, 3)

    assert result == ITERATION_LIMIT_MESSAGE
    assert mock_llm.get_next_step.call_count == 3

    messages = context.as_list()
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == ITERATION_LIMIT_MESSAGE


def test_tool_exception_becomes_an_observation():
    """A raising tool must not kill the loop. The error is fed back as the tool
    result so the model can recover, and the tool_calls message still gets an
    answer -- otherwise the next request would be malformed."""
    mock_llm = MagicMock()
    mock_registry = MagicMock()
    context = Context("You are a test assistant.")

    mock_llm.get_next_step.side_effect = [
        {"type": "tool_call", "id": "call_1", "tool_name": "create",
         "arguments": {"path": "README.md", "type": "folder"}},
        {"type": "final_answer", "content": "That name is taken, so I stopped."},
    ]
    mock_registry.schemas.return_value = []
    mock_registry.call.side_effect = FileExistsError(17, "File exists")

    context.add_user_message("Create a folder called README.md")
    result = react_step(mock_llm, context, mock_registry, 5)

    # The loop survived and reached a real final answer.
    assert result == "That name is taken, so I stopped."

    messages = context.as_list()

    # The failed tool call was still answered, and the model was told why.
    tool_results = [m for m in messages if m["role"] == "tool"]
    assert len(tool_results) == 1
    assert "FileExistsError" in tool_results[0]["content"]
    assert "create" in tool_results[0]["content"]

    assert messages[-1]["role"] == "assistant"
