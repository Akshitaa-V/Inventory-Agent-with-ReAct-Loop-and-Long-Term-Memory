"""
Unit tests for react_step's result contract.

Before this contract existed, reaching the iteration cap returned an
ordinary string, indistinguishable from a real answer. Nothing could tell
the two apart -- so `agent_runs_total` counted a run that never answered
under "success", and a parent delegating to a sub-agent would have no way
to know its delegate gave up.

Two properties are pinned here, and they pull in opposite directions:

  - `.success` reports whether the run actually produced an answer, and
    the run_end event carries the same fact so the metrics agree with it.
  - The result is still a plain string in every other respect, because
    around thirty call sites print it, compare it, search it and format
    it, and none of them should have to change.
"""

from unittest.mock import MagicMock

import pytest

from agent.cli_format import pretty_print_tables
from agent.context import Context
from agent.hooks import HookManager
from agent.loop import ITERATION_LIMIT_MESSAGE, RunResult, react_step
from agent.permissions import PermissionPolicy


def _registry():
    registry = MagicMock()
    registry.schemas.return_value = []
    registry.call.return_value = "ok"
    registry.is_builtin.return_value = True
    return registry


def _llm(*decisions, repeat=None):
    llm = MagicMock()
    if repeat is not None:
        llm.get_next_step.return_value = repeat
    else:
        llm.get_next_step.side_effect = list(decisions)
    return llm


def _run(llm, iterations=5):
    context = Context("system")
    context.add_user_message("go")
    return react_step(
        llm, context, _registry(), iterations,
        policy=PermissionPolicy(default="allow"),
    )


@pytest.fixture
def run_events(monkeypatch):
    manager = HookManager()
    captured = []
    manager.on("run_end", lambda name, **p: captured.append(p))
    monkeypatch.setattr("agent.loop.hooks", manager)
    return captured


# ── The success signal ───────────────────────────────────────────────

def test_a_real_answer_reports_success():
    result = _run(_llm({"type": "final_answer", "content": "All done"}))

    assert result.success is True
    assert result.reason == "final_answer"
    assert result == "All done"


def test_reaching_the_iteration_cap_reports_failure():
    """The bug this contract exists to fix."""
    result = _run(
        _llm(repeat={"type": "tool_call", "id": "c1", "tool_name": "read", "arguments": {}}),
        iterations=3,
    )

    assert result.success is False
    assert result.reason == "iteration_limit"
    assert result == ITERATION_LIMIT_MESSAGE


def test_the_result_carries_the_run_id():
    context = Context("system")
    context.add_user_message("go")

    result = react_step(
        _llm({"type": "final_answer", "content": "hi"}), context, _registry(), 5,
        policy=PermissionPolicy(default="allow"), run_id="r-42",
    )

    assert result.run_id == "r-42"


# ── The metrics agree with it ────────────────────────────────────────

def test_a_successful_run_is_reported_as_successful(run_events):
    _run(_llm({"type": "final_answer", "content": "done"}))

    assert run_events[0]["success"] is True
    assert run_events[0]["reason"] == "final_answer"


def test_a_capped_run_is_reported_as_unsuccessful(run_events):
    """timed() only infers failure from an exception, and nothing is
    raised here -- so without an explicit override the run_end event said
    success=True and agent_runs_total counted a non-answer as a success."""
    _run(
        _llm(repeat={"type": "tool_call", "id": "c1", "tool_name": "read", "arguments": {}}),
        iterations=2,
    )

    assert run_events[0]["success"] is False
    assert run_events[0]["reason"] == "iteration_limit"


def test_a_capped_run_still_closes_its_span_exactly_once(run_events):
    _run(
        _llm(repeat={"type": "tool_call", "id": "c1", "tool_name": "read", "arguments": {}}),
        iterations=2,
    )

    assert len(run_events) == 1
    assert run_events[0]["duration_seconds"] >= 0


def test_a_raising_run_is_reported_as_unsuccessful_with_an_error(run_events):
    """The third outcome. An exception carries `error` and no `reason`,
    since it never reached either normal ending."""
    llm = MagicMock()
    llm.get_next_step.side_effect = RuntimeError("network down")

    with pytest.raises(RuntimeError):
        _run(llm)

    assert run_events[0]["success"] is False
    assert "network down" in run_events[0]["error"]
    assert "reason" not in run_events[0]


# ── Still a string, in every way the callers depend on ───────────────

def test_the_result_is_a_string():
    result = _run(_llm({"type": "final_answer", "content": "text"}))

    assert isinstance(result, str)


def test_it_compares_equal_to_the_plain_answer():
    """test_loop.py asserts `result == "Hello!"` and
    `result == ITERATION_LIMIT_MESSAGE`; both must keep working."""
    assert _run(_llm({"type": "final_answer", "content": "Hello!"})) == "Hello!"


def test_string_operations_work_unchanged():
    """The e2e tests use `in`, `.lower()` and truthiness on this value."""
    result = _run(_llm({"type": "final_answer", "content": "Created ASSET-tag.png"}))

    assert "asset-tag" in result.lower()
    assert result.startswith("Created")
    assert len(result) == len("Created ASSET-tag.png")


def test_the_table_formatter_accepts_it():
    """main.py pipes the result straight into pretty_print_tables, which
    does regex work on it."""
    result = _run(_llm({"type": "final_answer", "content": "| a | b |\n|---|---|\n| 1 | 2 |"}))

    assert pretty_print_tables(result)


def test_truthiness_is_string_truthiness_not_success():
    """Deliberate. test_qr_round_trip's `assert answer` means "the agent
    said something", and making truthiness mean success would silently
    change what that verifies. Success is checked explicitly."""
    failed = RunResult(ITERATION_LIMIT_MESSAGE, success=False, reason="iteration_limit")
    empty_but_successful = RunResult("", success=True, reason="final_answer")

    assert bool(failed) is True          # non-empty text, despite failing
    assert bool(empty_but_successful) is False  # empty text, despite success
    assert failed.success is False
    assert empty_but_successful.success is True


def test_repr_shows_the_outcome_for_debugging():
    result = RunResult("hi", success=False, reason="iteration_limit", run_id="r1")

    assert "success=False" in repr(result)
    assert "iteration_limit" in repr(result)
