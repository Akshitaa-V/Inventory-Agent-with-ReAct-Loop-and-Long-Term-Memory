"""
Unit tests for run identity on the observability seam.

Two separate concerns are covered here.

The first is that a run_id reaches every event belonging to a run -- the
run itself, each LLM call, each tool call, each permission decision --
so that one run's execution can be picked out of a shared event stream.
That is what 8.4's "structured logs or traces that allow the execution
of an individual agent run to be inspected" needs, and nothing else in
the system supplies it.

The second is nesting. A sub-agent runs inside its parent, so its
duration is already inside the parent's. If both land in the same metric
series, summing gives a wall-clock total larger than real elapsed time
and counts one delegated task twice. The "kind" label separates them,
and it is derived from parent_run_id rather than passed, so the two can
never contradict each other.
"""

from unittest.mock import MagicMock

import pytest

from agent.context import Context
from agent.hooks import HookManager
from agent.loop import react_step
from agent.permissions import PermissionPolicy


def _llm(*decisions):
    llm = MagicMock()
    llm.get_next_step.side_effect = list(decisions)
    return llm


def _registry():
    registry = MagicMock()
    registry.schemas.return_value = []
    registry.call.return_value = "ok"
    registry.is_builtin.return_value = True
    return registry


def _tool_call(name="read", call_id="c1"):
    return {"type": "tool_call", "id": call_id, "tool_name": name, "arguments": {}}


def _final(content="done"):
    return {"type": "final_answer", "content": content}


@pytest.fixture
def events(monkeypatch):
    manager = HookManager()
    captured = []
    for name in (
        "run_start", "run_end",
        "llm_call_start", "llm_call_end",
        "tool_call_start", "tool_call_end",
        "permission_decision",
    ):
        manager.on(name, lambda event_name, **p: captured.append((event_name, p)))
    monkeypatch.setattr("agent.loop.hooks", manager)
    return captured


def _payloads(events, name):
    return [p for event_name, p in events if event_name == name]


def _run(events_unused=None, **kwargs):
    context = Context("system")
    context.add_user_message("go")
    return react_step(
        _llm(_tool_call(), _final()), context, _registry(), 5,
        policy=PermissionPolicy(default="allow"), **kwargs,
    )


# ── A run_id exists and reaches every event ──────────────────────────

def test_a_run_id_is_generated_when_none_is_supplied(events):
    _run()

    run_id = _payloads(events, "run_start")[0]["run_id"]
    assert run_id
    assert isinstance(run_id, str)


def test_a_supplied_run_id_is_used_as_given(events):
    """A parent delegating to a sub-agent needs to know the child's id up
    front, so it passes one in rather than discovering it afterwards."""
    _run(run_id="fixed-id-123")

    assert _payloads(events, "run_start")[0]["run_id"] == "fixed-id-123"


def test_every_event_in_a_run_carries_the_same_run_id(events):
    """The whole point: one run's events must be attributable to it out
    of a stream shared with every other run in the process."""
    _run(run_id="r1")

    for name in ("run_start", "run_end", "llm_call_start", "llm_call_end",
                 "tool_call_start", "tool_call_end", "permission_decision"):
        payloads = _payloads(events, name)
        assert payloads, f"no {name} event was emitted"
        for payload in payloads:
            assert payload.get("run_id") == "r1", f"{name} lost the run_id"


def test_two_runs_get_different_ids(events):
    _run()
    first = _payloads(events, "run_start")[0]["run_id"]
    events.clear()
    _run()
    second = _payloads(events, "run_start")[0]["run_id"]

    assert first != second


def test_a_permission_refusal_still_carries_the_run_id(events):
    """The refusal path returns early, so it is the one most likely to
    drop the identity."""
    context = Context("system")
    context.add_user_message("go")
    react_step(
        _llm(_tool_call("forget_fact"), _final()), context, _registry(), 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}), run_id="r2",
    )

    payload = _payloads(events, "permission_decision")[0]
    assert payload["run_id"] == "r2"
    assert payload["allowed"] is False


# ── kind: root vs sub ────────────────────────────────────────────────

def test_a_run_with_no_parent_is_labelled_root(events):
    _run()

    assert _payloads(events, "run_start")[0]["kind"] == "root"


def test_a_run_with_a_parent_is_labelled_sub(events):
    _run(parent_run_id="parent-1")

    assert _payloads(events, "run_start")[0]["kind"] == "sub"


def test_kind_is_derived_from_the_parent_rather_than_passed(events):
    """Derived, so "sub with no parent" and "root with a parent" are
    both unrepresentable -- the label and the linkage cannot disagree."""
    with pytest.raises(TypeError):
        _run(kind="sub")


def test_a_sub_run_records_which_run_it_was_delegated_from(events):
    """8.4 requires sub-agent executions to be recorded as part of the
    corresponding parent run, which needs the link, not just the split."""
    _run(run_id="child-1", parent_run_id="parent-1")

    payload = _payloads(events, "run_start")[0]
    assert payload["run_id"] == "child-1"
    assert payload["parent_run_id"] == "parent-1"


def test_a_root_run_carries_no_parent_key_at_all(events):
    """Absent rather than None, so a log line for a top-level run does
    not carry an empty field implying a parent that never existed."""
    _run()

    assert "parent_run_id" not in _payloads(events, "run_start")[0]


def test_the_kind_label_is_on_the_end_event_too(events):
    """The end event is what metrics.py reads to pick a series, so the
    label has to survive to that point, not just appear at the start."""
    _run(parent_run_id="p")

    assert _payloads(events, "run_end")[0]["kind"] == "sub"


def test_a_nested_run_emits_its_own_complete_span(events):
    """Simulates delegation: an inner run started inside an outer one.
    Both spans must open and close, and land in different kinds."""
    outer = Context("system")
    outer.add_user_message("go")

    inner_registry = MagicMock()
    inner_registry.schemas.return_value = []
    inner_registry.is_builtin.return_value = True

    def delegate(name, arguments):
        inner = Context("sub system")
        inner.add_user_message("sub task")
        react_step(
            _llm(_final("sub done")), inner, _registry(), 5,
            policy=PermissionPolicy(default="allow"),
            run_id="child", parent_run_id="parent",
        )
        return "delegated"

    inner_registry.call.side_effect = delegate

    react_step(
        _llm(_tool_call("delegate"), _final("all done")), outer, inner_registry, 5,
        policy=PermissionPolicy(default="allow"), run_id="parent",
    )

    kinds = [p["kind"] for p in _payloads(events, "run_end")]
    assert sorted(kinds) == ["root", "sub"]

    starts = _payloads(events, "run_start")
    assert len(starts) == 2
    assert {p["run_id"] for p in starts} == {"parent", "child"}


# ── Backwards compatibility ──────────────────────────────────────────

def test_callers_passing_four_positional_arguments_still_work(events):
    """Existing call sites -- main.py, dashboard.py, the e2e tests --
    pass no run identity and must keep working."""
    context = Context("system")
    context.add_user_message("go")

    result = react_step(_llm(_final("fine")), context, _registry(), 5)

    assert result == "fine"
    assert _payloads(events, "run_start")[0]["kind"] == "root"
