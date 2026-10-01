"""
Unit tests for permission enforcement inside react_step.

Kept separate from test_loop.py, which covers the loop's own mechanics,
because this file is testing one specific property of it: that a refused
tool call does not run, does not look like a tool call to the
instrumentation, and still leaves the conversation in a state the next
LLM request can be built from.

Every test drives a fake LLM and a MagicMock registry, so nothing here
touches a real tool, the network or stdin. The hook assertions use a
fresh HookManager patched over agent.loop's module-level singleton --
the same technique test_metrics.py uses -- so they cannot be perturbed
by whatever else in the suite has emitted events.
"""

from unittest.mock import MagicMock

import pytest

from agent.context import Context
from agent.hooks import HookManager
from agent.loop import react_step
from agent.permissions import Decision, PermissionPolicy, auto_approve, auto_deny


def _llm(*decisions):
    """A stand-in for LLMClient that returns the given decisions in
    order. Anything after the last one repeats the final answer, so a
    test cannot hang if the loop iterates more than expected."""
    llm = MagicMock()
    llm.get_next_step.side_effect = list(decisions)
    return llm


def _tool_call(tool_name, call_id="call_1", arguments=None):
    return {
        "type": "tool_call",
        "id": call_id,
        "tool_name": tool_name,
        "arguments": {} if arguments is None else arguments,
    }


def _final(content="Done"):
    return {"type": "final_answer", "content": content}


def _registry():
    registry = MagicMock()
    registry.schemas.return_value = []
    registry.call.return_value = "tool ran"
    registry.is_builtin.return_value = True
    return registry


@pytest.fixture
def captured_events(monkeypatch):
    """Records every event react_step emits, via a fresh HookManager."""
    manager = HookManager()
    events = []
    for name in (
        "run_start", "run_end",
        "llm_call_start", "llm_call_end",
        "tool_call_start", "tool_call_end",
        "permission_decision",
    ):
        manager.on(name, lambda event_name, **payload: events.append((event_name, payload)))
    monkeypatch.setattr("agent.loop.hooks", manager)
    return events


def _names(events):
    return [name for name, _ in events]


# ── The allow path is unchanged ──────────────────────────────────────

def test_an_allowed_tool_still_runs_normally():
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    result = react_step(
        _llm(_tool_call("read"), _final()),
        context, registry, 5,
        policy=PermissionPolicy(rules={"read": "allow"}),
        confirm=auto_deny,
    )

    assert result == "Done"
    registry.call.assert_called_once_with("read", {})


def test_no_policy_means_every_tool_is_allowed():
    """The backwards-compatible default. Existing callers that pass four
    positional arguments must behave exactly as they did before
    permissions existed, or every pre-existing test would be enforcing a
    policy nobody wrote."""
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    react_step(_llm(_tool_call("anything_at_all"), _final()), context, registry, 5)

    registry.call.assert_called_once()


# ── Deny ─────────────────────────────────────────────────────────────

def test_a_denied_tool_is_never_executed():
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("forget_fact"), _final("I cannot do that.")),
        context, registry, 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    registry.call.assert_not_called()


def test_a_denied_tool_still_gets_a_tool_result_in_the_context():
    """Without this the next request is malformed: a tool_calls message
    with no matching tool result."""
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("forget_fact", call_id="call_7"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    messages = context.as_list()
    tool_results = [m for m in messages if m["role"] == "tool"]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_call_id"] == "call_7"

    # The tool result must directly answer the assistant message that
    # requested it, in that order.
    assistant_with_calls = [
        i for i, m in enumerate(messages)
        if m["role"] == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_with_calls) == 1
    assert messages[assistant_with_calls[0] + 1]["role"] == "tool"


def test_the_refusal_text_tells_the_model_not_to_retry_or_route_around_it():
    """A refusal the model reads as a transient failure gets retried
    until the iteration cap, which wastes a whole turn."""
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("forget_fact"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    refusal = [m for m in context.as_list() if m["role"] == "tool"][0]["content"]
    assert refusal.startswith("Permission denied:")
    assert "forget_fact" in refusal
    assert "not permitted" in refusal
    assert "Do not retry" in refusal
    # Must not look like the loop's own tool-exception string, or the
    # model cannot tell a policy block from a broken tool.
    assert "raised" not in refusal


def test_the_loop_recovers_and_returns_a_real_answer_after_a_refusal():
    context = Context("system")
    context.add_user_message("go")

    result = react_step(
        _llm(_tool_call("forget_fact"), _final("That is not permitted, sorry.")),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    assert result == "That is not permitted, sorry."
    assert context.as_list()[-1]["role"] == "assistant"


def test_a_refused_call_still_consumes_an_iteration():
    """It cost a real LLM round-trip, so it has to count -- otherwise a
    model that keeps asking for a denied tool loops forever."""
    llm = MagicMock()
    llm.get_next_step.return_value = _tool_call("forget_fact")
    context = Context("system")
    context.add_user_message("go")

    react_step(
        llm, context, _registry(), 3,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    assert llm.get_next_step.call_count == 3


# ── Require-user-confirmation ────────────────────────────────────────

def test_an_approved_confirmation_runs_the_tool():
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("ocr_extract_text"), _final()),
        context, registry, 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=auto_approve,
    )

    registry.call.assert_called_once()


def test_a_declined_confirmation_does_not_run_the_tool():
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("ocr_extract_text"), _final()),
        context, registry, 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=auto_deny,
    )

    registry.call.assert_not_called()
    refusal = [m for m in context.as_list() if m["role"] == "tool"][0]["content"]
    assert "declined" in refusal


def test_the_confirmation_callback_receives_the_tool_name_and_arguments():
    """The callback has to be able to show the user what it is approving."""
    seen = []
    context = Context("system")
    context.add_user_message("go")

    def spy(tool_name, arguments):
        seen.append((tool_name, arguments))
        return True

    react_step(
        _llm(_tool_call("ocr_extract_text", arguments='{"image_path": "r.jpg"}'), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=spy,
    )

    assert seen == [("ocr_extract_text", '{"image_path": "r.jpg"}')]


def test_the_default_confirmation_behaviour_is_to_deny():
    """No injected callback means nothing can ask a human, so a gated
    tool must not run. This is the case that would otherwise be a silent
    hole: a config that gates a tool, wired by a caller that forgot to
    supply a prompt."""
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("ocr_extract_text"), _final()),
        context, registry, 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        # confirm deliberately not passed
    )

    registry.call.assert_not_called()


def test_the_callback_is_not_consulted_for_an_allowed_tool():
    called = []
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("read"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"read": "allow"}),
        confirm=lambda name, args: called.append(name) or True,
    )

    assert called == []


def test_auto_approve_cannot_loosen_a_deny_rule():
    """Deny is absolute. This matters because the end-to-end tests all
    run with auto_approve, and a deny rule must still hold there."""
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("forget_fact"), _final()),
        context, registry, 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
        confirm=auto_approve,
    )

    registry.call.assert_not_called()


def test_a_raising_confirmation_callback_fails_closed():
    """A broken prompt channel must not crash the run, and must not be
    read as approval -- the user was never actually asked."""
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    def broken(tool_name, arguments):
        raise RuntimeError("no tty")

    result = react_step(
        _llm(_tool_call("ocr_extract_text"), _final("Could not ask you.")),
        context, registry, 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=broken,
    )

    registry.call.assert_not_called()
    assert result == "Could not ask you."
    refusal = [m for m in context.as_list() if m["role"] == "tool"][0]["content"]
    assert "confirmation prompt itself failed" in refusal
    assert "RuntimeError: no tty" in refusal


def test_a_truthy_non_boolean_callback_result_is_treated_as_approval():
    """bool() is applied, so a callback returning e.g. "yes" still means
    yes rather than silently failing closed."""
    registry = _registry()
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("ocr_extract_text"), _final()),
        context, registry, 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=lambda name, args: "yes",
    )

    registry.call.assert_called_once()


# ── Instrumentation: a refused call is not a tool call ───────────────

def test_a_refused_call_emits_no_tool_call_span(captured_events):
    """The requirement that drove where the check sits. Counting a
    blocked call as a failed tool call would conflate "the tool broke"
    with "the tool was not allowed to run" in the tool metrics."""
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("forget_fact"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    assert "tool_call_start" not in _names(captured_events)
    assert "tool_call_end" not in _names(captured_events)


def test_a_declined_confirmation_emits_no_tool_call_span(captured_events):
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("ocr_extract_text"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=auto_deny,
    )

    assert "tool_call_start" not in _names(captured_events)
    assert "tool_call_end" not in _names(captured_events)


def test_an_allowed_call_still_emits_its_tool_call_span(captured_events):
    """The other half: enforcement must not have silenced the normal
    path's instrumentation."""
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("read"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"read": "allow"}),
    )

    names = _names(captured_events)
    assert "tool_call_start" in names
    assert "tool_call_end" in names


def test_the_run_span_still_closes_when_a_call_is_refused(captured_events):
    """The `continue` must not skip past the end of the run span."""
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("forget_fact"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    names = _names(captured_events)
    assert names.count("run_start") == 1
    assert names.count("run_end") == 1


# ── The permission_decision event ────────────────────────────────────

def _permission_payloads(events):
    return [p for name, p in events if name == "permission_decision"]


def test_a_denied_call_reports_the_decision(captured_events):
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("forget_fact"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"forget_fact": "deny"}),
    )

    payloads = _permission_payloads(captured_events)
    assert len(payloads) == 1
    assert payloads[0]["tool_name"] == "forget_fact"
    assert payloads[0]["decision"] == "deny"
    assert payloads[0]["source"] == "rule"
    assert payloads[0]["allowed"] is False


def test_an_allowed_call_is_reported_too(captured_events):
    """So the dashboard can show the allow/deny balance rather than only
    refusals."""
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("read"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(default="allow"),
    )

    payloads = _permission_payloads(captured_events)
    assert len(payloads) == 1
    assert payloads[0]["decision"] == "allow"
    assert payloads[0]["source"] == "default"
    assert payloads[0]["allowed"] is True


def test_a_confirmed_call_reports_the_policy_outcome_and_the_effect(captured_events):
    """`decision` is what the policy said, `allowed` is what happened
    after the user was asked. Both are needed: the first is the label
    metrics.py documents, the second is the outcome."""
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("ocr_extract_text"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=auto_approve,
    )

    payload = _permission_payloads(captured_events)[0]
    assert payload["decision"] == Decision.REQUIRE_USER_CONFIRMATION.value
    assert payload["allowed"] is True


def test_a_declined_confirmation_reports_the_same_decision_but_not_allowed(captured_events):
    context = Context("system")
    context.add_user_message("go")

    react_step(
        _llm(_tool_call("ocr_extract_text"), _final()),
        context, _registry(), 5,
        policy=PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        confirm=auto_deny,
    )

    payload = _permission_payloads(captured_events)[0]
    assert payload["decision"] == Decision.REQUIRE_USER_CONFIRMATION.value
    assert payload["allowed"] is False


def test_no_permission_event_is_emitted_when_the_model_gives_a_final_answer(captured_events):
    context = Context("system")
    context.add_user_message("go")

    react_step(_llm(_final()), context, _registry(), 5, policy=PermissionPolicy())

    assert _permission_payloads(captured_events) == []
