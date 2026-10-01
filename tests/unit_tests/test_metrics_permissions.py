"""
Unit tests for permission metrics -- the listener in agent/metrics.py and
its wiring to the "permission_decision" event.

Kept out of test_metrics.py so that file stays as its author left it.
Where test_metrics.py calls the private listeners directly, these drive
events through a HookManager after init_metrics() has registered onto
it, because the thing most likely to break here is the wiring rather
than the arithmetic: a listener that is never registered fails silently,
since hooks.emit with no listener is a no-op by design.

Prometheus counters are process-global and cumulative, so every
assertion is on the DELTA caused by one emit, never an absolute value.
"""

import pytest

from agent import metrics
from agent.hooks import HookManager
from agent.loop import react_step
from agent.permissions import PermissionPolicy, auto_approve, auto_deny


def _count(**labels):
    return metrics.PERMISSION_DECISIONS_TOTAL.labels(**labels)._value.get()


@pytest.fixture
def emit(monkeypatch):
    """Returns a function that emits a permission_decision event into a
    throwaway HookManager with the real listeners registered on it."""
    manager = HookManager()
    monkeypatch.setattr(metrics, "hooks", manager)
    metrics.init_metrics()
    return lambda **payload: manager.emit("permission_decision", **payload)


# ── The wiring ───────────────────────────────────────────────────────

def test_init_metrics_registers_the_permission_listener(monkeypatch):
    manager = HookManager()
    monkeypatch.setattr(metrics, "hooks", manager)

    metrics.init_metrics()

    assert "permission_decision" in manager._listeners
    assert len(manager._listeners["permission_decision"]) == 1


def test_init_metrics_still_registers_every_pre_existing_listener(monkeypatch):
    """Adding the permission listener must not have disturbed the five
    that were already there."""
    manager = HookManager()
    monkeypatch.setattr(metrics, "hooks", manager)

    metrics.init_metrics()

    for event in ("run_start", "run_end", "llm_call_end", "tool_call_end", "memory_op_end"):
        assert len(manager._listeners[event]) == 1


# ── The four label values ────────────────────────────────────────────

def test_an_allow_is_counted_under_allow(emit):
    before = _count(decision="allow", tool_name="read")

    emit(tool_name="read", decision="allow", source="default", allowed=True)

    assert _count(decision="allow", tool_name="read") == before + 1


def test_a_deny_is_counted_under_deny(emit):
    before = _count(decision="deny", tool_name="forget_fact")

    emit(tool_name="forget_fact", decision="deny", source="rule", allowed=False)

    assert _count(decision="deny", tool_name="forget_fact") == before + 1


def test_an_approved_confirmation_is_counted_under_confirm_approved(emit):
    """The policy outcome was require-user-confirmation and the user
    agreed. Both facts have to be readable from the one label."""
    before = _count(decision="confirm_approved", tool_name="ocr_extract_text")

    emit(
        tool_name="ocr_extract_text",
        decision="require-user-confirmation",
        source="rule",
        allowed=True,
    )

    assert _count(decision="confirm_approved", tool_name="ocr_extract_text") == before + 1


def test_a_declined_confirmation_is_counted_under_confirm_denied(emit):
    before = _count(decision="confirm_denied", tool_name="ocr_extract_text")

    emit(
        tool_name="ocr_extract_text",
        decision="require-user-confirmation",
        source="rule",
        allowed=False,
    )

    assert _count(decision="confirm_denied", tool_name="ocr_extract_text") == before + 1


def test_a_bare_require_user_confirmation_label_is_never_recorded(emit):
    """The raw policy outcome must always be resolved into one of the two
    confirm_ values, or the dashboard shows a bucket that answers
    neither "was confirmation required" nor "what did the user say"."""
    before = _count(decision="require-user-confirmation", tool_name="ocr_extract_text")

    emit(tool_name="ocr_extract_text", decision="require-user-confirmation", allowed=True)
    emit(tool_name="ocr_extract_text", decision="require-user-confirmation", allowed=False)

    assert _count(decision="require-user-confirmation", tool_name="ocr_extract_text") == before


# ── Counting exactly once ────────────────────────────────────────────

def test_a_confirmed_call_increments_exactly_one_series(emit):
    """The decision that drove the design. Recording the policy outcome
    AND the resolution separately would make the sum over this counter
    exceed the number of checks actually made."""
    before = {
        label: _count(decision=label, tool_name="ocr_extract_text")
        for label in ("allow", "deny", "confirm_approved", "confirm_denied")
    }

    emit(
        tool_name="ocr_extract_text",
        decision="require-user-confirmation",
        source="rule",
        allowed=True,
    )

    after = {
        label: _count(decision=label, tool_name="ocr_extract_text")
        for label in before
    }
    changed = {label for label in before if after[label] != before[label]}
    assert changed == {"confirm_approved"}
    assert after["confirm_approved"] == before["confirm_approved"] + 1


def test_the_gate_count_is_recoverable_by_summing_the_two_confirm_labels(emit):
    """`confirm_approved + confirm_denied` has to equal the number of
    calls that hit a confirmation gate, since that is the query which
    replaces the bare require-user-confirmation bucket."""
    before = (
        _count(decision="confirm_approved", tool_name="check_product_recall")
        + _count(decision="confirm_denied", tool_name="check_product_recall")
    )

    for allowed in (True, False, True):
        emit(
            tool_name="check_product_recall",
            decision="require-user-confirmation",
            allowed=allowed,
        )

    after = (
        _count(decision="confirm_approved", tool_name="check_product_recall")
        + _count(decision="confirm_denied", tool_name="check_product_recall")
    )
    assert after == before + 3


# ── Tolerance of a malformed payload ─────────────────────────────────

def test_a_payload_with_no_decision_is_recorded_as_unknown(emit):
    """Instrumentation must not be the thing that breaks a run. hooks
    swallows listener exceptions anyway, but a silently-dropped metric is
    worse than one labelled unknown."""
    before = _count(decision="unknown", tool_name="mystery")

    emit(tool_name="mystery")

    assert _count(decision="unknown", tool_name="mystery") == before + 1


def test_a_payload_with_no_tool_name_is_recorded_as_unknown(emit):
    before = _count(decision="deny", tool_name="unknown")

    emit(decision="deny", allowed=False)

    assert _count(decision="deny", tool_name="unknown") == before + 1


# ── End to end through the real loop ─────────────────────────────────

def _llm(*decisions):
    from unittest.mock import MagicMock
    llm = MagicMock()
    llm.get_next_step.side_effect = list(decisions)
    return llm


def _registry():
    from unittest.mock import MagicMock
    registry = MagicMock()
    registry.schemas.return_value = []
    registry.call.return_value = "ok"
    registry.is_builtin.return_value = True
    return registry


def _run(policy, confirm, tool_name, monkeypatch):
    from agent.context import Context

    manager = HookManager()
    monkeypatch.setattr(metrics, "hooks", manager)
    monkeypatch.setattr("agent.loop.hooks", manager)
    metrics.init_metrics()

    context = Context("system")
    context.add_user_message("go")
    react_step(
        _llm(
            {"type": "tool_call", "id": "c1", "tool_name": tool_name, "arguments": {}},
            {"type": "final_answer", "content": "done"},
        ),
        context, _registry(), 5,
        policy=policy, confirm=confirm,
    )


def test_a_real_denied_run_lands_in_the_counter(monkeypatch):
    """The whole path: loop refuses, emits, listener records -- with no
    manual emit anywhere in the test."""
    before = _count(decision="deny", tool_name="forget_fact")

    _run(PermissionPolicy(rules={"forget_fact": "deny"}), auto_deny, "forget_fact", monkeypatch)

    assert _count(decision="deny", tool_name="forget_fact") == before + 1


def test_a_real_approved_confirmation_lands_in_the_counter(monkeypatch):
    before = _count(decision="confirm_approved", tool_name="ocr_extract_text")

    _run(
        PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        auto_approve, "ocr_extract_text", monkeypatch,
    )

    assert _count(decision="confirm_approved", tool_name="ocr_extract_text") == before + 1


def test_a_real_declined_confirmation_lands_in_the_counter(monkeypatch):
    before = _count(decision="confirm_denied", tool_name="ocr_extract_text")

    _run(
        PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"}),
        auto_deny, "ocr_extract_text", monkeypatch,
    )

    assert _count(decision="confirm_denied", tool_name="ocr_extract_text") == before + 1
