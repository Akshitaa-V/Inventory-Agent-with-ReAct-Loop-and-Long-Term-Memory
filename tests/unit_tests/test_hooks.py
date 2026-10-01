"""
Unit tests for agent/hooks.py -- the lifecycle hook mechanism that
metrics.py (and eventually permission management / sub-agents) is
built on. These formalize the manual verification done during
development into committed, repeatable coverage.

Each test uses its own fresh HookManager instance, never the shared
`agent.hooks.hooks` singleton, so tests can't interfere with each
other or with anything else that's registered listeners globally.
"""

import pytest

from agent.hooks import HookManager, timed


def test_emit_calls_registered_listener():
    hooks = HookManager()
    received = []
    hooks.on("thing_happened", lambda name, **p: received.append((name, p)))

    hooks.emit("thing_happened", foo="bar")

    assert received == [("thing_happened", {"foo": "bar"})]


def test_emit_with_no_listeners_does_nothing():
    hooks = HookManager()
    hooks.emit("nobody_is_listening", x=1)  # must not raise


def test_emit_calls_every_listener_for_the_same_event():
    hooks = HookManager()
    calls = []
    hooks.on("x", lambda name, **p: calls.append("first"))
    hooks.on("x", lambda name, **p: calls.append("second"))

    hooks.emit("x")

    assert calls == ["first", "second"]


def test_a_broken_listener_does_not_stop_other_listeners_or_crash():
    hooks = HookManager()
    calls = []

    def broken(name, **p):
        raise RuntimeError("listener bug")

    hooks.on("x", broken)
    hooks.on("x", lambda name, **p: calls.append("still ran"))

    hooks.emit("x")  # must not raise

    assert calls == ["still ran"]


def test_timed_success_emits_start_and_end_with_success_true():
    hooks = HookManager()
    events = []
    hooks.on("op_start", lambda name, **p: events.append(("start", p)))
    hooks.on("op_end", lambda name, **p: events.append(("end", p)))

    with timed(hooks, "op", tool_name="navigate"):
        pass

    assert events[0] == ("start", {"tool_name": "navigate"})
    end_name, end_payload = events[1]
    assert end_name == "end"
    assert end_payload["success"] is True
    assert end_payload["error"] is None
    assert end_payload["tool_name"] == "navigate"
    assert end_payload["duration_seconds"] >= 0


def test_timed_failure_propagates_exception_and_records_it():
    hooks = HookManager()
    events = []
    hooks.on("op_end", lambda name, **p: events.append(p))

    with pytest.raises(ValueError, match="boom"):
        with timed(hooks, "op"):
            raise ValueError("boom")

    assert events[0]["success"] is False
    assert events[0]["error"] == "boom"


def test_timed_context_dict_adds_fields_to_the_end_event():
    """This is the mechanism that lets token counts (known only after
    the LLM responds) get attached to a single end event."""
    hooks = HookManager()
    events = []
    hooks.on("llm_call_end", lambda name, **p: events.append(p))

    with timed(hooks, "llm_call") as ctx:
        ctx["input_tokens"] = 150
        ctx["output_tokens"] = 40

    assert events[0]["input_tokens"] == 150
    assert events[0]["output_tokens"] == 40
    assert events[0]["success"] is True  # unaffected by unrelated context fields


def test_timed_context_can_override_automatic_success_and_error():
    """This is the pattern loop.py actually uses: it catches its own
    exception (to turn it into an observation string) rather than
    letting it propagate, so timed() never sees it -- the context dict
    is how failure still gets reported correctly in that case."""
    hooks = HookManager()
    events = []
    hooks.on("tool_call_end", lambda name, **p: events.append(p))

    with timed(hooks, "tool_call", tool_name="broken_tool") as ctx:
        try:
            raise RuntimeError("internal failure")
        except RuntimeError as exc:
            ctx["success"] = False
            ctx["error"] = str(exc)
        # no re-raise -- the caller handled it, same as loop.py does

    assert events[0]["success"] is False
    assert events[0]["error"] == "internal failure"
    assert events[0]["tool_name"] == "broken_tool"  # label still present


def test_timed_does_not_emit_start_and_end_out_of_order():
    hooks = HookManager()
    order = []
    hooks.on("op_start", lambda name, **p: order.append("start"))
    hooks.on("op_end", lambda name, **p: order.append("end"))

    with timed(hooks, "op"):
        order.append("inside")

    assert order == ["start", "inside", "end"]
