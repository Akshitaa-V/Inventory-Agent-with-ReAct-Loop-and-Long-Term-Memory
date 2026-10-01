"""
Unit tests for agent/metrics.py -- confirms each hook listener updates
the correct Prometheus metric with the correct value.

Prometheus counters/histograms are module-level singletons that persist
for the life of the process (they're cumulative by design), so tests
assert on the DELTA caused by one call, not an absolute value -- running
the suite twice, or alongside other tests that also touch these same
metrics, must not make these tests flaky.

These call the private listener functions directly rather than going
through agent.hooks.hooks (the shared global singleton) or init_metrics().
That keeps this file testing metrics.py's own logic in isolation --
"given this event payload, does the right metric update correctly" --
independent of the hook-dispatch mechanics themselves, which are already
covered in test_hooks.py.
"""

from agent import metrics


def _counter_value(counter, **labels):
    if labels:
        return counter.labels(**labels)._value.get()
    return counter._value.get()


def _histogram_sum_and_count(histogram, **labels):
    """Reads a histogram's current _sum/_count via the public collect()
    API, filtered to the given labels (or the unlabeled series if none
    are given). This is the supported way to introspect a metric's
    current value -- prometheus_client's internal attribute names
    aren't part of its public API and shouldn't be relied on directly."""
    total_sum = total_count = None
    for family in histogram.collect():
        for sample in family.samples:
            if sample.labels != labels:
                continue
            if sample.name.endswith("_sum"):
                total_sum = sample.value
            elif sample.name.endswith("_count"):
                total_count = sample.value
    return total_sum, total_count


def test_run_start_increments_active_runs():
    before = metrics.AGENT_ACTIVE_RUNS._value.get()
    metrics._on_run_start("run_start")
    after = metrics.AGENT_ACTIVE_RUNS._value.get()
    assert after == before + 1


def test_run_end_decrements_active_and_records_success():
    metrics._on_run_start("run_start")
    active_before = metrics.AGENT_ACTIVE_RUNS._value.get()
    total_before = _counter_value(metrics.AGENT_RUNS_TOTAL, status="success", kind="root")

    metrics._on_run_end("run_end", success=True, duration_seconds=1.5)

    assert metrics.AGENT_ACTIVE_RUNS._value.get() == active_before - 1
    assert _counter_value(
        metrics.AGENT_RUNS_TOTAL, status="success", kind="root"
    ) == total_before + 1


def test_run_end_records_failure_under_the_error_label():
    before = _counter_value(metrics.AGENT_RUNS_TOTAL, status="error", kind="root")
    metrics._on_run_end("run_end", success=False, duration_seconds=0.2)
    after = _counter_value(metrics.AGENT_RUNS_TOTAL, status="error", kind="root")
    assert after == before + 1


def test_a_run_with_no_kind_label_is_recorded_as_root():
    """Callers predating sub-agents emit no "kind". They are whole
    top-level runs, so that is where they must land."""
    before = _counter_value(metrics.AGENT_RUNS_TOTAL, status="success", kind="root")
    metrics._on_run_end("run_end", success=True, duration_seconds=0.1)
    after = _counter_value(metrics.AGENT_RUNS_TOTAL, status="success", kind="root")
    assert after == before + 1


def test_a_sub_run_is_recorded_separately_from_a_root_run():
    """The double-counting fix. A sub-agent's duration sits inside its
    parent's, so the two must not share a series -- otherwise summing the
    histogram reports more wall-clock time than actually elapsed."""
    root_before = _counter_value(metrics.AGENT_RUNS_TOTAL, status="success", kind="root")
    sub_before = _counter_value(metrics.AGENT_RUNS_TOTAL, status="success", kind="sub")

    metrics._on_run_end("run_end", success=True, duration_seconds=2.0, kind="sub")

    assert _counter_value(metrics.AGENT_RUNS_TOTAL, status="success", kind="sub") == sub_before + 1
    assert _counter_value(metrics.AGENT_RUNS_TOTAL, status="success", kind="root") == root_before


def test_run_id_is_never_used_as_a_prometheus_label():
    """run_id rides on the events for structured logs, but a label with a
    fresh value per run would create an unbounded number of time series.
    This pins that the payload field is ignored for labelling: passing it
    must not raise, and must not add a dimension."""
    metrics._on_run_end("run_end", success=True, duration_seconds=0.1, run_id="abc123", kind="root")

    label_names = {
        name
        for family in metrics.AGENT_RUNS_TOTAL.collect()
        for sample in family.samples
        for name in sample.labels
    }
    assert "run_id" not in label_names
    assert label_names == {"status", "kind"}


def test_llm_call_end_records_call_and_duration():
    calls_before = _counter_value(metrics.LLM_CALLS_TOTAL, status="success", model="gpt-4o")
    sum_before, count_before = _histogram_sum_and_count(metrics.LLM_CALL_DURATION)

    metrics._on_llm_call_end("llm_call_end", success=True, duration_seconds=2.0, model="gpt-4o")

    assert _counter_value(metrics.LLM_CALLS_TOTAL, status="success", model="gpt-4o") == calls_before + 1
    new_sum, new_count = _histogram_sum_and_count(metrics.LLM_CALL_DURATION)
    assert new_count == count_before + 1
    assert new_sum == _approx(sum_before + 2.0)


def _approx(value):
    """Small local helper -- avoids importing pytest.approx just for
    this one float-sum comparison."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < 1e-9
    return _Approx()


def test_llm_call_end_records_tokens_when_present():
    input_before = _counter_value(metrics.LLM_TOKENS_TOTAL, direction="input")
    output_before = _counter_value(metrics.LLM_TOKENS_TOTAL, direction="output")

    metrics._on_llm_call_end("llm_call_end", success=True, duration_seconds=1.0,
                              input_tokens=150, output_tokens=40)

    assert _counter_value(metrics.LLM_TOKENS_TOTAL, direction="input") == input_before + 150
    assert _counter_value(metrics.LLM_TOKENS_TOTAL, direction="output") == output_before + 40


def test_llm_call_end_does_not_record_tokens_when_absent():
    """Token usage is optional ('where available', per the handout) --
    a call with no token data must not crash or record bogus zeros
    that would corrupt the running total."""
    input_before = _counter_value(metrics.LLM_TOKENS_TOTAL, direction="input")

    metrics._on_llm_call_end("llm_call_end", success=True, duration_seconds=1.0)

    assert _counter_value(metrics.LLM_TOKENS_TOTAL, direction="input") == input_before


def test_tool_call_end_records_by_tool_name_and_status():
    before = _counter_value(metrics.TOOL_CALLS_TOTAL, tool="navigate", type="builtin", status="success")

    metrics._on_tool_call_end("tool_call_end", tool_name="navigate", tool_type="builtin", success=True, duration_seconds=0.05)

    assert _counter_value(metrics.TOOL_CALLS_TOTAL, tool="navigate", type="builtin", status="success") == before + 1


def test_tool_call_end_records_failures_separately_from_successes():
    before = _counter_value(metrics.TOOL_CALLS_TOTAL, tool="ocr_extract_text", type="mcp", status="error")

    metrics._on_tool_call_end("tool_call_end", tool_name="ocr_extract_text", tool_type="mcp", success=False, duration_seconds=0.01)

    assert _counter_value(metrics.TOOL_CALLS_TOTAL, tool="ocr_extract_text", type="mcp", status="error") == before + 1


def test_memory_op_end_records_by_operation_and_status():
    before = _counter_value(metrics.MEMORY_OPERATIONS_TOTAL, operation="remember", status="success")

    metrics._on_memory_op_end("memory_op_end", operation="remember", success=True, duration_seconds=0.02)

    assert _counter_value(metrics.MEMORY_OPERATIONS_TOTAL, operation="remember", status="success") == before + 1


def test_record_permission_decision_updates_the_right_labels():
    before = _counter_value(metrics.PERMISSION_DECISIONS_TOTAL, decision="deny", tool_name="delete_file")

    metrics.record_permission_decision("deny", "delete_file")

    after = _counter_value(metrics.PERMISSION_DECISIONS_TOTAL, decision="deny", tool_name="delete_file")
    assert after == before + 1


def test_record_subagent_run_updates_the_right_labels():
    before = _counter_value(metrics.SUBAGENT_RUNS_TOTAL, role="receipt-ocr-agent", status="success")

    metrics.record_subagent_run("receipt-ocr-agent", success=True)

    after = _counter_value(metrics.SUBAGENT_RUNS_TOTAL, role="receipt-ocr-agent", status="success")
    assert after == before + 1


def test_init_metrics_wires_listeners_onto_a_hook_manager(monkeypatch):
    """Confirms init_metrics() actually registers every listener it's
    supposed to, using a throwaway HookManager so this doesn't attach
    extra listeners onto the real shared singleton for the rest of the
    test session."""
    from agent.hooks import HookManager

    fake_hooks = HookManager()
    monkeypatch.setattr(metrics, "hooks", fake_hooks)

    metrics.init_metrics()

    for event in ("run_start", "run_end", "llm_call_end", "tool_call_end", "memory_op_end"):
        assert event in fake_hooks._listeners
        assert len(fake_hooks._listeners[event]) == 1
