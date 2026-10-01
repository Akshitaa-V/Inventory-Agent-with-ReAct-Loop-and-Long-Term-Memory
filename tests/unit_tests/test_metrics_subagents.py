"""
Unit tests for sub-agent metrics -- the per-role series read off the
run_end event.

Kept out of test_metrics.py so that file stays as its author left it.
Sub-agent runs needed no new event: a delegated run already emits
"run_end" carrying its role, kind and success, so this is read off the
event the loop was emitting anyway.

The load-bearing detail is that only a sub-run contributes. A root run
emits the same event with the same fields, and recording it here too
would make subagent_runs_total count the main agent as a sub-agent.

Prometheus counters are process-global and cumulative, so every
assertion is on the DELTA caused by one emit.
"""

import pytest

from agent import metrics
from agent.hooks import HookManager


def _count(**labels):
    return metrics.SUBAGENT_RUNS_TOTAL.labels(**labels)._value.get()


@pytest.fixture
def emit(monkeypatch):
    """Emits run_end into a throwaway HookManager with the real
    listeners registered, so the wiring is exercised rather than just
    the arithmetic."""
    manager = HookManager()
    monkeypatch.setattr(metrics, "hooks", manager)
    metrics.init_metrics()
    return lambda **payload: manager.emit("run_end", **payload)


# ── Only sub-runs count ──────────────────────────────────────────────

def test_a_successful_sub_run_is_recorded_under_its_role(emit):
    before = _count(role="inventory-auditor", status="success")

    emit(kind="sub", role="inventory-auditor", success=True, duration_seconds=0.4)

    assert _count(role="inventory-auditor", status="success") == before + 1


def test_a_failed_sub_run_is_recorded_as_an_error(emit):
    """A sub-agent that exhausts its iterations returns normally, so this
    only distinguishes it because RunResult made the loop report
    success=False for an iteration limit."""
    before = _count(role="qr-labeller", status="error")

    emit(kind="sub", role="qr-labeller", success=False, duration_seconds=0.2)

    assert _count(role="qr-labeller", status="error") == before + 1


def test_a_root_run_is_not_recorded_as_a_sub_agent(emit):
    """The main agent emits the same event with the same fields. Counting
    it here would make subagent_runs_total include the parent."""
    before = _count(role="main", status="success")

    emit(kind="root", role="main", success=True, duration_seconds=1.0)

    assert _count(role="main", status="success") == before


def test_a_run_with_no_kind_is_treated_as_root(emit):
    """Callers predating sub-agents emit no kind and are whole top-level
    runs, so they must not land in the per-role series."""
    before = _count(role="legacy", status="success")

    emit(role="legacy", success=True, duration_seconds=0.1)

    assert _count(role="legacy", status="success") == before


# ── Roles stay separate ──────────────────────────────────────────────

def test_two_roles_are_counted_separately(emit):
    """The point of the role label: 8.4 lists sub-agent activity as a
    dashboard requirement, and one bucket for all of them would not show
    which specialization is being used."""
    auditor_before = _count(role="inventory-auditor", status="success")
    labeller_before = _count(role="qr-labeller", status="success")

    emit(kind="sub", role="inventory-auditor", success=True, duration_seconds=0.1)
    emit(kind="sub", role="qr-labeller", success=True, duration_seconds=0.1)

    assert _count(role="inventory-auditor", status="success") == auditor_before + 1
    assert _count(role="qr-labeller", status="success") == labeller_before + 1


def test_a_sub_run_with_no_role_is_recorded_as_unknown(emit):
    """Instrumentation must not be the thing that breaks a run, and a
    silently dropped metric is worse than one labelled unknown."""
    before = _count(role="unknown", status="success")

    emit(kind="sub", success=True, duration_seconds=0.1)

    assert _count(role="unknown", status="success") == before + 1


# ── It does not disturb the run metrics on the same event ────────────

def test_a_sub_run_still_updates_the_kind_split_run_metrics(emit):
    """Both series are read off one event; recording the role must not
    replace recording the run."""
    runs_before = metrics.AGENT_RUNS_TOTAL.labels(status="success", kind="sub")._value.get()

    emit(kind="sub", role="inventory-auditor", success=True, duration_seconds=0.3)

    after = metrics.AGENT_RUNS_TOTAL.labels(status="success", kind="sub")._value.get()
    assert after == runs_before + 1


def test_run_end_still_has_exactly_one_listener(monkeypatch):
    """Recording the role was added inside the existing listener rather
    than as a second one on the same event, which is what keeps
    test_metrics.py's one-listener-per-event assertion true."""
    manager = HookManager()
    monkeypatch.setattr(metrics, "hooks", manager)

    metrics.init_metrics()

    assert len(manager._listeners["run_end"]) == 1
