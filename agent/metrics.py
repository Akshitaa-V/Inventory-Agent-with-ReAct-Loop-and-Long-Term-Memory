"""
agent/metrics.py -- Prometheus instrumentation, wired to agent/hooks.py.

This is the ONLY file in the codebase that imports prometheus_client.
Every other file (loop.py, tool_registry.py, ltm.py, llm_client.py) only
ever touches agent/hooks.py -- it has no idea Prometheus exists. That's
the decoupling the handout asks for: swap Prometheus for something else
later by editing only this file.

Metrics defined here cover the full dashboard requirement list, even
though not all of them are wired up yet:
  - Agent throughput / active runs        -> wired now (root vs sub via "kind")
  - Agent-run and component latency        -> wired now
  - Success and failure rates              -> wired now (via "status" label)
  - LLM usage incl. token consumption      -> wired now
  - Built-in and MCP tool usage            -> wired now
  - Permission decisions                   -> wired now
  - Sub-agent activity                     -> wired now (by role)

Permissions and sub-agents are both wired through the events above, so
nothing outside this file knows Prometheus is involved. Sub-agent runs
needed no new event: a delegated run already emits "run_end" carrying
its role and kind, so the per-role series is read off the event the loop
was emitting anyway.
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from agent.hooks import hooks

# ── Agent runs ──────────────────────────────────────────────────────

# "kind" separates a root run from a sub-agent run delegated inside one.
# A sub-run's duration sits *within* its parent's, so without this label
# summing the histogram gives a wall-clock total larger than real elapsed
# time, and throughput counts a delegated task twice.
#
# AGENT_ACTIVE_RUNS deliberately has no such label. It measures what is
# happening right now, and a nested sub-run genuinely is a second thing
# in flight -- there is nothing double-counted to separate. Only the two
# metrics that aggregate over time need the split.
#
# Note what is NOT a label here: run_id. It is carried on every run,
# llm_call, tool_call and permission_decision event for structured logs
# and traces, but a label taking a fresh value on every run would create
# one Prometheus time series per run, growing without bound until the
# server runs out of memory. Per-run inspection belongs in logs; metrics
# stay aggregate.
AGENT_RUNS_TOTAL = Counter(
    "agent_runs_total", "Total number of agent runs, by outcome and kind.",
    ["status", "kind"],
)
AGENT_ACTIVE_RUNS = Gauge(
    "agent_active_runs", "Number of agent runs currently in progress."
)
AGENT_RUN_DURATION = Histogram(
    "agent_run_duration_seconds", "Wall-clock duration of a full agent run.",
    ["kind"],
)

# ── LLM calls ────────────────────────────────────────────────────────

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total", "Total number of LLM API calls, by outcome and model.",
    ["status", "model"],
)
LLM_CALL_DURATION = Histogram(
    "llm_call_duration_seconds", "Duration of a single LLM API call."
)
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total", "Total LLM tokens consumed, by direction.", ["direction"]
)

# ── Tool calls (built-in and MCP alike -- same metric, no special-casing) ──

TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total", "Total tool invocations, by tool, type and outcome.",
    ["tool", "type", "status"],
)
TOOL_CALL_DURATION = Histogram(
    "tool_call_duration_seconds", "Duration of a single tool call.", ["tool"]
)

# ── Memory operations ────────────────────────────────────────────────

MEMORY_OPERATIONS_TOTAL = Counter(
    "memory_operations_total", "Total memory operations, by kind and outcome.",
    ["operation", "status"],
)
MEMORY_OPERATION_DURATION = Histogram(
    "memory_operation_duration_seconds", "Duration of a single memory operation.",
    ["operation"],
)

# ── Permission decisions (built-in and MCP alike, one increment per check) ──

PERMISSION_DECISIONS_TOTAL = Counter(
    "permission_decisions_total", "Permission checks, by decision outcome.",
    ["decision", "tool_name"],
)

# ── Sub-agent activity (defined now, wired by the sub-agent feature) ────

SUBAGENT_RUNS_TOTAL = Counter(
    "subagent_runs_total", "Sub-agent delegated runs, by role and outcome.",
    ["role", "status"],
)


def _status_label(success: bool) -> str:
    return "success" if success else "error"


# ── Hook listeners: this is the actual wiring. Nothing else in the
# codebase needs to know these functions exist. ─────────────────────

def _on_run_start(event_name, **payload):
    AGENT_ACTIVE_RUNS.inc()


def _on_run_end(event_name, **payload):
    AGENT_ACTIVE_RUNS.dec()
    status = _status_label(payload.get("success", True))
    # Defaults to "root" so a caller that emits a run event without the
    # label -- an older test, or code predating sub-agents -- still lands
    # in the series that means "a whole top-level run".
    kind = payload.get("kind", "root")
    AGENT_RUNS_TOTAL.labels(status=status, kind=kind).inc()
    AGENT_RUN_DURATION.labels(kind=kind).observe(payload.get("duration_seconds", 0))

    # A delegated run additionally gets a per-role series. Handled here
    # rather than by a second "run_end" listener because the two are one
    # concern -- what to record when a run ends -- and because a second
    # listener on this event would break the assertion in test_metrics.py
    # that each event has exactly one.
    if kind == "sub":
        record_subagent_run(payload.get("role", "unknown"), payload.get("success", True))


def _on_llm_call_end(event_name, **payload):
    status = _status_label(payload.get("success", True))
    model = payload.get("model", "unknown")
    LLM_CALLS_TOTAL.labels(status=status, model=model).inc()
    LLM_CALL_DURATION.observe(payload.get("duration_seconds", 0))
    if payload.get("input_tokens") is not None:
        LLM_TOKENS_TOTAL.labels(direction="input").inc(payload["input_tokens"])
    if payload.get("output_tokens") is not None:
        LLM_TOKENS_TOTAL.labels(direction="output").inc(payload["output_tokens"])


def _on_tool_call_end(event_name, **payload):
    tool = payload.get("tool_name", "unknown")
    tool_type = payload.get("tool_type", "unknown")
    status = _status_label(payload.get("success", True))
    TOOL_CALLS_TOTAL.labels(tool=tool, type=tool_type, status=status).inc()
    TOOL_CALL_DURATION.labels(tool=tool).observe(payload.get("duration_seconds", 0))


def _on_memory_op_end(event_name, **payload):
    operation = payload.get("operation", "unknown")
    status = _status_label(payload.get("success", True))
    MEMORY_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()
    MEMORY_OPERATION_DURATION.labels(operation=operation).observe(payload.get("duration_seconds", 0))


def _on_permission_decision(event_name, **payload):
    """Records one permission check.

    Emitted by the loop for every check it makes, and by ToolRegistry
    only when it refuses -- the registry is reached solely for calls the
    loop already allowed, so the two never both count the same check.

    A require-user-confirmation outcome is recorded ONCE, under
    confirm_approved or confirm_denied, rather than once for the policy
    outcome and again for the user's answer. Counting it twice would
    make the sum over this counter larger than the number of checks
    actually made, so "how many permission checks happened" would stop
    being answerable. Both facts survive the single increment: the
    confirm_ prefix is the policy outcome, the suffix is the resolution,
    and `confirm_approved + confirm_denied` recovers the number of calls
    that hit a confirmation gate.
    """
    decision = payload.get("decision") or "unknown"
    if decision == "require-user-confirmation":
        decision = "confirm_approved" if payload.get("allowed") else "confirm_denied"
    record_permission_decision(decision, payload.get("tool_name", "unknown"))


def init_metrics() -> None:
    """Registers all hook listeners. Call this once at startup, before
    the first run -- e.g. from agent/main.py, right after `hooks` would
    otherwise be used. Idempotent-ish: calling it twice will just
    register duplicate listeners, so call it exactly once per process."""
    hooks.on("run_start", _on_run_start)
    hooks.on("run_end", _on_run_end)
    hooks.on("llm_call_end", _on_llm_call_end)
    hooks.on("tool_call_end", _on_tool_call_end)
    hooks.on("memory_op_end", _on_memory_op_end)
    hooks.on("permission_decision", _on_permission_decision)


def start_metrics_server(port: int = 8000) -> None:
    """Starts the Prometheus-compatible /metrics HTTP endpoint in a
    background thread. Call once at agent startup. A Prometheus server
    (configured separately, by whoever owns that piece) scrapes this
    endpoint on an interval."""
    start_http_server(port)


# ── Ready-made functions for the permission-management and sub-agent
# features to call, without needing to know anything about Prometheus. ──

def record_permission_decision(decision: str, tool_name: str) -> None:
    """Call this whenever a permission decision is made.

    `decision` is one of four values, exactly one increment per check:
      'allow'            -- policy allowed it outright
      'deny'             -- policy refused it outright
      'confirm_approved' -- policy required confirmation, user agreed
      'confirm_denied'   -- policy required confirmation, it was refused
                            (the user declined, nobody could be asked, or
                            the prompt itself failed)

    The last two replace a bare 'require-user-confirmation' so that the
    policy outcome and the user's resolution are both readable from the
    label without counting the same check twice. Callers that want the
    number of calls that hit a confirmation gate sum the two.

    In practice this is called by _on_permission_decision above rather
    than directly; it stays public so a non-hook caller can report a
    decision without knowing Prometheus is involved.
    """
    PERMISSION_DECISIONS_TOTAL.labels(decision=decision, tool_name=tool_name).inc()


def record_subagent_run(role: str, success: bool) -> None:
    """Call this from the sub-agent delegation mechanism whenever a
    sub-agent run completes. `role` is the sub-agent's specialized role
    (e.g. 'receipt-ocr-agent', 'recall-checker-agent')."""
    SUBAGENT_RUNS_TOTAL.labels(role=role, status=_status_label(success)).inc()
