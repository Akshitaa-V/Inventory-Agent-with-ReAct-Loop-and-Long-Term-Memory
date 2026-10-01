"""
End-to-end test for sub-agent delegation, against the real LLM.

This is the test the handout (8.4) asks for: "at least one end-to-end
test covering sub-agent delegation, permission enforcement, tool
execution, and observability". All four are asserted below on a single
real run, because the point is that they hold together, not separately.

It uses the project's own config.json -- the real policy and the real
sub-agent definitions -- rather than fixtures, so a change to either
that breaks delegation shows up here. MCP servers are switched off: the
delegation under test uses built-in tools only, and spawning three
subprocesses would just make the test slower.

Like the other real-LLM tests it skips without a token rather than
failing. It does instruct the model to delegate explicitly: whether a
model *chooses* to delegate is a prompting question, and this test is
about whether the harness delegates correctly when it does.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

from agent.agents import AgentCatalog, AgentSpec          # noqa: E402
from agent.config import Config, load_config              # noqa: E402
from agent.hooks import HookManager                       # noqa: E402
from agent.permissions import Decision, PermissionPolicy  # noqa: E402
from agent.runner import AgentRunner                      # noqa: E402
from agent.tool_registry import DELEGATE_TOOL_NAME, ToolRegistry  # noqa: E402
from agent import metrics                                 # noqa: E402

INVENTORY = """\
| Item | Category | Price | Currency | Purchase Date | Vendor | Serial |
|------|----------|-------|----------|---------------|--------|--------|
| Dell XPS 13 | electronics | 1199.99 | USD | 15 Aug 2026 | Best Buy | DXP-13-45927 |
| Office Chair | furniture | 249.00 | USD | 2 Mar 2026 | IKEA | not provided |
"""

DELEGATION_PROMPT = (
    "Use the inventory-auditor sub-agent to find out what purchase price is "
    "recorded in the inventory for the Dell XPS 13. Delegate that lookup to "
    "the sub-agent with delegate_to_subagent rather than reading inventory.md "
    "yourself, then tell me the price it reports."
)


def _config(workspace):
    config_path = os.path.join(REPO_ROOT, "config.json")
    if not os.path.exists(config_path):
        pytest.skip("config.json not found at repo root -- cannot run E2E test")
    try:
        real = load_config(config_path)
    except RuntimeError as exc:
        pytest.skip(f"cannot run E2E test: {exc}")

    return Config(
        model=real.model,
        temperature=real.temperature,
        base_url=real.base_url,
        endpoint=real.endpoint,
        max_iterations=real.max_iterations,
        workspace_root=str(workspace),
        ltm_db_path=real.ltm_db_path,
        api_key=real.api_key,
        # Off on purpose: this delegation needs only built-in tools, and
        # three subprocess spawns would make the test slower for nothing.
        mcp_servers={"recall": {"enabled": False}, "ocr": {"enabled": False},
                     "qr": {"enabled": False}},
        permissions=real.permissions,
        agents=real.agents,
    )


def test_delegation_end_to_end(tmp_path, monkeypatch):
    """One real run, covering all four things 8.4 asks of this test."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inventory.md").write_text(INVENTORY, encoding="utf-8")

    config = _config(workspace)
    policy = PermissionPolicy.from_config(config.permissions)
    catalog = AgentCatalog.from_config(config.agents)

    # The startup check main.py runs: no sub-agent may exceed the parent.
    catalog.require_narrower_than(policy)
    assert "inventory-auditor" in catalog, "the shipped config defines no auditor"

    # A fresh hook manager so the assertions see only this run's events.
    manager = HookManager()
    events = []
    for name in ("run_start", "run_end", "tool_call_end", "permission_decision"):
        manager.on(name, lambda event_name, **p: events.append((event_name, p)))
    monkeypatch.setattr("agent.loop.hooks", manager)
    monkeypatch.setattr("agent.tool_registry.hooks", manager)
    # The real listeners too, so the Prometheus series are exercised.
    monkeypatch.setattr(metrics, "hooks", manager)
    metrics.init_metrics()

    subagent_before = metrics.SUBAGENT_RUNS_TOTAL.labels(
        role="inventory-auditor", status="success"
    )._value.get()

    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers=config.mcp_servers,
        policy=policy,
    )
    runner = AgentRunner(config, tools, confirm=None, catalog=catalog)
    main_spec = AgentSpec(
        role="main",
        description="The inventory agent the user talks to directly.",
        instructions=(
            "You are an inventory agent. You have a sub-agent available for "
            "questions about what is already recorded in the inventory. When "
            "the user asks you to delegate, use delegate_to_subagent."
        ),
        max_iterations=config.max_iterations,
        policy=policy,
    )

    result = runner.run(main_spec, DELEGATION_PROMPT, run_id="e2e-parent")

    def payloads(name):
        return [p for event_name, p in events if event_name == name]

    # --- 1. Delegation happened -------------------------------------
    delegate_calls = [
        p for p in payloads("tool_call_end")
        if p.get("tool_name") == DELEGATE_TOOL_NAME
    ]
    assert delegate_calls, (
        "the agent never called delegate_to_subagent. Tools it did call: "
        f"{[p.get('tool_name') for p in payloads('tool_call_end')]}. "
        f"It answered: {result}"
    )
    assert delegate_calls[0]["tool_type"] == "builtin", (
        "delegation was reported as an MCP tool call"
    )

    # --- 2. The sub-run ran, and is tied to its parent --------------
    sub_runs = [p for p in payloads("run_end") if p.get("kind") == "sub"]
    assert sub_runs, f"no sub-run was recorded. The agent answered: {result}"
    sub = sub_runs[0]
    assert sub["role"] == "inventory-auditor"
    assert sub["parent_run_id"] == "e2e-parent", (
        "the sub-run was not recorded as part of its parent run"
    )
    assert sub["run_id"] != "e2e-parent"
    assert sub["success"] is True, (
        f"the sub-agent did not finish: {sub.get('reason')}"
    )

    root_runs = [p for p in payloads("run_end") if p.get("kind") == "root"]
    assert len(root_runs) == 1
    assert root_runs[0]["role"] == "main"

    # --- 3. Tool execution, inside the sub-run ----------------------
    read_calls = [
        p for p in payloads("tool_call_end")
        if p.get("tool_name") in {"read", "read_many", "search", "navigate"}
    ]
    assert read_calls, (
        "the sub-agent never read anything, so it cannot have audited the "
        f"inventory. Tools called: {[p.get('tool_name') for p in payloads('tool_call_end')]}"
    )
    assert all(p["run_id"] == sub["run_id"] for p in read_calls), (
        "the reads were not attributed to the sub-run"
    )

    # --- 4. Permission enforcement ----------------------------------
    # The auditor's own policy denies everything it was not granted, and
    # that policy is what its run was given.
    auditor_policy = catalog.get("inventory-auditor").policy
    assert auditor_policy.decide("read") is Decision.ALLOW
    assert auditor_policy.decide("modify") is Decision.DENY, (
        "the auditor can write, so this run proves nothing about enforcement"
    )
    # Every decision recorded during the sub-run was made under that policy,
    # so nothing it did fell outside what the auditor was allowed.
    sub_decisions = [p for p in payloads("permission_decision")
                     if p.get("run_id") == sub["run_id"]]
    assert sub_decisions, "no permission decision was recorded for the sub-run"
    for decision in sub_decisions:
        assert decision["allowed"] is True, (
            f"the sub-agent was refused {decision['tool_name']!r}, which means "
            f"its policy and its tool list disagree"
        )
        assert auditor_policy.decide(decision["tool_name"]) is not Decision.DENY

    # The sub-agent was never offered a tool its policy denies.
    assert not any(
        p.get("tool_name") == "modify" for p in payloads("tool_call_end")
        if p.get("run_id") == sub["run_id"]
    )

    # --- Observability: the Prometheus series moved -----------------
    subagent_after = metrics.SUBAGENT_RUNS_TOTAL.labels(
        role="inventory-auditor", status="success"
    )._value.get()
    assert subagent_after == subagent_before + 1, (
        "the sub-agent run did not reach subagent_runs_total, so the "
        "dashboard's Sub-agent Activity panel would stay empty"
    )

    # --- The result came back and was used --------------------------
    assert result.success is True, f"the parent run failed: {result.reason}"
    assert "1199" in str(result) or "1,199" in str(result), (
        f"the price the sub-agent found did not reach the final answer. "
        f"The agent said: {result}"
    )
