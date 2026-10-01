"""
End-to-end tests for complex, use-case-driven scenarios.

Where test_subagent_delegation.py proves the harness delegates correctly
when told to, and test_receipt_to_inventory.py proves one straightforward
receipt reaches the inventory, these tests each chain several of the
project's own features together in the way a real insurance-claim
preparation session actually would -- multiple tool calls building on
each other, two different sub-agents delegated to within one request, and
a require-user-confirmation outcome and a deny outcome resolved side by
side in the same turn. Each scenario was run manually against the real
CLI first; these tests assert the same behaviour programmatically.

Like the other real-LLM tests, each skips without a token or a
config.json rather than failing, and uses the project's own config --
its real policy, its real sub-agent definitions -- rather than fixtures,
so a change to either that breaks one of these shows up here.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

from agent.agents import AgentCatalog, AgentSpec          # noqa: E402
from agent.config import Config, load_config              # noqa: E402
from agent.hooks import HookManager                       # noqa: E402
from agent.ltm import Memory                               # noqa: E402
from agent.permissions import Decision, PermissionPolicy, auto_approve  # noqa: E402
from agent.runner import AgentRunner                       # noqa: E402
from agent.tool_registry import DELEGATE_TOOL_NAME, ToolRegistry  # noqa: E402
from agent import metrics                                  # noqa: E402

INVENTORY = """\
| Item | Category | Price | Currency | Purchase Date | Vendor | Serial |
|------|----------|-------|----------|---------------|--------|--------|
| Sony WH-1000XM5 Headphones | electronics | 89.99 | EUR | 15 Aug 2026 | Best Buy | SNY-XM5-88213 |
| HP Compaq 6200 Pro SFF Desktop | electronics | 125.36 | USD | 29 Dec 2020 | Drake Group | HP Compaq 6200 Pro SFF |
| Desk Lamp - RANARP | furniture | 24.50 | EUR | 2 Jul 2026 | IKEA | not provided |
| Digital Blood Pressure Monitor | medical equipment | 45.00 | EUR | 20 Jun 2026 | MediShop Medical Supplies | Model: Omron M3, Serial No: OM3-2026-00457 |
"""

MAIN_INSTRUCTIONS = (
    "You are an inventory agent. You have two sub-agents available: "
    "inventory-auditor, for questions about what is already recorded, "
    "and qr-labeller, for generating QR asset tags. Use "
    "delegate_to_subagent to reach either one when the task calls for it."
)


def _config(workspace, mcp_servers=None):
    config_path = os.path.join(REPO_ROOT, "config.json")
    if not os.path.exists(config_path):
        pytest.skip("config.json not found at repo root -- cannot run E2E test")
    try:
        real = load_config(config_path)
    except RuntimeError as exc:
        pytest.skip(f"cannot run E2E test: {exc}")
    if not os.environ.get("INNKUBE_TOKEN"):
        pytest.skip("INNKUBE_TOKEN environment variable not set -- skipping E2E test")

    return Config(
        model=real.model,
        temperature=real.temperature,
        base_url=real.base_url,
        endpoint=real.endpoint,
        max_iterations=real.max_iterations,
        workspace_root=str(workspace),
        ltm_db_path=real.ltm_db_path,
        api_key=real.api_key,
        mcp_servers=mcp_servers if mcp_servers is not None else real.mcp_servers,
        permissions=real.permissions,
        agents=real.agents,
    )


def _hook_manager(monkeypatch, event_names):
    """Installs a fresh HookManager on every module that emits to hooks,
    and returns the list its listeners append events to -- the same
    pattern test_subagent_delegation.py uses, so this run's events never
    mix with anything from a previous test."""
    manager = HookManager()
    events = []
    for name in event_names:
        manager.on(name, lambda event_name, **p: events.append((event_name, p)))
    monkeypatch.setattr("agent.loop.hooks", manager)
    monkeypatch.setattr("agent.tool_registry.hooks", manager)
    monkeypatch.setattr(metrics, "hooks", manager)
    metrics.init_metrics()
    return events


def _payloads(events, name):
    return [p for event_name, p in events if event_name == name]


# ── Scenario A: chained tool calls gated by confirmation ───────────────

def test_insurance_claim_prep_chains_recall_checks(tmp_path, monkeypatch):
    """A single request that must: read the inventory, reason about which
    items qualify (electronics over a price threshold), and call the
    confirmation-gated recall tool once per qualifying item -- the kind
    of multi-step, judgment-requiring task a real claim-prep session
    needs, not a single isolated tool call."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inventory.md").write_text(INVENTORY, encoding="utf-8")

    config = _config(workspace, mcp_servers={
        "recall": {"enabled": True}, "ocr": {"enabled": False}, "qr": {"enabled": False},
    })
    policy = PermissionPolicy.from_config(config.permissions)
    events = _hook_manager(monkeypatch, ("tool_call_end", "permission_decision"))

    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers=config.mcp_servers,
        policy=policy,
    )
    runner = AgentRunner(config, tools, confirm=auto_approve)
    main_spec = AgentSpec(
        role="main", description="main", instructions=MAIN_INSTRUCTIONS,
        max_iterations=config.max_iterations, policy=policy,
    )

    prompt = (
        "Go through everything already on record, and for anything worth "
        "over 100 EUR or USD that's electronics, check whether it has a "
        "safety recall. Then give me a short claim-ready summary: item, "
        "value, and recall status."
    )
    result = runner.run(main_spec, prompt, run_id="e2e-claim-prep")

    recall_calls = [
        p for p in _payloads(events, "tool_call_end")
        if p.get("tool_name") == "check_product_recall"
    ]
    assert recall_calls, (
        f"the agent never checked for recalls. It answered: {result}"
    )

    confirm_decisions = [
        p for p in _payloads(events, "permission_decision")
        if p.get("tool_name") == "check_product_recall"
    ]
    assert confirm_decisions, "no permission decision recorded for check_product_recall"
    assert all(d["decision"] == Decision.REQUIRE_USER_CONFIRMATION.value for d in confirm_decisions)
    assert all(d["allowed"] is True for d in confirm_decisions), (
        "auto_approve was supplied, so every confirmation should have been granted"
    )

    assert result.success is True, f"the run did not finish cleanly: {result.reason}"


# ── Scenario B: two different sub-agents delegated to in one request ───

def test_delegation_chains_two_different_subagents(tmp_path, monkeypatch):
    """One request that needs both configured sub-agents in sequence:
    inventory-auditor to look a fact up, then qr-labeller to act on what
    it found. This is what distinguishes "delegation exists" from
    "delegation composes" -- the second sub-agent's input depends on the
    first sub-agent's output, and both must still be attributed to the
    same parent run."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inventory.md").write_text(INVENTORY, encoding="utf-8")

    config = _config(workspace, mcp_servers={
        "recall": {"enabled": False}, "ocr": {"enabled": False}, "qr": {"enabled": True},
    })
    policy = PermissionPolicy.from_config(config.permissions)
    catalog = AgentCatalog.from_config(config.agents)
    catalog.require_narrower_than(policy)
    assert "inventory-auditor" in catalog and "qr-labeller" in catalog, (
        "the shipped config must define both sub-agents for this scenario"
    )

    events = _hook_manager(monkeypatch, ("run_end", "tool_call_end"))

    subagent_metric_before = {
        role: metrics.SUBAGENT_RUNS_TOTAL.labels(role=role, status="success")._value.get()
        for role in ("inventory-auditor", "qr-labeller")
    }

    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers=config.mcp_servers,
        policy=policy,
    )
    runner = AgentRunner(config, tools, confirm=None, catalog=catalog)
    main_spec = AgentSpec(
        role="main", description="main", instructions=MAIN_INSTRUCTIONS,
        max_iterations=config.max_iterations, policy=policy,
    )

    prompt = (
        "Use the inventory-auditor to find the exact price and serial "
        "number recorded for the Sony WH-1000XM5 Headphones, then use "
        "the qr-labeller to generate an asset tag QR code encoding that "
        "price and serial number together."
    )
    result = runner.run(main_spec, prompt, run_id="e2e-chained-delegation")

    sub_runs = {p["role"]: p for p in _payloads(events, "run_end") if p.get("kind") == "sub"}
    assert "inventory-auditor" in sub_runs, (
        f"inventory-auditor was never delegated to. Result: {result}"
    )
    assert "qr-labeller" in sub_runs, (
        f"qr-labeller was never delegated to. Result: {result}"
    )
    for role, run in sub_runs.items():
        assert run["parent_run_id"] == "e2e-chained-delegation", (
            f"{role}'s sub-run was not attributed to the parent run"
        )
        assert run["success"] is True, f"{role} did not finish: {run.get('reason')}"

    delegate_calls = [
        p for p in _payloads(events, "tool_call_end")
        if p.get("tool_name") == DELEGATE_TOOL_NAME
    ]
    assert len(delegate_calls) >= 2, (
        f"expected at least two delegations, saw {len(delegate_calls)}"
    )

    qr_calls = [
        p for p in _payloads(events, "tool_call_end")
        if p.get("tool_name") == "generate_qr_code"
    ]
    assert qr_calls, f"qr-labeller never generated a code. Result: {result}"

    qr_dir = workspace / "qr_codes"
    assert qr_dir.is_dir() and any(qr_dir.iterdir()), (
        "no QR code file was written to workspace/qr_codes"
    )

    for role in ("inventory-auditor", "qr-labeller"):
        after = metrics.SUBAGENT_RUNS_TOTAL.labels(role=role, status="success")._value.get()
        assert after == subagent_metric_before[role] + 1, (
            f"subagent_runs_total did not move for {role}"
        )

    assert result.success is True, f"the parent run failed: {result.reason}"


# ── Scenario C: confirmation approval and denial in the same turn ──────

def test_denied_tool_stays_blocked_beside_an_approved_one(tmp_path, monkeypatch):
    """One request that mixes a require-user-confirmation outcome (which
    auto_approve grants) with a deny outcome (forget_fact), in the same
    turn. The point is that deny must win regardless of the confirmation
    callback: this is the one case a shallower test could get right by
    accident if it only ever exercised the two outcomes separately."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inventory.md").write_text(INVENTORY, encoding="utf-8")

    ltm_path = tmp_path / "chroma"
    memory = Memory(db_path=str(ltm_path))
    memory.remember("user", "prefers", "metric units")
    facts_before = memory.recall_all()
    assert "metric" in facts_before

    config = _config(workspace, mcp_servers={
        "recall": {"enabled": True}, "ocr": {"enabled": False}, "qr": {"enabled": False},
    })
    config = Config(**{**config.__dict__, "ltm_db_path": str(ltm_path)})
    policy = PermissionPolicy.from_config(config.permissions)
    assert policy.decide("forget_fact") is Decision.DENY, (
        "the shipped config must deny forget_fact for this scenario to prove anything"
    )

    events = _hook_manager(monkeypatch, ("permission_decision", "tool_call_end"))

    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "receipt_memory.json"),
        memory=memory,
        mcp_servers=config.mcp_servers,
        policy=policy,
    )
    runner = AgentRunner(config, tools, confirm=auto_approve)
    main_spec = AgentSpec(
        role="main", description="main", instructions=MAIN_INSTRUCTIONS,
        max_iterations=config.max_iterations, policy=policy,
    )

    prompt = (
        "Check the Digital Blood Pressure Monitor for recalls, and also "
        "forget that I prefer metric units."
    )
    result = runner.run(main_spec, prompt, run_id="e2e-mixed-outcomes")

    decisions = _payloads(events, "permission_decision")
    recall_decision = next((d for d in decisions if d["tool_name"] == "check_product_recall"), None)
    forget_decision = next((d for d in decisions if d["tool_name"] == "forget_fact"), None)

    assert recall_decision is not None, f"check_product_recall was never evaluated. Result: {result}"
    assert recall_decision["decision"] == Decision.REQUIRE_USER_CONFIRMATION.value
    assert recall_decision["allowed"] is True, "auto_approve should have granted this one"

    assert forget_decision is not None, f"forget_fact was never evaluated. Result: {result}"
    assert forget_decision["decision"] == Decision.DENY.value
    assert forget_decision["allowed"] is False, (
        "forget_fact must stay blocked even though the confirm callback "
        "auto-approves everything -- deny must never consult it"
    )

    assert not any(
        p["tool_name"] == "forget_fact"
        for p in _payloads(events, "tool_call_end")
    ), "forget_fact must never actually execute when denied"

    facts_after = memory.recall_all()
    assert "metric" in facts_after, (
        "the fact was removed despite forget_fact being denied by policy"
    )

    assert result.success is True, f"the run did not finish cleanly: {result.reason}"