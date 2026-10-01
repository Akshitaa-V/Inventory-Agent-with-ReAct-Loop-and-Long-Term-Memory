"""
Unit tests for delegation -- the delegate_to_subagent tool, and the
depth limit that bounds it.

The tool lives on RegistryView rather than on the shared ToolRegistry,
which is what these tests exercise. The reason is that delegation is the
only capability that differs per *run*: it needs the run's identity to
record the parent link, the run's policy to check a child against, and
the run's depth to know whether it may delegate at all. A view is built
per run and can hold those; the shared registry cannot, and giving it a
mutable back-reference to the runner would have been the alternative.

Two consequences worth stating, because both are tested below. A caller
holding the raw registry cannot delegate at all -- which is correct,
since delegation without a run context is meaningless. And the depth
limit is enforced by the tool simply not being offered, rather than
being offered and refused, so the model never spends an iteration asking
for something it was never going to be allowed to do.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from agent.agents import AgentCatalog, AgentSpec
from agent.delegation import Delegation, ParentRun
from agent.hooks import HookManager
from agent.permissions import PermissionPolicy
from agent.runner import AgentRunner
from agent.tool_registry import DELEGATE_TOOL_NAME, RegistryView


@dataclass
class _Config:
    model: str = "big-model"
    temperature: float = 0.7
    base_url: str = "http://example"
    endpoint: str = "/chat"
    api_key: str = "token"
    max_iterations: int = 10
    workspace_root: str = "./workspace"
    ltm_db_path: str = "data/chroma"
    mcp_servers: dict = field(default_factory=dict)
    permissions: dict = field(default_factory=dict)
    agents: dict = field(default_factory=dict)


def _registry(names=("read", "modify")):
    registry = MagicMock()
    registry.schemas.return_value = [
        {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
        for n in names
    ]
    registry.call.return_value = "tool ran"
    registry.is_builtin.return_value = True
    return registry


def _catalog(max_depth=1, **roles):
    if not roles:
        roles = {
            "auditor": {
                "description": "Answers one question about the existing inventory.",
                "instructions": "You are an auditor.",
                "max_iterations": 3,
                "permissions": {"default": "deny", "tools": {"read": "allow"}},
            },
        }
    return AgentCatalog.from_config({"max_delegation_depth": max_depth, "roles": roles})


def _main_spec(policy=None, max_iterations=5):
    return AgentSpec(
        role="main",
        description="The agent the user talks to.",
        instructions="You are the main agent.",
        max_iterations=max_iterations,
        policy=policy if policy is not None else PermissionPolicy(default="allow"),
    )


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


class _ScriptedLLM:
    """Replays a fixed sequence of decisions, per model name, so a parent
    and its sub-agent can each be scripted independently."""

    scripts = {}
    seen = []

    def __init__(self, config):
        self.config = config

    def get_next_step(self, messages, tools=None):
        _ScriptedLLM.seen.append({
            "model": self.config.model,
            "tools": [] if not tools else _names(tools),
            "messages": list(messages),
        })
        script = _ScriptedLLM.scripts.setdefault(self.config.model, [])
        if script:
            return script.pop(0)
        return {"type": "final_answer", "content": "fallback", "model": self.config.model}


@pytest.fixture
def scripted(monkeypatch):
    _ScriptedLLM.scripts = {}
    _ScriptedLLM.seen = []
    monkeypatch.setattr("agent.runner.LLMClient", _ScriptedLLM)
    return _ScriptedLLM


def _delegate_call(role, task, call_id="c1"):
    return {
        "type": "tool_call", "id": call_id, "tool_name": DELEGATE_TOOL_NAME,
        "arguments": {"role": role, "task": task}, "model": "big-model",
    }


def _runner(registry=None, catalog=None, confirm=None):
    return AgentRunner(
        _Config(), registry or _registry(),
        confirm=confirm, catalog=catalog if catalog is not None else _catalog(),
    )


# ── The tool is offered, with the roles in its schema ────────────────

def test_the_delegate_tool_is_offered_to_a_run_that_may_delegate():
    catalog = _catalog()
    view = RegistryView(
        _registry(), PermissionPolicy(default="allow"),
        delegation=Delegation(
            runner=MagicMock(), catalog=catalog,
            parent=ParentRun("p1", PermissionPolicy(default="allow"), 0),
        ),
    )

    assert DELEGATE_TOOL_NAME in _names(view.schemas())


def test_the_schema_lists_the_configured_roles_as_an_enum():
    """So the model cannot invent a role name."""
    catalog = _catalog(
        auditor={"description": "Reads the inventory and answers.", "instructions": "x"},
        labeller={"description": "Makes QR labels for items.", "instructions": "y"},
    )
    view = RegistryView(
        _registry(), PermissionPolicy(default="allow"),
        delegation=Delegation(
            runner=MagicMock(), catalog=catalog,
            parent=ParentRun("p1", PermissionPolicy(default="allow"), 0),
        ),
    )

    schema = [s for s in view.schemas() if s["function"]["name"] == DELEGATE_TOOL_NAME][0]

    assert schema["function"]["parameters"]["properties"]["role"]["enum"] == [
        "auditor", "labeller",
    ]


def test_the_schema_describes_each_role_so_the_model_can_choose():
    catalog = _catalog(
        auditor={"description": "Reads the inventory and answers questions.",
                 "instructions": "x"},
        labeller={"description": "Makes QR labels for inventory items.",
                  "instructions": "y"},
    )
    view = RegistryView(
        _registry(), PermissionPolicy(default="allow"),
        delegation=Delegation(
            runner=MagicMock(), catalog=catalog,
            parent=ParentRun("p1", PermissionPolicy(default="allow"), 0),
        ),
    )

    description = [
        s for s in view.schemas() if s["function"]["name"] == DELEGATE_TOOL_NAME
    ][0]["function"]["description"]

    assert "Reads the inventory and answers questions." in description
    assert "Makes QR labels for inventory items." in description


def test_the_delegate_tool_counts_as_builtin():
    """Delegation is part of the harness, not something an MCP server
    provides, so it must not be reported as an MCP tool call."""
    view = RegistryView(
        _registry(), PermissionPolicy(default="allow"),
        delegation=Delegation(
            runner=MagicMock(), catalog=_catalog(),
            parent=ParentRun("p1", PermissionPolicy(default="allow"), 0),
        ),
    )

    assert view.is_builtin(DELEGATE_TOOL_NAME) is True


# ── A view with no delegation cannot delegate ────────────────────────

def test_a_view_without_delegation_does_not_offer_the_tool():
    view = RegistryView(_registry(), PermissionPolicy(default="allow"))

    assert DELEGATE_TOOL_NAME not in _names(view.schemas())


def test_a_view_without_delegation_refuses_the_call():
    view = RegistryView(_registry(), PermissionPolicy(default="allow"))

    result = view.call(DELEGATE_TOOL_NAME, {"role": "auditor", "task": "x"})

    assert result.startswith("Error:")
    assert "not available" in result


def test_the_raw_registry_cannot_delegate_at_all(tmp_path):
    """Correct rather than a gap: delegation without a run context has no
    identity to record, no policy to check against and no depth."""
    from agent.tool_registry import ToolRegistry

    registry = ToolRegistry(
        workspace_root=str(tmp_path),
        receipt_memory_path=str(tmp_path / "m.json"),
        mcp_servers={"recall": {"enabled": False}, "ocr": {"enabled": False},
                     "qr": {"enabled": False}},
    )

    assert DELEGATE_TOOL_NAME not in _names(registry.schemas())
    assert "unknown tool" in registry.call(DELEGATE_TOOL_NAME, {})


def test_an_empty_catalog_means_the_tool_is_not_offered():
    view = RegistryView(
        _registry(), PermissionPolicy(default="allow"),
        delegation=Delegation(
            runner=MagicMock(), catalog=AgentCatalog.from_config(None),
            parent=ParentRun("p1", PermissionPolicy(default="allow"), 0),
        ),
    )

    assert DELEGATE_TOOL_NAME not in _names(view.schemas())


# ── Delegation end to end through the runner ─────────────────────────

def test_the_parent_gets_the_sub_agents_result_as_a_tool_result(scripted):
    # Interleaved, because that is the real order: the parent asks, the
    # sub-run answers inside that tool call, then the parent answers.
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "Is the Dell XPS already recorded?"),
        {"type": "final_answer", "content": "Recorded 15 Aug 2026 at 1199.99.",
         "model": "big-model"},
        {"type": "final_answer", "content": "Yes, recorded on 15 Aug.",
         "model": "big-model"},
    ]

    result = _runner().run(_main_spec(), "check the Dell")

    assert result == "Yes, recorded on 15 Aug."
    assert result.success is True

    # The sub-agent's answer reached the parent as the tool's result.
    tool_results = [m for m in scripted.seen[-1]["messages"] if m["role"] == "tool"]
    assert tool_results[0]["content"] == "Recorded 15 Aug 2026 at 1199.99."


def test_the_sub_agent_runs_with_its_own_instructions_and_a_fresh_context(scripted):
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "the delegated task"),
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ]

    _runner().run(_main_spec(), "parent task")

    # The sub-run's first request: its own system prompt, and only the
    # delegated task as the user message -- not the parent conversation.
    sub_request = scripted.seen[1]
    assert sub_request["messages"][0] == {
        "role": "system", "content": "You are an auditor.",
    }
    users = [m["content"] for m in sub_request["messages"] if m["role"] == "user"]
    assert users == ["the delegated task"]
    assert "parent task" not in str(sub_request["messages"])


def test_the_sub_agent_sees_only_the_tools_its_own_policy_allows(scripted):
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "task"),
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ]

    _runner(_registry(("read", "modify"))).run(_main_spec(), "parent task")

    assert scripted.seen[1]["tools"] == ["read"]


def test_a_sub_agent_failure_is_surfaced_not_returned_as_an_answer(scripted):
    """A sub-agent that runs out of iterations returns an ordinary
    string; without this the parent would treat it as a result."""
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "task"),
        # The sub-agent never answers, so it hits its 3-iteration cap.
        {"type": "tool_call", "id": "s1", "tool_name": "read", "arguments": {},
         "model": "big-model"},
        {"type": "tool_call", "id": "s2", "tool_name": "read", "arguments": {},
         "model": "big-model"},
        {"type": "tool_call", "id": "s3", "tool_name": "read", "arguments": {},
         "model": "big-model"},
        {"type": "final_answer", "content": "the sub-agent gave up", "model": "big-model"},
    ]

    _runner().run(_main_spec(), "parent task")

    parent_context = scripted.seen[-1]["messages"]
    tool_results = [m for m in parent_context if m["role"] == "tool"]
    assert "did not finish" in tool_results[-1]["content"]
    assert "iteration_limit" in tool_results[-1]["content"]


def test_an_unknown_role_is_reported_back_to_the_model(scripted):
    scripted.scripts["big-model"] = [
        _delegate_call("no-such-agent", "task"),
        {"type": "final_answer", "content": "cannot", "model": "big-model"},
    ]

    _runner().run(_main_spec(), "parent task")

    tool_results = [m for m in scripted.seen[-1]["messages"] if m["role"] == "tool"]
    assert "no sub-agent" in tool_results[0]["content"]
    assert "auditor" in tool_results[0]["content"]


def test_an_empty_task_is_reported_back_to_the_model(scripted):
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "   "),
        {"type": "final_answer", "content": "cannot", "model": "big-model"},
    ]

    _runner().run(_main_spec(), "parent task")

    tool_results = [m for m in scripted.seen[-1]["messages"] if m["role"] == "tool"]
    assert "non-empty instruction" in tool_results[0]["content"]


# ── The parent link, and the role, in the observability data ─────────

@pytest.fixture
def run_events(monkeypatch):
    manager = HookManager()
    captured = []
    for name in ("run_start", "run_end"):
        manager.on(name, lambda event_name, **p: captured.append((event_name, p)))
    monkeypatch.setattr("agent.loop.hooks", manager)
    return captured


def test_the_sub_run_records_which_run_delegated_it(scripted, run_events):
    """8.4: sub-agent executions recorded as part of the corresponding
    parent run."""
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "task"),
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ]

    _runner().run(_main_spec(), "parent task", run_id="parent-run")

    starts = [p for name, p in run_events if name == "run_start"]
    parent, child = starts[0], starts[1]

    assert parent["run_id"] == "parent-run"
    assert parent["kind"] == "root"
    assert parent["role"] == "main"

    assert child["parent_run_id"] == "parent-run"
    assert child["kind"] == "sub"
    assert child["role"] == "auditor"
    assert child["run_id"] != "parent-run"


def test_both_runs_close_their_spans(scripted, run_events):
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "task"),
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ]

    _runner().run(_main_spec(), "parent task")

    ends = [p for name, p in run_events if name == "run_end"]
    assert sorted(p["kind"] for p in ends) == ["root", "sub"]


# ── Depth enforcement ───────────────────────────────────────────────

def test_a_sub_agent_is_not_offered_the_delegate_tool_at_the_limit(scripted):
    """The primary enforcement: at depth 1 with a limit of 1, the tool is
    absent, so the model cannot spend an iteration asking."""
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "task"),
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ]
    catalog = _catalog(max_depth=1, auditor={
        "description": "Answers questions about the inventory.",
        "instructions": "You are an auditor.",
        "permissions": {"default": "allow"},
    })

    _runner(catalog=catalog).run(_main_spec(), "parent task")

    assert DELEGATE_TOOL_NAME in scripted.seen[0]["tools"]
    assert DELEGATE_TOOL_NAME not in scripted.seen[1]["tools"]


def test_a_depth_limit_of_zero_offers_delegation_to_nobody(scripted):
    """A way to switch delegation off without deleting the definitions."""
    scripted.scripts["big-model"] = [
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ]

    _runner(catalog=_catalog(max_depth=0)).run(_main_spec(), "task")

    assert DELEGATE_TOOL_NAME not in scripted.seen[0]["tools"]


def test_a_deeper_limit_allows_one_more_level(scripted):
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "task"),
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ]
    catalog = _catalog(max_depth=2, auditor={
        "description": "Answers questions about the inventory.",
        "instructions": "You are an auditor.",
        "permissions": {"default": "allow"},
    })

    _runner(catalog=catalog).run(_main_spec(), "parent task")

    assert DELEGATE_TOOL_NAME in scripted.seen[1]["tools"]


def test_the_handler_refuses_past_the_limit_even_if_it_is_reached():
    """Belt and braces: the tool is not offered past the limit, but a
    caller that built a delegation anyway is still refused."""
    catalog = _catalog(max_depth=1)
    view = RegistryView(
        _registry(), PermissionPolicy(default="allow"),
        delegation=Delegation(
            runner=MagicMock(), catalog=catalog,
            parent=ParentRun("p1", PermissionPolicy(default="allow"), depth=1),
        ),
    )

    result = view.call(DELEGATE_TOOL_NAME, {"role": "auditor", "task": "x"})

    assert "not available at this depth" in result


# ── The delegate tool is subject to the policy like any other ────────

def test_a_policy_can_deny_delegation():
    view = RegistryView(
        _registry(), PermissionPolicy(rules={DELEGATE_TOOL_NAME: "deny"}, default="allow"),
        delegation=Delegation(
            runner=MagicMock(), catalog=_catalog(),
            parent=ParentRun("p1", PermissionPolicy(default="allow"), 0),
        ),
    )

    assert DELEGATE_TOOL_NAME not in _names(view.schemas())
    assert view.call(DELEGATE_TOOL_NAME, {}).startswith("Permission denied:")


# ── The escalation check runs on every delegation ────────────────────

def test_an_escalating_sub_agent_cannot_be_delegated_to(scripted):
    """The construction-time layer, now with a real call site. The
    sub-agent allows a tool the parent denies, so the sub-run is refused
    before it starts."""
    catalog = _catalog(max_depth=1, sneaky={
        "description": "Claims to be harmless but wants a denied tool.",
        "instructions": "You are sneaky.",
        "permissions": {"default": "deny", "tools": {"forget_fact": "allow"}},
    })
    parent_policy = PermissionPolicy(rules={"forget_fact": "deny"}, default="allow")
    scripted.scripts["big-model"] = [
        _delegate_call("sneaky", "task"),
        {"type": "final_answer", "content": "I could not delegate that.",
         "model": "big-model"},
    ]

    result = _runner(catalog=catalog).run(_main_spec(policy=parent_policy), "parent task")

    # Reported to the parent's model as a refusal rather than crashing the
    # run: a refused action is a tool result here, as everywhere else.
    tool_results = [m for m in scripted.seen[-1]["messages"] if m["role"] == "tool"]
    assert tool_results[0]["content"].startswith("Permission denied:")
    assert "forget_fact" in tool_results[0]["content"]
    assert "Do not retry" in tool_results[0]["content"]
    assert result == "I could not delegate that."


def test_the_escalating_sub_agent_never_starts(scripted):
    """Refused before it runs, so it makes no LLM call of its own."""
    catalog = _catalog(max_depth=1, sneaky={
        "description": "Claims to be harmless but wants a denied tool.",
        "instructions": "You are sneaky.",
        "permissions": {"default": "deny", "tools": {"forget_fact": "allow"}},
    })
    scripted.scripts["big-model"] = [
        _delegate_call("sneaky", "task"),
        {"type": "final_answer", "content": "could not", "model": "big-model"},
    ]

    _runner(catalog=catalog).run(
        _main_spec(policy=PermissionPolicy(rules={"forget_fact": "deny"}, default="allow")),
        "parent task",
    )

    # Two calls only: the parent's delegate request and its final answer.
    # (The role name does appear in the parent's own tool call and in the
    # refusal, which is expected -- what must not exist is a sub-run.)
    assert len(scripted.seen) == 2
    system_prompts = [c["messages"][0]["content"] for c in scripted.seen]
    assert all(p == "You are the main agent." for p in system_prompts), (
        "a sub-run context was built for a spec that should have been refused"
    )


# ── Budget: independent, not shared ──────────────────────────────────

def test_a_sub_agents_iterations_do_not_draw_on_the_parents(scripted):
    """The decision: independent budgets. 8.4 requires a sub-agent to
    have "its own execution limits", and a shared budget would make the
    configured cap mean "that many, or whatever is left" -- so the same
    delegation would succeed early in a turn and fail late.

    Total work stays bounded because each delegation costs the parent one
    iteration: at depth 1 the worst case is parent_max * (1 + sub_max).
    """
    catalog = _catalog(max_depth=1, auditor={
        "description": "Answers questions about the inventory.",
        "instructions": "You are an auditor.",
        "max_iterations": 3,
        "permissions": {"default": "allow"},
    })
    # The parent delegates twice, then answers: 3 parent iterations.
    scripted.scripts["big-model"] = [
        _delegate_call("auditor", "first", call_id="c1"),
        # sub-run 1 uses all 3 of its own iterations
        {"type": "tool_call", "id": "s1", "tool_name": "read", "arguments": {}, "model": "big-model"},
        {"type": "tool_call", "id": "s2", "tool_name": "read", "arguments": {}, "model": "big-model"},
        {"type": "tool_call", "id": "s3", "tool_name": "read", "arguments": {}, "model": "big-model"},
        _delegate_call("auditor", "second", call_id="c2"),
        # sub-run 2 gets a full 3 again, not the remainder of anything
        {"type": "tool_call", "id": "s4", "tool_name": "read", "arguments": {}, "model": "big-model"},
        {"type": "tool_call", "id": "s5", "tool_name": "read", "arguments": {}, "model": "big-model"},
        {"type": "tool_call", "id": "s6", "tool_name": "read", "arguments": {}, "model": "big-model"},
        {"type": "final_answer", "content": "both done", "model": "big-model"},
    ]

    result = _runner(catalog=catalog).run(_main_spec(max_iterations=5), "parent task")

    assert result == "both done"
    # 3 parent calls + 3 + 3 sub calls, so the second sub-run was not
    # starved by the first.
    assert len(scripted.seen) == 9
