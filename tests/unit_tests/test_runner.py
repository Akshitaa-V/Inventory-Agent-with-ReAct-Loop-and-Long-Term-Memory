"""
Unit tests for AgentRunner -- the one way an agent gets run.

The runner's job is per-run assembly: the context, the model, the tool
view, the limits, the identity. Everything process-wide is handed to it
once and shared, so several tests below assert sharing rather than
behaviour -- a runner that quietly built a registry or a client per run
would pass a behavioural test and still re-run MCP discovery on every
delegation.

The other theme is that there is now one code path. The end-of-session
summary used to be a second, hand-assembled LLM call, and being
assembled separately is exactly why it reached the metrics with no model
label and no run identity. Routing it through the runner fixes both
because react_step already records them -- which is what the last group
of tests pins, on a summary-shaped spec.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from agent.agents import AgentSpec
from agent.context import Context
from agent.delegation import ParentRun
from agent.hooks import HookManager
from agent.permissions import PermissionPolicy, PolicyEscalation, auto_deny
from agent.runner import AgentRunner


@dataclass
class _Config:
    """Stands in for agent.config.Config -- the runner only reads the LLM
    fields and copies the rest with dataclasses.replace."""
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


def _spec(**overrides):
    fields = {
        "role": "reader",
        "description": "Reads one thing and reports back.",
        "instructions": "You are a reader.",
        "max_iterations": 4,
        "policy": PermissionPolicy(rules={"read": "allow"}, default="deny"),
    }
    fields.update(overrides)
    return AgentSpec(**fields)


def _registry(names=("read", "modify")):
    registry = MagicMock()
    registry.schemas.return_value = [
        {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
        for n in names
    ]
    registry.call.return_value = "tool ran"
    registry.is_builtin.return_value = True
    return registry


def _runner(registry=None, memory=None, confirm=None, config=None):
    return AgentRunner(
        config or _Config(), registry or _registry(), memory=memory, confirm=confirm
    )


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """Replaces the real HTTP client with one that answers immediately,
    and records every request so the model and tools sent can be checked."""
    calls = []

    class _StubLLM:
        def __init__(self, config):
            self.config = config

        def get_next_step(self, messages, tools=None):
            calls.append({
                "model": self.config.model,
                "temperature": self.config.temperature,
                "tools": tools,
                "messages": messages,
            })
            return {
                "type": "final_answer",
                "content": "answered",
                "model": self.config.model,
                "input_tokens": 11,
                "output_tokens": 7,
            }

    monkeypatch.setattr("agent.runner.LLMClient", _StubLLM)
    return calls


# ── A run returns a RunResult ────────────────────────────────────────

def test_a_run_returns_a_result_with_the_answer_and_success():
    result = _runner().run(_spec(), "do the thing")

    assert result == "answered"
    assert result.success is True
    assert result.reason == "final_answer"


def test_the_spec_iteration_cap_is_what_bounds_the_run(monkeypatch):
    """Its own execution limit, independent of the process config's. The
    model is driven to never answer, so the cap is the only thing that
    can end the run -- and the number of LLM calls shows whose cap it
    used."""
    registry = _registry()
    attempts = []

    class _NeverAnswers:
        def __init__(self, config):
            self.config = config

        def get_next_step(self, messages, tools=None):
            attempts.append(1)
            return {"type": "tool_call", "id": "c1", "tool_name": "read",
                    "arguments": {}, "model": "big-model"}

    monkeypatch.setattr("agent.runner.LLMClient", _NeverAnswers)

    # The process config allows 10; the spec allows 3.
    result = _runner(registry, config=_Config(max_iterations=10)).run(
        _spec(max_iterations=3, policy=PermissionPolicy(default="allow")), "loop"
    )

    assert len(attempts) == 3
    assert result.success is False
    assert result.reason == "iteration_limit"


# ── Context ──────────────────────────────────────────────────────────

def test_a_fresh_context_is_built_from_the_specs_instructions(stub_llm):
    """A delegated run gets its own context, which is what keeps a
    sub-agent's working history out of its parent's."""
    _runner().run(_spec(instructions="You are a specialist."), "task text")

    messages = stub_llm[0]["messages"]
    assert messages[0] == {"role": "system", "content": "You are a specialist."}
    assert messages[1] == {"role": "user", "content": "task text"}


def test_an_existing_context_is_continued_rather_than_replaced(stub_llm):
    """The CLI hands in its own so a session's turns accumulate."""
    context = Context("original system prompt")
    context.add_user_message("first turn")
    context.add_assistant_message("first answer")

    runner = _runner()
    runner.run(_spec(), "second turn", context=context)

    messages = stub_llm[0]["messages"]
    assert messages[0]["content"] == "original system prompt"
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "first turn", "second turn",
    ]


def test_two_runs_with_no_context_do_not_share_one(stub_llm):
    runner = _runner()
    runner.run(_spec(), "first task")
    runner.run(_spec(), "second task")

    first_users = [m["content"] for m in stub_llm[0]["messages"] if m["role"] == "user"]
    second_users = [m["content"] for m in stub_llm[1]["messages"] if m["role"] == "user"]
    assert first_users == ["first task"]
    assert second_users == ["second task"]


# ── include_memory ───────────────────────────────────────────────────

def test_memory_is_not_injected_unless_the_spec_asks(stub_llm):
    memory = MagicMock()
    memory.recall_all.return_value = "FACT: user prefers metric units"
    memory.recall_all_summaries.return_value = "SUMMARY: earlier session"

    _runner(memory=memory).run(_spec(include_memory=False), "task")

    assert "metric units" not in stub_llm[0]["messages"][0]["content"]
    memory.recall_all.assert_not_called()


def test_memory_is_injected_when_the_spec_asks(stub_llm):
    memory = MagicMock()
    memory.recall_all.return_value = "FACT: user prefers metric units"
    memory.recall_all_summaries.return_value = "SUMMARY: earlier session"

    _runner(memory=memory).run(_spec(include_memory=True), "task")

    prompt = stub_llm[0]["messages"][0]["content"]
    assert "You are a reader." in prompt
    assert "metric units" in prompt
    assert "earlier session" in prompt


def test_asking_for_memory_with_none_available_is_harmless(stub_llm):
    _runner(memory=None).run(_spec(include_memory=True), "task")

    assert stub_llm[0]["messages"][0]["content"] == "You are a reader."


def test_empty_memory_sections_are_not_appended(stub_llm):
    memory = MagicMock()
    memory.recall_all.return_value = ""
    memory.recall_all_summaries.return_value = ""

    _runner(memory=memory).run(_spec(include_memory=True), "task")

    assert stub_llm[0]["messages"][0]["content"] == "You are a reader."


# ── Model and temperature ────────────────────────────────────────────

def test_a_spec_without_overrides_uses_the_process_model(stub_llm):
    _runner().run(_spec(), "task")

    assert stub_llm[0]["model"] == "big-model"
    assert stub_llm[0]["temperature"] == 0.7


def test_a_spec_can_run_on_a_different_model(stub_llm):
    """So a narrow task can use a cheaper model without changing
    anything else about the harness."""
    _runner().run(_spec(model="small-model"), "task")

    assert stub_llm[0]["model"] == "small-model"
    assert stub_llm[0]["temperature"] == 0.7


def test_a_spec_can_override_only_the_temperature(stub_llm):
    _runner().run(_spec(temperature=0.0), "task")

    assert stub_llm[0]["model"] == "big-model"
    assert stub_llm[0]["temperature"] == 0.0


def test_an_override_does_not_leak_into_the_next_run(stub_llm):
    """The process config is copied, never mutated."""
    runner = _runner()
    runner.run(_spec(model="small-model"), "first")
    runner.run(_spec(), "second")

    assert stub_llm[0]["model"] == "small-model"
    assert stub_llm[1]["model"] == "big-model"


def test_the_shared_client_is_reused_when_nothing_is_overridden():
    """Constructed once at runner construction, not per run."""
    runner = _runner()
    first = runner._llm_for(_spec())
    second = runner._llm_for(_spec())

    assert first is second


# ── Tools: a view, never a new registry ──────────────────────────────

def test_the_run_sees_only_what_its_policy_allows(stub_llm):
    """The spec's allowlist policy filters the schemas the model is
    offered, via the view."""
    _runner(_registry(("read", "modify"))).run(_spec(), "task")

    offered = [t["function"]["name"] for t in stub_llm[0]["tools"]]
    assert offered == ["read"]


def test_the_shared_registry_is_not_rebuilt_per_run():
    """The reason the view exists. A registry per run would re-run MCP
    discovery and duplicate the receipt MemoryStore, whose writes are
    whole-file and would be lost."""
    registry = _registry()
    runner = _runner(registry)

    runner.run(_spec(policy=PermissionPolicy(default="allow")), "one")
    runner.run(_spec(policy=PermissionPolicy(default="allow")), "two")

    # schemas() is read from the same instance every time; nothing else
    # ever constructs a registry.
    assert registry.schemas.call_count >= 2


def test_a_denied_tool_is_refused_during_the_run(stub_llm, monkeypatch):
    """End to end through the view: the loop asks for a tool the spec
    forbids, and the shared registry is never reached."""
    registry = _registry()
    calls = iter([
        {"type": "tool_call", "id": "c1", "tool_name": "modify", "arguments": {}},
        {"type": "final_answer", "content": "could not", "model": "big-model"},
    ])

    class _ToolThenAnswer:
        def __init__(self, config):
            self.config = config

        def get_next_step(self, messages, tools=None):
            return next(calls)

    monkeypatch.setattr("agent.runner.LLMClient", _ToolThenAnswer)

    result = _runner(registry).run(_spec(), "task")

    assert result == "could not"
    registry.call.assert_not_called()


# ── Confirmation channel ─────────────────────────────────────────────

def test_the_runners_confirmation_channel_is_used(monkeypatch):
    """Shared by every run, so a sub-agent asks the same way its parent
    does -- "the same harness" applies to prompting too."""
    asked = []
    registry = _registry()
    calls = iter([
        {"type": "tool_call", "id": "c1", "tool_name": "read", "arguments": {}},
        {"type": "final_answer", "content": "done", "model": "big-model"},
    ])

    class _ToolThenAnswer:
        def __init__(self, config):
            self.config = config

        def get_next_step(self, messages, tools=None):
            return next(calls)

    monkeypatch.setattr("agent.runner.LLMClient", _ToolThenAnswer)

    def confirm(tool_name, arguments):
        asked.append(tool_name)
        return True

    runner = _runner(registry, confirm=confirm)
    runner.run(_spec(policy=PermissionPolicy(
        rules={"read": "require-user-confirmation"}, default="deny")), "task")

    assert asked == ["read"]
    registry.call.assert_called_once()


def test_a_runner_with_no_confirmation_channel_denies():
    """Fail closed: a runner built without a way to ask must not proceed
    as though it had been answered yes."""
    runner = _runner()

    assert runner._confirm is auto_deny


# ── Run identity ─────────────────────────────────────────────────────

@pytest.fixture
def run_events(monkeypatch):
    manager = HookManager()
    captured = []
    for name in ("run_start", "run_end", "llm_call_end"):
        manager.on(name, lambda event_name, **p: captured.append((event_name, p)))
    monkeypatch.setattr("agent.loop.hooks", manager)
    return captured


def _payloads(events, name):
    return [p for event_name, p in events if event_name == name]


def test_run_identity_is_threaded_through(run_events):
    parent = ParentRun(run_id="parent-1", policy=PermissionPolicy(default="allow"), depth=0)

    _runner().run(_spec(), "task", run_id="child-1", parent=parent)

    start = _payloads(run_events, "run_start")[0]
    assert start["run_id"] == "child-1"
    assert start["parent_run_id"] == "parent-1"
    assert start["kind"] == "sub"
    assert start["role"] == "reader"


def test_a_run_id_is_generated_when_none_is_given(run_events):
    _runner().run(_spec(), "task")

    assert _payloads(run_events, "run_start")[0]["run_id"]


def test_the_role_is_recorded_on_the_run(run_events):
    """8.4 asks for sub-agent executions to be recorded as part of the
    parent run; the role is what makes one attributable to a particular
    specialization rather than just "some sub-run"."""
    _runner().run(_spec(role="qr-labeller"), "task")

    assert _payloads(run_events, "run_start")[0]["role"] == "qr-labeller"
    assert _payloads(run_events, "run_end")[0]["role"] == "qr-labeller"


def test_a_run_with_no_parent_is_a_root_run(run_events):
    _runner().run(_spec(), "task")

    assert _payloads(run_events, "run_start")[0]["kind"] == "root"


# ── The escalation check at construction ─────────────────────────────

def test_a_spec_escalating_beyond_its_parent_is_refused():
    """The layer that cannot be skipped by a caller who built a spec in
    code instead of reading it from config."""
    parent = ParentRun(
        run_id="p1",
        policy=PermissionPolicy(rules={"modify": "deny"}, default="allow"),
        depth=0,
    )
    escalating = _spec(policy=PermissionPolicy(rules={"modify": "allow"}, default="deny"))

    with pytest.raises(PolicyEscalation, match="modify"):
        _runner().run(escalating, "task", parent=parent)


def test_a_narrowing_spec_passes_the_check(stub_llm):
    parent = ParentRun(
        run_id="p1",
        policy=PermissionPolicy(rules={"modify": "deny"}, default="allow"),
        depth=0,
    )

    result = _runner().run(_spec(), "task", parent=parent)

    assert result.success is True


def test_the_check_runs_before_anything_else(stub_llm):
    """No LLM call, no context, nothing -- an escalating spec must not
    get as far as doing work."""
    parent = ParentRun(run_id="p1", policy=PermissionPolicy(default="deny"), depth=0)
    escalating = _spec(policy=PermissionPolicy(default="allow"))

    with pytest.raises(PolicyEscalation):
        _runner().run(escalating, "task", parent=parent)

    assert stub_llm == []


def test_a_parent_identity_cannot_be_supplied_without_its_policy():
    """Structural rather than validated: ParentRun has no defaults, so a
    caller cannot pass the identity and forget the policy, which would
    silently skip the escalation check."""
    with pytest.raises(TypeError):
        ParentRun(run_id="p1", depth=0)


# ── One code path: a summary-shaped spec ─────────────────────────────

def _summary_spec():
    """Shaped exactly like main.SUMMARY_SPEC: no tools, one iteration."""
    return AgentSpec(
        role="session-summariser",
        description="Summarizes a finished session.",
        instructions="You summarize sessions. Reply with the summary only.",
        max_iterations=1,
        policy=PermissionPolicy(default="deny"),
    )


def test_a_tool_less_spec_sends_no_tools_at_all(stub_llm):
    """The deny-everything policy makes the view expose no schemas, which
    makes llm_client omit "tools" from the request -- the same single-shot
    request the summary used to assemble by hand."""
    _runner().run(_summary_spec(), "conversation text")

    assert stub_llm[0]["tools"] == []


def test_a_tool_less_run_still_reports_the_model(run_events):
    """One of the two gaps that existed while the summary was a separate
    code path: its LLM call reached the metrics with model="unknown"."""
    _runner().run(_summary_spec(), "conversation text")

    payload = _payloads(run_events, "llm_call_end")[0]
    assert payload["model"] == "big-model"


def test_a_tool_less_run_still_reports_a_run_id(run_events):
    """The other gap: the summary's call belonged to no run at all, so it
    could not be tied to anything in the structured data."""
    _runner().run(_summary_spec(), "conversation text", run_id="summary-run")

    assert _payloads(run_events, "llm_call_end")[0]["run_id"] == "summary-run"


def test_a_tool_less_run_still_reports_its_tokens(run_events):
    _runner().run(_summary_spec(), "conversation text")

    payload = _payloads(run_events, "llm_call_end")[0]
    assert payload["input_tokens"] == 11
    assert payload["output_tokens"] == 7


def test_a_tool_less_run_produces_an_ordinary_result():
    result = _runner().run(_summary_spec(), "conversation text")

    assert result == "answered"
    assert result.success is True
