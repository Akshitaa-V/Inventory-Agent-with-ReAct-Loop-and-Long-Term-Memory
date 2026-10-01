"""
agent/runner.py -- runs one agent, described by an AgentSpec.

This is the single way an agent gets run in this project. Before it, the
harness had two: the ReAct loop for the main agent, and a hand-rolled
LLM call in main.py for the end-of-session summary, which had its own
prompt assembly, its own instrumentation and -- because it was assembled
separately -- its own gaps in that instrumentation.

Both now go through run() below, which matters beyond tidiness. The
handout (8.4) requires that "a sub-agent must run on the same harness
that its parent agent", and the most direct evidence for that is a
parent that is itself an AgentSpec handed to the same runner. It is also
why the summary's missing model label and missing run identity are fixed
here by *routing* rather than by patching a second code path: react_step
already records both, so anything that goes through it gets them.

What this module owns is per-run assembly: the context, the model, the
tool view, the limits, the identity. What it deliberately does not own
is anything process-wide -- the shared tool registry, the long-term
memory, the LLM credentials and the confirmation channel are handed to
it once at construction and shared by every run, because duplicating any
of them per run is either wasteful (MCP rediscovery) or unsafe (two
MemoryStore instances over one file).
"""

import uuid
from dataclasses import replace

from agent.agents import AgentSpec, require_narrower
from agent.context import Context
from agent.delegation import Delegation, ParentRun
from agent.llm_client import LLMClient
from agent.loop import RunResult, react_step
from agent.permissions import auto_deny
from agent.tool_registry import RegistryView


class AgentRunner:
    """Runs agents described by AgentSpec against shared resources.

    config:   the process LLM configuration. A spec that overrides the
              model or temperature gets a client built from a copy of
              this; one that does not reuses the shared client, so the
              common case allocates nothing.
    registry: the one shared ToolRegistry. Each run gets a RegistryView
              onto it carrying that run's policy -- never its own
              registry, which would re-run MCP discovery and duplicate
              the receipt MemoryStore.
    memory:   long-term memory, consulted only for a spec that asks for
              it via include_memory. May be None.
    confirm:  how to ask the user about a require-user-confirmation
              outcome. Shared by every run, so a sub-agent asks the same
              way its parent does. Defaults to auto_deny, so a runner
              built without a way to ask never proceeds as though it had
              been answered yes.
    catalog:  the sub-agents available to delegate to, and the depth
              limit. Omitted or empty means no run is offered the
              delegation tool at all.
    """

    def __init__(self, config, registry, memory=None, confirm=None, catalog=None):
        self._config = config
        self._registry = registry
        self._memory = memory
        self._confirm = confirm if confirm is not None else auto_deny
        self._catalog = catalog
        self._shared_llm = LLMClient(config)

    def run(
        self,
        spec: AgentSpec,
        task: str,
        *,
        context: Context = None,
        run_id: str = None,
        parent: ParentRun = None,
    ) -> RunResult:
        """Runs `spec` on `task` and returns its RunResult.

        context: an existing conversation to continue. The CLI passes its
            own so a session's turns accumulate in one history; a
            delegated run passes nothing and gets a fresh context built
            from the spec's instructions, which is what keeps a
            sub-agent's working context out of its parent's.
        parent: the run this was delegated from, or None for a top-level
            run. One argument rather than three, because a run identity,
            a policy to be checked against and a depth are only useful
            together -- and because a caller that could pass the identity
            while forgetting the policy would silently skip the
            escalation check. Depth is derived from it too, so nothing
            has to keep the two consistent by hand.
        """
        if parent is not None:
            # The layer that cannot be skipped by a caller who built a
            # spec in code rather than reading it from config, and the
            # only complete one past a single level of delegation, since
            # only the real parent is authoritative.
            require_narrower(spec, parent.policy)

        if run_id is None:
            run_id = uuid.uuid4().hex[:12]
        depth = parent.depth + 1 if parent is not None else 0

        if context is None:
            context = Context(self._instructions_for(spec))
        context.add_user_message(task)

        return react_step(
            self._llm_for(spec),
            context,
            RegistryView(
                self._registry, spec.policy,
                delegation=self._delegation_for(run_id, spec, depth),
            ),
            spec.max_iterations,
            policy=spec.policy,
            confirm=self._confirm,
            run_id=run_id,
            parent_run_id=parent.run_id if parent is not None else None,
            role=spec.role,
        )

    def _delegation_for(self, run_id: str, spec: AgentSpec, depth: int) -> Delegation:
        """What this run needs in order to delegate, or None if it may not.

        Returning None at the depth limit is the primary enforcement: the
        tool is then absent from the view's schemas, so the model is never
        offered something it would only be refused for. The handler checks
        the limit again anyway, as the layer a caller cannot skip.

        Note the parent this builds describes *this* run: what it
        delegates to becomes its child, and spec.policy -- the very
        policy the view enforces -- is what any child is checked against.
        """
        if not self._catalog or depth >= self._catalog.max_delegation_depth:
            return None
        return Delegation(
            runner=self,
            catalog=self._catalog,
            parent=ParentRun(run_id=run_id, policy=spec.policy, depth=depth),
        )

    def _instructions_for(self, spec: AgentSpec) -> str:
        """The spec's instructions, with long-term memory prepended only
        if it asked for it.

        A specialist normally works from the brief its parent hands it,
        so injecting the whole fact store would dilute that and cost
        tokens on every delegation -- hence include_memory defaulting to
        False on the spec rather than this being unconditional, which is
        what the CLI used to do for its one and only agent.
        """
        instructions = spec.instructions
        if not (spec.include_memory and self._memory):
            return instructions

        for section in (self._memory.recall_all(), self._memory.recall_all_summaries()):
            if section:
                instructions += "\n\n" + section
        return instructions

    def _llm_for(self, spec: AgentSpec) -> LLMClient:
        """A client honouring the spec's model and temperature.

        Returns the shared client untouched when the spec overrides
        neither, which is the usual case. An override copies the process
        config rather than mutating it, so one run's choice of a cheaper
        model cannot leak into the next.
        """
        if spec.model is None and spec.temperature is None:
            return self._shared_llm

        overrides = {}
        if spec.model is not None:
            overrides["model"] = spec.model
        if spec.temperature is not None:
            overrides["temperature"] = spec.temperature
        return LLMClient(replace(self._config, **overrides))
