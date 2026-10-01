"""
agent/agents.py -- sub-agent definitions from configuration.

Week 3 (handout 8.4) requires specialized sub-agents, each "a separately
agent instance with its own role or instructions, context, execution
limits, and permissions". This module is where those definitions are read
and validated. It is only the *definitions*: nothing here runs an agent,
builds a context, or talks to an LLM.

It deliberately mirrors agent/permissions.py. Config holds the "agents"
section as a plain dict and hands it here to be interpreted, exactly as
it hands the "permissions" section to PermissionPolicy.from_config --
Config stays a dumb data holder, and the component that cares about a
section owns its meaning and its validation. Misconfiguration raises at
construction time rather than surfacing mid-run, for the same reason: a
half-understood agent definition that quietly falls back to a default is
worse than one that stops the process with a readable message.

The per-role fields are the things a sub-agent must be able to differ on
that the harness cannot already vary. Its context is created per run and
its run identity is generated per run, so neither appears here.
"""

import re
from dataclasses import dataclass

from agent.permissions import PermissionPolicy, PolicyEscalation

# Applied to a role that does not set its own. Lower than the top-level
# max_iterations a whole session gets, because a delegated task is meant
# to be narrow -- a sub-agent that needs many steps is a sign the task
# should not have been delegated whole.
DEFAULT_MAX_ITERATIONS = 8

# How many levels of delegation are allowed. 1 means the main agent may
# delegate, but a sub-agent may not delegate further. Without a limit,
# handing a sub-agent the delegation tool makes unbounded recursion
# reachable from a single user request.
DEFAULT_MAX_DELEGATION_DEPTH = 1

_ALLOWED_SECTION_KEYS = {"max_delegation_depth", "max_iterations", "roles"}
_ALLOWED_ROLE_KEYS = {
    "description", "instructions", "max_iterations",
    "permissions", "model", "temperature", "include_memory",
}
_REQUIRED_ROLE_KEYS = {"description", "instructions"}

# A role name ends up as a Prometheus label value (see
# metrics.record_subagent_run) and as a field in log lines, so it is kept
# to characters that are safe and readable in both. This rejects names
# with spaces, quotes or newlines, which would be legal JSON keys but
# would make a metric label or a log line hard to read or to query.
_ROLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class AgentSpec:
    """One sub-agent's definition. Immutable once validated.

    role:         how the agent is named in config, in the delegation
                  tool, and in the observability data.
    description:  what this agent is for, in words aimed at the *parent*
                  agent -- it is what the parent reads when deciding
                  whether to delegate, so it describes the capability
                  rather than the implementation.
    instructions: the system prompt this agent runs under.
    max_iterations: its own execution limit, independent of its parent's.
    policy:       its own permissions, as a validated PermissionPolicy.
    model:        an LLM to use instead of the process default, or None
                  to inherit it. Lets a narrow task run on a cheaper
                  model without changing anything else.
    temperature:  likewise, or None to inherit.
    include_memory: whether the long-term fact store and session
                  summaries are prepended to its prompt. Defaults to
                  False: a specialist works from the brief its parent
                  hands it, and injecting the whole fact store would
                  both dilute that and cost tokens on every delegation.
    """

    role: str
    description: str
    instructions: str
    max_iterations: int
    policy: PermissionPolicy
    model: str | None = None
    temperature: float | None = None
    include_memory: bool = False


class AgentCatalog:
    """The set of sub-agents available to delegate to, plus the depth
    limit that applies across all of them.

    An empty catalog is valid and means delegation is not configured --
    which is exactly the behaviour of a config file written before
    sub-agents existed.
    """

    def __init__(self, specs=(), max_delegation_depth: int = DEFAULT_MAX_DELEGATION_DEPTH):
        self._specs = {}
        for spec in specs:
            if spec.role in self._specs:
                raise ValueError(f"duplicate agent role {spec.role!r}")
            self._specs[spec.role] = spec
        self._max_delegation_depth = _positive_int(
            max_delegation_depth, "max_delegation_depth", allow_zero=True
        )

    @classmethod
    def from_config(cls, raw: dict | None) -> "AgentCatalog":
        """Builds a catalog from the config's "agents" section.

        A missing or empty section yields an empty catalog, so a config
        predating this feature keeps working untouched.
        """
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"'agents' must be an object, got {type(raw).__name__}"
            )

        unknown = set(raw) - _ALLOWED_SECTION_KEYS
        if unknown:
            expected = ", ".join(sorted(_ALLOWED_SECTION_KEYS))
            raise ValueError(
                f"unknown key(s) in 'agents': {', '.join(sorted(unknown))} "
                f"(expected: {expected})"
            )

        roles = raw.get("roles") or {}
        if not isinstance(roles, dict):
            raise ValueError(
                f"'agents.roles' must be an object mapping role names to "
                f"definitions, got {type(roles).__name__}"
            )

        # A section-level max_iterations is the default every role
        # inherits unless it sets its own, so a config with several
        # similar agents does not have to repeat it.
        section_default_iterations = _positive_int(
            raw.get("max_iterations", DEFAULT_MAX_ITERATIONS),
            "agents.max_iterations",
        )

        specs = []
        seen = set()
        for raw_name, definition in roles.items():
            role = _role_name(raw_name)
            if role in seen:
                raise ValueError(f"duplicate agent role {role!r}")
            seen.add(role)
            specs.append(_spec_from_config(role, definition, section_default_iterations))

        return cls(
            specs=specs,
            max_delegation_depth=raw.get(
                "max_delegation_depth", DEFAULT_MAX_DELEGATION_DEPTH
            ),
        )

    @property
    def max_delegation_depth(self) -> int:
        return self._max_delegation_depth

    def roles(self) -> list[str]:
        """Configured role names, in a stable order."""
        return sorted(self._specs)

    def get(self, role: str) -> AgentSpec:
        """The spec for `role`, or KeyError naming what is available.

        Raises rather than returning None because there is no sensible
        default agent to fall back to. A runtime caller handed a role name
        by the model should check membership first (`role in catalog`) and
        report an unknown one back to the model as a tool result, the same
        way the registry reports an unknown tool.
        """
        try:
            return self._specs[role.strip() if isinstance(role, str) else role]
        except (KeyError, AttributeError):
            available = ", ".join(self.roles()) or "none configured"
            raise KeyError(
                f"no agent role {role!r}; available: {available}"
            ) from None

    def __contains__(self, role) -> bool:
        return isinstance(role, str) and role.strip() in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def require_narrower_than(self, parent_policy: PermissionPolicy) -> None:
        """Raise unless every configured role narrows `parent_policy`.

        Called once at startup with the top-level policy. Every role is
        checked before the first raises, so one message lists everything
        that needs fixing rather than making the reader rerun to find the
        next problem.
        """
        failures = []
        for role in self.roles():
            try:
                require_narrower(self._specs[role], parent_policy)
            except PolicyEscalation as exc:
                failures.append(str(exc))
        if failures:
            raise PolicyEscalation("\n".join(failures))

    def summary(self) -> str:
        """A short, human-readable description, in the same spirit as
        PermissionPolicy.summary() -- for printing at startup so it is
        visible which sub-agents exist."""
        if not self._specs:
            return "Sub-agents: none configured."
        listed = "; ".join(
            f"{role} ({self._specs[role].max_iterations} iter)" for role in self.roles()
        )
        return (
            f"Sub-agents: {len(self._specs)} configured, max delegation depth "
            f"{self._max_delegation_depth} -- {listed}."
        )


def require_narrower(spec: AgentSpec, parent_policy: PermissionPolicy) -> None:
    """Raise unless `spec` grants no more than `parent_policy`.

    This is the single place the rule is enforced, called from both
    layers, so the two cannot drift apart -- the same arrangement as
    permissions.refusal_text serving the loop and the registry.

      - At startup, AgentCatalog.require_narrower_than() calls it for
        every configured role, so an escalating definition stops the
        process with a readable message instead of surfacing at the
        moment someone delegates.
      - At sub-run construction, the code building the sub-agent calls it
        with the *actual* parent's policy. That is the layer that cannot
        be skipped, and it is the only complete one: with delegation
        deeper than one level, two roles can each narrow the top-level
        policy while one is still wider than the other, so only the real
        parent is authoritative.

    The reasoning is the same as for enforcing permissions in both the
    loop and the registry. If a sub-agent could hold permissions its
    parent lacks, delegating would be a way to do what the parent was
    refused -- the policy would bound the main agent and nothing else.
    """
    violations = spec.policy.narrowing_violations(parent_policy)
    if violations:
        raise PolicyEscalation(
            f"agent {spec.role!r} would grant more than its parent: "
            + "; ".join(violations)
            + ". A sub-agent's permissions may only narrow its parent's, "
            "never widen them."
        )


# ── Validation helpers ───────────────────────────────────────────────

def _role_name(raw_name) -> str:
    if not isinstance(raw_name, str):
        raise ValueError(
            f"agent role name must be a string, got "
            f"{type(raw_name).__name__} ({raw_name!r})"
        )
    role = raw_name.strip()
    if not role:
        raise ValueError("agent role with an empty name")
    if not _ROLE_NAME_RE.match(role):
        raise ValueError(
            f"invalid agent role name {raw_name!r}; use letters, digits, "
            f"'_', '-' or '.', starting with a letter or digit (it becomes "
            f"a metrics label and a log field)"
        )
    return role


def _positive_int(value, label: str, allow_zero: bool = False) -> int:
    # bool is a subclass of int, so True would otherwise pass as 1 and
    # silently become an iteration limit.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{label} must be an integer, got {type(value).__name__} ({value!r})"
        )
    if value < 0 or (value == 0 and not allow_zero):
        floor = "0 or greater" if allow_zero else "greater than 0"
        raise ValueError(f"{label} must be {floor}, got {value}")
    return value


def _required_text(definition: dict, key: str, role: str) -> str:
    value = definition.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"agent {role!r}: '{key}' must be a string, got "
            f"{type(value).__name__}"
        )
    if not value.strip():
        raise ValueError(f"agent {role!r}: '{key}' must not be empty")
    return value


def _spec_from_config(role: str, definition, section_default_iterations: int) -> AgentSpec:
    if not isinstance(definition, dict):
        raise ValueError(
            f"agent {role!r}: definition must be an object, got "
            f"{type(definition).__name__}"
        )

    unknown = set(definition) - _ALLOWED_ROLE_KEYS
    if unknown:
        expected = ", ".join(sorted(_ALLOWED_ROLE_KEYS))
        raise ValueError(
            f"agent {role!r}: unknown key(s) {', '.join(sorted(unknown))} "
            f"(expected: {expected})"
        )

    missing = _REQUIRED_ROLE_KEYS - set(definition)
    if missing:
        raise ValueError(
            f"agent {role!r}: missing required key(s) "
            f"{', '.join(sorted(missing))}"
        )

    # Nested, and delegated to the component that owns permission config,
    # so a sub-agent's rules are validated by exactly the same code and
    # error messages as the top-level ones.
    try:
        policy = PermissionPolicy.from_config(definition.get("permissions"))
    except ValueError as exc:
        raise ValueError(f"agent {role!r}: {exc}") from None

    temperature = definition.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError(
                f"agent {role!r}: 'temperature' must be a number, got "
                f"{type(temperature).__name__}"
            )
        if not 0 <= temperature <= 2:
            raise ValueError(
                f"agent {role!r}: 'temperature' must be between 0 and 2, "
                f"got {temperature}"
            )
        temperature = float(temperature)

    model = definition.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(
            f"agent {role!r}: 'model' must be a string, got {type(model).__name__}"
        )

    include_memory = definition.get("include_memory", False)
    if not isinstance(include_memory, bool):
        raise ValueError(
            f"agent {role!r}: 'include_memory' must be true or false, got "
            f"{type(include_memory).__name__}"
        )

    return AgentSpec(
        role=role,
        description=_required_text(definition, "description", role),
        instructions=_required_text(definition, "instructions", role),
        max_iterations=_positive_int(
            definition.get("max_iterations", section_default_iterations),
            f"agent {role!r}: 'max_iterations'",
        ),
        policy=policy,
        model=model,
        temperature=temperature,
        include_memory=include_memory,
    )
