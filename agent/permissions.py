"""
agent/permissions.py -- the permission policy model.

Week 3 (handout 8.4) requires permission policies for built-in and MCP
tools, with three outcomes: allow, deny, and require-user-confirmation.

This module is only the *model*. It answers one question -- "what should
happen if this tool is called?" -- and does nothing else: it never
intercepts a tool call, never prompts the user, never records a metric.
Resolving a require-user-confirmation outcome into an actual yes/no, and
enforcing a deny, belong to the wiring in loop.py and tool_registry.py.

Keeping those out of here is deliberate. A policy that is pure and
side-effect-free can be unit-tested without a terminal, an LLM, a tool
registry, or stdin, which is also why this file can land before any of
the wiring exists.

Built-in and MCP tools need no separate handling here. By the time a
tool is callable it has a single flat name in one namespace ("modify",
"ocr_extract_text", "generate_qr_code"), because ToolRegistry.schemas()
already concatenates the built-in schemas with the MCP-discovered ones
and the model can only ever name a tool from that combined list. So one
name -> outcome map covers both kinds, and there is no builtin/mcp
branch below. That distinction still matters for *reporting* -- see
ToolRegistry.is_builtin -- just not for resolving a decision.
"""

from enum import Enum


class Decision(str, Enum):
    """The three outcomes the handout requires.

    The values are deliberately the exact strings that metrics.py's
    record_permission_decision() expects for its `decision` label, so a
    caller can pass `decision.value` straight through with no mapping
    table that could drift out of sync. Subclassing `str` additionally
    means a Decision compares equal to its own spelling, which keeps
    log formatting and assertions simple.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_USER_CONFIRMATION = "require-user-confirmation"


# The outcome applied to any tool with no explicit rule, when the config
# does not say otherwise. `allow` is chosen so that a config.json written
# before this feature existed keeps behaving exactly as it did -- adding
# this module changes nothing until someone actually writes rules. The
# stricter policy for this project lives in the config file, not here, so
# that tightening it never means editing code.
DEFAULT_DECISION = Decision.ALLOW

# Keys accepted in the config's "permissions" section. Anything else is
# rejected rather than ignored: a "tool"/"tools" typo would otherwise
# make every rule silently vanish and leave the agent wide open while
# looking correctly configured.
_ALLOWED_CONFIG_KEYS = {"default", "tools"}


def parse_decision(value) -> Decision:
    """Turns a config value into a Decision, or raises ValueError.

    Hand-written config is allowed to spell an outcome with underscores
    ("require_user_confirmation"), in mixed case, or with stray
    surrounding whitespace -- all three are natural to type and none is
    ambiguous -- so those are normalized to the canonical hyphenated
    form on the way in. Only one spelling ever reaches the rest of the
    system.

    An unrecognized outcome is a hard error, never a fallback to the
    default. Silently treating "deney" as the default would read as
    `allow` at runtime: a misconfiguration that looks like a policy but
    enforces nothing. Failing at construction time instead means a bad
    config kills startup, in the same spirit as load_config() raising on
    a missing API token, rather than surfacing mid-run.
    """
    if isinstance(value, Decision):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"permission outcome must be a string, got {type(value).__name__} ({value!r})"
        )

    normalized = value.strip().lower().replace("_", "-")
    try:
        return Decision(normalized)
    except ValueError:
        valid = ", ".join(d.value for d in Decision)
        raise ValueError(
            f"unknown permission outcome {value!r}; expected one of: {valid}"
        ) from None


class PermissionPolicy:
    """Resolves a tool name to a Decision.

    Two tiers, most specific first:
      1. An explicit rule for that exact tool name.
      2. The policy's default, for every tool without one.

    A tool-type tier (one default for built-ins, another for MCP tools)
    was considered and left out: nothing in the handout needs it, and
    the config shape has room to add it later without breaking the
    rules already written.
    """

    def __init__(self, rules: dict | None = None, default=DEFAULT_DECISION):
        self._default = parse_decision(default)

        self._rules: dict[str, Decision] = {}
        for raw_name, outcome in (rules or {}).items():
            if not isinstance(raw_name, str):
                raise ValueError(
                    f"permission rule name must be a string, got "
                    f"{type(raw_name).__name__} ({raw_name!r})"
                )
            name = raw_name.strip()
            if not name:
                raise ValueError("permission rule with an empty tool name")
            # Two keys that differ only in surrounding whitespace collapse
            # to the same tool here, and one would silently overwrite the
            # other. Rejecting is better than picking a winner the author
            # of the config did not choose.
            if name in self._rules:
                raise ValueError(
                    f"duplicate permission rule for tool {name!r}"
                )
            try:
                self._rules[name] = parse_decision(outcome)
            except ValueError as exc:
                # Name the tool: "unknown permission outcome 'deney'" on its
                # own does not say which of a dozen rules to go and fix.
                raise ValueError(f"permission rule for {name!r}: {exc}") from None

    @classmethod
    def from_config(cls, raw: dict | None) -> "PermissionPolicy":
        """Builds a policy from the config's "permissions" section.

        Config holds this section as a plain dict and hands it here to be
        interpreted, which is the same split already used for
        "mcp_servers": Config stays a dumb data holder, and the component
        that cares about a section owns its meaning and its validation.

        A missing or empty section yields an all-allow policy -- i.e.
        exactly today's behaviour.
        """
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"'permissions' must be an object, got {type(raw).__name__}"
            )

        unknown = set(raw) - _ALLOWED_CONFIG_KEYS
        if unknown:
            expected = ", ".join(sorted(_ALLOWED_CONFIG_KEYS))
            raise ValueError(
                f"unknown key(s) in 'permissions': {', '.join(sorted(unknown))} "
                f"(expected: {expected})"
            )

        tools = raw.get("tools") or {}
        if not isinstance(tools, dict):
            raise ValueError(
                f"'permissions.tools' must be an object mapping tool names to "
                f"outcomes, got {type(tools).__name__}"
            )

        return cls(rules=tools, default=raw.get("default", DEFAULT_DECISION))

    @property
    def default(self) -> Decision:
        return self._default

    def decide(self, tool_name: str) -> Decision:
        """The outcome for `tool_name`.

        Tool names are matched exactly (after stripping whitespace), not
        case-insensitively: they are case-sensitive identifiers that the
        model emits verbatim from the tool schemas, so folding case would
        let a rule for "modify" appear to cover a "Modify" that is not a
        real tool at all -- a false sense of coverage.

        Never raises: an unknown or misspelled tool name simply gets the
        default. Rejecting unknown tools is the registry's job, and it
        already reports them as "unknown tool"; a policy that raised here
        would turn a model typo into a crashed run.
        """
        if not isinstance(tool_name, str):
            raise ValueError(
                f"tool name must be a string, got {type(tool_name).__name__}"
            )
        return self._rules.get(tool_name.strip(), self._default)

    def source_for(self, tool_name: str) -> str:
        """Where decide() got its answer: "rule" or "default".

        Not needed to enforce anything -- this exists for the structured
        logs the handout asks for, so an individual run can show *why* a
        tool was blocked, not just that it was.
        """
        if not isinstance(tool_name, str):
            raise ValueError(
                f"tool name must be a string, got {type(tool_name).__name__}"
            )
        return "rule" if tool_name.strip() in self._rules else "default"

    def rules(self) -> dict[str, Decision]:
        """A copy of the explicit rules, so callers cannot mutate the
        policy after it has been validated."""
        return dict(self._rules)

    def narrows(self, parent: "PermissionPolicy") -> bool:
        """True if this policy grants no more than `parent` anywhere.

        Equal is fine -- narrowing means "no wider", not "strictly
        narrower" -- so a sub-agent may restate its parent's rules.
        """
        return not _narrowing_violations(self, parent)

    def narrowing_violations(self, parent: "PermissionPolicy") -> list[str]:
        """Where this policy exceeds `parent`, as readable descriptions.

        Empty when it narrows. Returned rather than raised so a caller
        can put every violation in one message instead of reporting them
        one failed attempt at a time.
        """
        return _narrowing_violations(self, parent)

    def summary(self) -> str:
        """A short, human-readable description, in the same spirit as
        MemoryStore.summary() -- for printing at session startup once
        this is wired in, so it is visible which policy is in force."""
        if not self._rules:
            return f"Permissions: no explicit rules, default '{self._default.value}'."

        by_decision: dict[Decision, list[str]] = {}
        for name, decision in sorted(self._rules.items()):
            by_decision.setdefault(decision, []).append(name)

        parts = [
            f"{decision.value}: {', '.join(names)}"
            # Iterate over Decision rather than by_decision so the order is
            # always allow, deny, require-user-confirmation, never dict order.
            for decision in Decision
            if (names := by_decision.get(decision))
        ]
        return (
            f"Permissions: default '{self._default.value}'; "
            f"{len(self._rules)} rule(s) -- " + "; ".join(parts) + "."
        )


# ── Comparing two policies ───────────────────────────────────────────

# How permissive each outcome is, for deciding whether one policy is
# narrower than another. require-user-confirmation sits in the middle
# because it *can* be granted: strictly more permissive than deny, which
# can never be, and strictly less than allow, which needs no permission
# at all.
_PERMISSIVENESS = {
    Decision.DENY: 0,
    Decision.REQUIRE_USER_CONFIRMATION: 1,
    Decision.ALLOW: 2,
}


class PolicyEscalation(ValueError):
    """One policy grants more than another that it must not exceed.

    A ValueError subclass, so a caller that only cares that the config
    was rejected can keep catching ValueError, while a caller that wants
    to report an escalation specifically -- a delegation tool telling the
    model why it could not delegate -- can catch just this.
    """


def _narrowing_violations(child: "PermissionPolicy", parent: "PermissionPolicy") -> list[str]:
    """Tool names where `child` grants more than `parent`, described.

    Finitely checkable despite tool names being an open set. Two parts
    cover every possible name:

      - the defaults, which decide every tool neither policy mentions;
      - the union of both rule sets, which is the only place the two can
        differ for a named tool.

    Checking only the child's own rules would miss the case that matters
    most: a tool the parent gates and the child never names, reached
    through the child's more permissive default.
    """
    violations = []

    if _PERMISSIVENESS[child.default] > _PERMISSIVENESS[parent.default]:
        violations.append(
            f"default '{child.default.value}' exceeds parent default "
            f"'{parent.default.value}'"
        )

    for tool_name in sorted(set(child.rules()) | set(parent.rules())):
        child_outcome = child.decide(tool_name)
        parent_outcome = parent.decide(tool_name)
        if _PERMISSIVENESS[child_outcome] > _PERMISSIVENESS[parent_outcome]:
            violations.append(
                f"'{tool_name}': '{child_outcome.value}' exceeds parent "
                f"'{parent_outcome.value}'"
            )

    return violations


# ── Confirmation callbacks ───────────────────────────────────────────
#
# A require-user-confirmation outcome only says that someone has to be
# asked; it does not say how. The asking is injected into the loop as a
# callback with the signature:
#
#     confirm(tool_name: str, arguments) -> bool
#
# `arguments` arrives exactly as the loop holds it -- normally the raw
# JSON string the model emitted, not a parsed dict -- because parsing
# tool arguments is ToolRegistry.call()'s job and duplicating it here
# would give two places to keep in step. A callback that wants
# structured arguments can parse them itself.
#
# The two callbacks below are the non-interactive ones, and they live
# here because they are pure: no prompting, no I/O, nothing that would
# undercut this module being testable without a terminal. The real
# interactive prompt lives in agent/main.py, which already owns the REPL
# and the only input() in the codebase.


def auto_deny(tool_name: str, arguments=None) -> bool:
    """Refuse every confirmation request.

    This is what the loop falls back to when no callback is injected.
    Anything that cannot ask a human -- a test, a scheduled run, the
    dashboard, a sub-agent with no console -- must not quietly proceed
    as though permission had been granted, so the default answer to "may
    I?" with nobody to ask is no.
    """
    return False


def auto_approve(tool_name: str, arguments=None) -> bool:
    """Approve every confirmation request.

    Only for callers that have deliberately opted in, which in this
    project means the end-to-end tests: they drive the real policy so
    that a mistake in it surfaces, but they have no human to answer a
    prompt and must never block on stdin.

    This cannot loosen a `deny` rule. The loop only consults a callback
    for the require-user-confirmation outcome, so a denied tool stays
    denied even under auto_approve.
    """
    return True


# ── Refusal messages ─────────────────────────────────────────────────

# Every refusal, from either enforcement layer, is worded here so the two
# cannot drift apart. The text is fed back to the model as a tool result,
# so it has two jobs: say plainly that the call was blocked by policy
# rather than that the tool failed, and tell the model not to retry or to
# route around it. A model that reads a policy block as a transient error
# will retry it until the iteration cap and waste the whole turn.
#
# The "Permission denied:" prefix is deliberately distinct from the
# "Error: tool ... raised ..." string the loop uses for a tool that threw,
# which is the other thing the model sees in the same position.

_REFUSAL_REASONS = {
    # The policy says deny. Nothing the caller can supply changes this.
    "denied": (
        "the tool '{tool_name}' is blocked by this agent's permission "
        "policy and was not executed. Do not retry it, and do not try to "
        "achieve the same result with a different tool -- tell the user "
        "this action is not permitted."
    ),
    # A human was asked and said no.
    "declined": (
        "the user was asked to confirm running '{tool_name}' and declined, "
        "so it was not executed. Do not retry it -- tell the user it was "
        "not run and ask how they would like to proceed."
    ),
    # Confirmation was required and none was obtained -- nobody was asked,
    # which is different from being asked and refused. This is what the
    # registry returns when a caller reaches it without going through a
    # layer that can prompt.
    "unconfirmed": (
        "running '{tool_name}' requires the user's confirmation, and none "
        "was obtained, so it was not executed. Do not retry it -- tell the "
        "user this action needs their explicit approval."
    ),
    # The confirmation channel itself broke. The user was never asked, so
    # this fails closed like the others.
    "prompt_failed": (
        "running '{tool_name}' requires user confirmation, but the "
        "confirmation prompt itself failed ({detail}), so the tool was not "
        "executed. Do not retry it -- report the problem to the user."
    ),
}


def refusal_text(tool_name: str, reason: str, detail: str | None = None) -> str:
    """The tool-result text for a refused call.

    `reason` is one of: denied, declined, unconfirmed, prompt_failed.
    An unrecognized reason is a programming error rather than a
    configuration one, so it raises rather than producing a vague
    refusal that would be hard to trace back here.
    """
    try:
        template = _REFUSAL_REASONS[reason]
    except KeyError:
        valid = ", ".join(sorted(_REFUSAL_REASONS))
        raise ValueError(
            f"unknown refusal reason {reason!r}; expected one of: {valid}"
        ) from None
    return "Permission denied: " + template.format(tool_name=tool_name, detail=detail)
