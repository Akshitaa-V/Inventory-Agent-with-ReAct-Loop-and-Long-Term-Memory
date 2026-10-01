"""
ReAct Loop.
"""

import uuid

from rich.console import Console

from agent.context import Context
from agent.llm_client import LLMClient
from agent.hooks import hooks, timed
from agent.permissions import Decision, PermissionPolicy, auto_deny, refusal_text

console = Console()
ITERATION_LIMIT_MESSAGE = "Reached the iteration limit without a final answer."

# Every line the loop prints uses one of these tags, left-padded to the same
# width, so the terminal reads as a structured event log rather than prose --
# a professor (or teammate) scanning the output can tell at a glance what
# kind of thing just happened without reading the whole sentence.
_TAG_WIDTH = 8


def _log(tag: str, style: str, message: str) -> None:
    console.print(f"[{style}]{tag:<{_TAG_WIDTH}}[/{style}] {message}")


class RunResult(str):
    """What a run produced, plus whether it actually succeeded.

    This is a `str` subclass, so it behaves exactly like the plain answer
    string react_step used to return -- printing, comparing, substring
    checks, and the markdown-table formatter all work unchanged. That is
    deliberate: roughly thirty call sites treat the return value as text,
    and none of them should have to change to learn one extra fact. The
    same trick is used by permissions.Decision for the same reason.

    `success` is the signal that did not exist before. Reaching the
    iteration cap returns a perfectly ordinary string, so a caller -- and
    the run metrics -- had no way to tell it apart from a real answer, and
    a run that never answered was counted as a success.

    `reason` is "final_answer" or "iteration_limit". A run that raised
    carries neither: the exception propagates, and the instrumentation
    records the error instead.

    Note that truthiness is still *string* truthiness, not success: an
    empty answer is falsy and a failed run with a message is truthy. Check
    `.success` explicitly. This is on purpose -- existing tests use
    `assert answer` to mean "the agent said something", and quietly
    redefining that would change what they verify.
    """

    def __new__(cls, answer: str, *, success: bool, reason: str, run_id: str = None):
        result = super().__new__(cls, answer)
        result.success = success
        result.reason = reason
        result.run_id = run_id
        return result

    def __repr__(self) -> str:
        return (
            f"RunResult({str.__repr__(self)}, success={self.success}, "
            f"reason={self.reason!r}, run_id={self.run_id!r})"
        )


def _resolve_permission(policy: PermissionPolicy, confirm, tool_name: str, arguments,
                        run_id: str = None):
    """Decides whether one tool call may proceed.

    Returns (allowed, refusal, confirmed):
      allowed   -- whether the call may proceed
      refusal   -- None when allowed, else the text for the tool result
      confirmed -- True only when a confirmation was actually obtained,
                   which is what the registry is told so that it does not
                   ask a second time

    A `deny` rule never reaches the confirmation callback, so an
    auto-approving caller -- which is what the end-to-end tests use --
    still cannot run a denied tool. Only require-user-confirmation
    consults the callback.
    """
    outcome = policy.decide(tool_name)

    confirmed = None
    prompt_error = None
    if outcome is Decision.REQUIRE_USER_CONFIRMATION:
        _log("CONFIRM", "yellow", f"{tool_name} -- waiting for your approval")
        try:
            confirmed = bool(confirm(tool_name, arguments))
        except Exception as exc:
            # Fail closed. A broken prompt channel must not kill the run,
            # but it must not be read as approval either -- the user was
            # never actually asked.
            confirmed = False
            prompt_error = f"{type(exc).__name__}: {exc}"

    allowed = outcome is Decision.ALLOW or confirmed is True

    # Reported for every check, allows included, so "permission decisions"
    # can show the allow/deny balance rather than only refusals. Nothing
    # listens yet; hooks.emit with no listeners is a no-op. `decision` is
    # the policy's own outcome and `allowed` is the effect after any
    # confirmation, so a listener can record either without this having to
    # collapse the two into one label.
    hooks.emit(
        "permission_decision",
        tool_name=tool_name,
        decision=outcome.value,
        source=policy.source_for(tool_name),
        allowed=allowed,
        run_id=run_id,
    )

    if allowed:
        _log("PERM", "green", f"{tool_name} -- allowed")
        return True, None, confirmed is True

    # Wording lives in permissions.refusal_text so this layer and the
    # registry's own refusals cannot drift apart.
    if outcome is Decision.DENY:
        _log("PERM", "red", f"{tool_name} -- denied by policy")
        return False, refusal_text(tool_name, "denied"), False
    if prompt_error is not None:
        _log("PERM", "red", f"{tool_name} -- confirmation prompt failed")
        return False, refusal_text(tool_name, "prompt_failed", prompt_error), False
    _log("PERM", "red", f"{tool_name} -- you declined")
    return False, refusal_text(tool_name, "declined"), False


def react_step(
    llm: LLMClient,
    context: Context,
    tool_registry,
    max_iterations: int,
    *,
    policy: PermissionPolicy | None = None,
    confirm=None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    role: str | None = None,
) -> str:
    """Runs the ReAct loop until the model gives a final answer or the
    iteration cap is reached.

    policy:  which tools may be called. Keyword-only, and defaults to an
             all-allow policy so that a caller which does not care about
             permissions behaves exactly as it did before they existed.
             Enforcement is opt-in by injection, not implicit.
    confirm: how to ask the user about a require-user-confirmation
             outcome, as confirm(tool_name, arguments) -> bool. Defaults
             to permissions.auto_deny: a caller with no way to ask a
             human must not proceed as though it had been answered yes.
    run_id:  identifies this run in the observability data, so one run's
             LLM calls, tool calls and permission decisions can be picked
             out of the stream. Generated here when not supplied; a
             caller passes one in when it needs to know the id up front
             (a parent delegating to a sub-agent, for instance).
    parent_run_id: the run this one was delegated from, if any. Its
             presence is what makes this a sub-run -- the "kind" label
             below is derived from it rather than passed separately, so
             the two can never disagree.
    role:    which configured agent this run is, recorded on the run
             events. 8.4 asks for sub-agent executions to be recorded as
             part of the parent run, and the role is what makes a
             sub-run attributable to a particular specialization rather
             than just "some sub-run".
    """
    if policy is None:
        policy = PermissionPolicy()
    if confirm is None:
        confirm = auto_deny
    if run_id is None:
        # Short enough to read in a log line, random enough not to collide
        # within a session. Never used as a Prometheus label -- see the
        # note in metrics.py about unbounded label cardinality.
        run_id = uuid.uuid4().hex[:12]

    # "sub" runs are nested inside their parent's timing, so their duration
    # is already counted in the parent's. Labelling them apart is what stops
    # the two being summed into a wall-clock total that exceeds real time.
    kind = "sub" if parent_run_id else "root"
    run_labels = {"run_id": run_id, "kind": kind}
    if parent_run_id:
        run_labels["parent_run_id"] = parent_run_id
    if role:
        run_labels["role"] = role

    # One line announcing that a sub-agent is now doing work. This is the
    # signal a root run's terminal was missing entirely -- without it, a
    # delegated call just looked like the parent stalling.
    if parent_run_id:
        _log("SUBAGENT", "magenta", f"{role or 'sub-agent'} started (delegated by {parent_run_id})")

    with timed(hooks, "run", **run_labels) as run_ctx:
        for _ in range(max_iterations):
            # Compact before LLM call if threshold reached
            if context.should_compact():
                context.compact()

            status_label = f"Thinking ({role})..." if parent_run_id else "Thinking..."
            with console.status(status_label):
                with timed(hooks, "llm_call", run_id=run_id) as llm_ctx:
                    decision = llm.get_next_step(context.as_list(), tools=tool_registry.schemas())
                    if decision.get("input_tokens") is not None:
                        llm_ctx["input_tokens"] = decision["input_tokens"]
                    if decision.get("output_tokens") is not None:
                        llm_ctx["output_tokens"] = decision["output_tokens"]
                    if decision.get("model") is not None:
                        llm_ctx["model"] = decision["model"]

            if decision["type"] == "final_answer":
                context.add_assistant_message(decision["content"])
                run_ctx["reason"] = "final_answer"
                if parent_run_id:
                    _log("SUBAGENT", "magenta", f"{role or 'sub-agent'} finished")
                return RunResult(
                    decision["content"], success=True,
                    reason="final_answer", run_id=run_id,
                )

            context.add_assistant_message(
                content="",
                tool_calls=[{
                    "id": decision["id"],
                    "type": "function",
                    "function": {
                        "name": decision["tool_name"],
                        "arguments": decision["arguments"],
                    },
                }],
            )

            # Enforcement sits here, immediately before the call: a refused
            # tool must not run, and must not produce a tool_call span
            # either -- it never happened, and counting it as a failed call
            # would conflate "the tool broke" with "the tool was blocked".
            # The `continue` below is what guarantees that structurally,
            # by never entering the timed() block at all.
            allowed, refusal, confirmed = _resolve_permission(
                policy, confirm, decision["tool_name"], decision["arguments"],
                run_id=run_id,
            )
            if not allowed:
                # Still answered, like every other outcome: a tool_calls
                # message without a matching tool result makes the next
                # request malformed. The refused call also still costs an
                # iteration, because it cost a real LLM round-trip.
                context.add_tool_result(decision["id"], refusal)
                continue

            # The registry enforces the policy again, independently. Telling
            # it a confirmation was already obtained is what stops it asking
            # a second time; it is passed only when one genuinely was, since
            # for an allow-outcome call no confirmation was involved at all.
            granted = {"confirmed": True} if confirmed else {}

            tool_type = "builtin" if tool_registry.is_builtin(decision["tool_name"]) else "mcp"

            # This is the professor-requested "tool feedback" line: the name
            # of the tool, and nothing else, the moment it is actually
            # invoked. Detail (arguments, timing, tokens) belongs in the
            # monitoring log, not the terminal -- a person watching just
            # needs to see that a call happened and how it turned out.
            _log("TOOL", "cyan", f"{decision['tool_name']} ({tool_type}) called")

            with timed(hooks, "tool_call", tool_name=decision["tool_name"],
                       tool_type=tool_type, run_id=run_id) as tool_ctx:
                try:
                    observation = tool_registry.call(
                        decision["tool_name"], decision["arguments"], **granted
                    )
                except Exception as exc:
                    # Surface the failure to the model as evidence instead of ending the
                    # session: a bad argument or a filesystem refusal is something it can
                    # reason about and retry. KeyboardInterrupt/SystemExit still propagate.
                    observation = (
                        f"Error: tool '{decision['tool_name']}' raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                    tool_ctx["success"] = False
                    tool_ctx["error"] = str(exc)
                    _log("TOOL", "red", f"{decision['tool_name']} failed ({type(exc).__name__})")
                else:
                    _log("TOOL", "green", f"{decision['tool_name']} done")

            # Outside the try: every tool_calls message must get an answer, or the
            # next request is malformed.
            context.add_tool_result(decision["id"], observation)

        # The cap was reached without an answer. The guard message still
        # goes into the context so the history never ends on a dangling
        # tool result -- but this is a failed run, and saying so is what
        # stops agent_runs_total counting it under "success". timed() only
        # detects failure from an exception, and nothing was raised here,
        # so the override has to be explicit.
        context.add_assistant_message(ITERATION_LIMIT_MESSAGE)
        run_ctx["success"] = False
        run_ctx["reason"] = "iteration_limit"
        if parent_run_id:
            _log("SUBAGENT", "magenta", f"{role or 'sub-agent'} stopped (iteration limit)")
        return RunResult(
            ITERATION_LIMIT_MESSAGE, success=False,
            reason="iteration_limit", run_id=run_id,
        )