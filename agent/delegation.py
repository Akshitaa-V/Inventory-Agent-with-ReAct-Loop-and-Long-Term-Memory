"""
agent/delegation.py -- the two small types delegation needs.

They live in their own module purely to keep imports acyclic:
agent/runner.py imports RegistryView from agent/tool_registry.py, so
tool_registry must not import runner, yet both need to agree on how a
parent run is described. This module depends on neither.
"""

from dataclasses import dataclass

from agent.permissions import PermissionPolicy


@dataclass(frozen=True)
class ParentRun:
    """The run a sub-run was delegated from.

    Every field is required and there are no defaults, which is the
    point. These three facts are only ever useful together:

      - run_id links the sub-run to its parent in the observability data,
        which 8.4 requires ("record properly sub-agent executions as
        part of the corresponding parent run");
      - policy is what the sub-agent's own policy is checked against, so
        a sub-agent cannot hold permissions its parent lacks;
      - depth is what bounds recursion.

    Bundling them means a caller cannot supply the identity and forget
    the policy, which would silently skip the escalation check -- the
    check that is the *only* complete one once delegation goes deeper
    than a single level, since only the real parent is authoritative.
    Passing three separate optional arguments made that omission
    possible; this makes it unrepresentable.
    """

    run_id: str
    policy: PermissionPolicy
    depth: int


@dataclass(frozen=True)
class Delegation:
    """What a RegistryView needs in order to offer delegation.

    Constructed only by AgentRunner, which is what guarantees `parent`
    describes the very run the view belongs to -- in particular that
    `parent.policy` is the same policy the view itself enforces.

    `runner` and `catalog` are held by duck-type rather than by import,
    so this module stays free of both.
    """

    runner: object
    catalog: object
    parent: ParentRun
