"""
Unit tests for the rule that a sub-agent may only narrow its parent.

The reasoning is the same as for enforcing permissions in both the loop
and the registry: if a sub-agent could hold permissions its parent
lacks, delegating would be a way to do exactly what the parent was
refused, and the policy would bound the main agent and nothing else.

Two things are tested. First the comparison itself, which is pure and
lives in permissions.py -- including the case most easily got wrong, a
tool the parent gates that the child never names and reaches through a
wider default. Then the enforcement, which happens at two layers for
the same reason permissions do: at startup where a human can read and
fix it, and at sub-run construction where a caller cannot skip it.
"""

import json
import os

import pytest

from agent.agents import AgentCatalog, AgentSpec, require_narrower
from agent.permissions import Decision, PermissionPolicy, PolicyEscalation


def _policy(rules=None, default="allow"):
    return PermissionPolicy(rules=rules or {}, default=default)


def _spec(role="child", policy=None):
    return AgentSpec(
        role=role,
        description="Does one narrow thing.",
        instructions="You are a specialist.",
        max_iterations=4,
        policy=policy if policy is not None else _policy(default="deny"),
    )


# ── The ordering of the three outcomes ───────────────────────────────

def test_deny_is_narrower_than_confirmation_which_is_narrower_than_allow():
    """require-user-confirmation sits between the two because it can be
    granted: more permissive than deny, less than allow."""
    parent = _policy({"t": "allow"})

    assert _policy({"t": "deny"}).narrows(parent)
    assert _policy({"t": "require-user-confirmation"}).narrows(parent)
    assert _policy({"t": "allow"}).narrows(parent)

    gated = _policy({"t": "require-user-confirmation"})
    assert _policy({"t": "deny"}).narrows(gated)
    assert not _policy({"t": "allow"}).narrows(gated)

    denied = _policy({"t": "deny"})
    assert not _policy({"t": "require-user-confirmation"}).narrows(denied)
    assert not _policy({"t": "allow"}).narrows(denied)


def test_an_identical_policy_counts_as_narrowing():
    """Narrowing means "no wider", not "strictly narrower" -- a sub-agent
    restating its parent's rules is fine."""
    parent = _policy({"a": "deny", "b": "require-user-confirmation"})

    assert parent.narrows(parent)


# ── The case most easily got wrong ───────────────────────────────────

def test_a_wider_default_reaching_a_tool_the_parent_gates_is_caught():
    """The child never names ocr_extract_text, so checking only its own
    rules would pass it -- but its allow-by-default reaches the tool the
    parent gates."""
    parent = _policy({"ocr_extract_text": "require-user-confirmation"}, default="allow")
    child = _policy({"read": "allow"}, default="allow")

    violations = child.narrowing_violations(parent)

    assert any("ocr_extract_text" in v for v in violations)


def test_a_wider_default_is_reported_on_its_own():
    parent = _policy(default="deny")
    child = _policy(default="allow")

    violations = child.narrowing_violations(parent)

    assert len(violations) == 1
    assert "default" in violations[0]


def test_a_narrower_default_covers_every_unnamed_tool():
    parent = _policy({"forget_fact": "deny"}, default="allow")
    child = _policy({"read": "allow"}, default="deny")

    assert child.narrows(parent)


def test_every_violation_is_reported_not_just_the_first():
    """So one message lists everything to fix rather than making the
    reader rerun to find the next problem."""
    parent = _policy({"a": "deny", "b": "deny", "c": "deny"}, default="deny")
    child = _policy({"a": "allow", "b": "allow", "c": "allow"}, default="deny")

    assert len(child.narrowing_violations(parent)) == 3


def test_violations_name_both_outcomes_so_the_fix_is_obvious():
    parent = _policy({"ocr_extract_text": "require-user-confirmation"})
    child = _policy({"ocr_extract_text": "allow"})

    violation = child.narrowing_violations(parent)[0]

    assert "ocr_extract_text" in violation
    assert "allow" in violation
    assert "require-user-confirmation" in violation


def test_a_narrowing_policy_reports_no_violations():
    parent = _policy({"forget_fact": "deny"}, default="allow")
    child = _policy({"read": "allow", "search": "allow"}, default="deny")

    assert child.narrowing_violations(parent) == []
    assert child.narrows(parent)


# ── Enforcement: the shared primitive ────────────────────────────────

def test_require_narrower_accepts_a_narrowing_spec():
    parent = _policy({"forget_fact": "deny"}, default="allow")

    require_narrower(_spec(policy=_policy({"read": "allow"}, default="deny")), parent)


def test_require_narrower_rejects_an_escalating_spec_and_names_the_role():
    parent = _policy({"ocr_extract_text": "require-user-confirmation"}, default="allow")
    spec = _spec(role="receipt-reader",
                 policy=_policy({"ocr_extract_text": "allow"}, default="deny"))

    with pytest.raises(PolicyEscalation) as exc:
        require_narrower(spec, parent)

    message = str(exc.value)
    assert "receipt-reader" in message
    assert "ocr_extract_text" in message
    assert "may only narrow" in message


def test_policy_escalation_is_catchable_as_a_value_error():
    """So a caller that only cares the config was rejected keeps working,
    while one that wants to report an escalation specifically can."""
    assert issubclass(PolicyEscalation, ValueError)

    with pytest.raises(ValueError):
        require_narrower(_spec(policy=_policy(default="allow")), _policy(default="deny"))


# ── Enforcement: the startup layer ───────────────────────────────────

def _catalog(**roles):
    base = {
        "description": "Does one narrow thing well enough to describe.",
        "instructions": "You are a specialist.",
    }
    return AgentCatalog.from_config({
        "roles": {name: {**base, **definition} for name, definition in roles.items()}
    })


def test_a_catalog_of_narrowing_roles_passes():
    parent = _policy({"forget_fact": "deny"}, default="allow")
    catalog = _catalog(
        reader={"permissions": {"default": "deny", "tools": {"read": "allow"}}},
        searcher={"permissions": {"default": "deny", "tools": {"search": "allow"}}},
    )

    catalog.require_narrower_than(parent)


def test_an_escalating_role_fails_the_startup_check():
    parent = _policy({"ocr_extract_text": "require-user-confirmation"}, default="allow")
    catalog = _catalog(
        reader={"permissions": {"default": "deny", "tools": {"ocr_extract_text": "allow"}}},
    )

    with pytest.raises(PolicyEscalation, match="ocr_extract_text"):
        catalog.require_narrower_than(parent)


def test_a_role_with_no_permissions_of_its_own_is_caught_against_a_strict_parent():
    """An omitted permissions block means the all-allow default, which is
    wider than any restrictive parent -- silently the opposite of what a
    specialized role is for."""
    catalog = _catalog(reader={})

    with pytest.raises(PolicyEscalation, match="default"):
        catalog.require_narrower_than(_policy(default="deny"))


def test_every_failing_role_appears_in_one_message():
    parent = _policy(default="deny")
    catalog = _catalog(first={}, second={})

    with pytest.raises(PolicyEscalation) as exc:
        catalog.require_narrower_than(parent)

    assert "first" in str(exc.value)
    assert "second" in str(exc.value)


def test_an_empty_catalog_passes_any_parent():
    AgentCatalog.from_config(None).require_narrower_than(_policy(default="deny"))


# ── The config this project ships ────────────────────────────────────

def _shipped():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_root, "config.json")
    if not os.path.isfile(config_path):
        pytest.skip("config.json not found at repo root")
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return (
        PermissionPolicy.from_config(raw.get("permissions")),
        AgentCatalog.from_config(raw.get("agents")),
    )


def test_the_shipped_agents_narrow_the_shipped_policy():
    """The check main.py runs at startup. If this fails, the agent will
    not start."""
    parent, catalog = _shipped()

    catalog.require_narrower_than(parent)


def test_no_shipped_agent_needs_a_confirmation_prompt():
    """Deliberate: every sub-agent tool is one the top-level policy
    already allows outright, so a delegation demo does not stop to ask
    about each tool the sub-agent uses."""
    parent, catalog = _shipped()

    for role in catalog.roles():
        for tool, outcome in catalog.get(role).policy.rules().items():
            if outcome is Decision.ALLOW:
                assert parent.decide(tool) is Decision.ALLOW, (
                    f"agent {role!r} allows {tool!r}, which the top-level policy "
                    f"gates as '{parent.decide(tool).value}' -- delegating would "
                    f"prompt, or escalate"
                )
