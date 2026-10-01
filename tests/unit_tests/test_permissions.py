"""
Unit tests for agent/permissions.py -- the permission policy model.

The policy is pure: it resolves a tool name to an outcome and does
nothing else. So every test here is a plain in-memory call with no
registry, no LLM, no stdin and no Prometheus -- which is the whole
reason the model was kept separate from the wiring that enforces it.

Two things are tested harder than the rest, because both are silent
failure modes rather than loud ones:
  - a misconfigured outcome must raise, never quietly become the default
    (a rule that reads as 'allow' is worse than no rule at all), and
  - the Decision values must stay byte-identical to the label strings
    metrics.py already expects, since nothing else would catch a drift
    between the two files until a dashboard panel came up empty.
"""

import json
import os

import pytest

from agent.config import load_config
from agent.permissions import (
    DEFAULT_DECISION,
    Decision,
    PermissionPolicy,
    parse_decision,
)


# ── The three required outcomes ──────────────────────────────────────

def test_all_three_handout_outcomes_resolve_for_explicit_rules():
    """8.4 requires exactly these three outcomes to be supported."""
    policy = PermissionPolicy(rules={
        "read": "allow",
        "forget_fact": "deny",
        "modify": "require-user-confirmation",
    })

    assert policy.decide("read") is Decision.ALLOW
    assert policy.decide("forget_fact") is Decision.DENY
    assert policy.decide("modify") is Decision.REQUIRE_USER_CONFIRMATION


def test_decision_values_match_the_metrics_label_contract():
    """metrics.record_permission_decision() takes the decision as a
    string label. These values are what it documents, so a caller can
    pass decision.value straight through. If someone renames one of
    these, this test is the only thing that will notice."""
    assert Decision.ALLOW.value == "allow"
    assert Decision.DENY.value == "deny"
    assert Decision.REQUIRE_USER_CONFIRMATION.value == "require-user-confirmation"


def test_decision_compares_equal_to_its_own_spelling():
    """Decision subclasses str, so log formatting and label passing work
    without callers reaching for .value everywhere."""
    assert Decision.DENY == "deny"
    assert f"{Decision.DENY.value}" == "deny"


# ── Defaults ─────────────────────────────────────────────────────────

def test_unlisted_tool_falls_through_to_the_default():
    policy = PermissionPolicy(rules={"modify": "deny"}, default="require-user-confirmation")

    assert policy.decide("some_other_tool") is Decision.REQUIRE_USER_CONFIRMATION


def test_default_is_allow_when_nothing_is_configured():
    """An empty policy must behave exactly as the agent did before
    permissions existed, so adding this module is a no-op until rules
    are actually written."""
    policy = PermissionPolicy()

    assert policy.default is Decision.ALLOW
    assert DEFAULT_DECISION is Decision.ALLOW
    assert policy.decide("anything_at_all") is Decision.ALLOW


def test_a_deny_default_blocks_everything_not_explicitly_allowed():
    """The allowlist posture: deny by default, open only what is named."""
    policy = PermissionPolicy(rules={"read": "allow"}, default="deny")

    assert policy.decide("read") is Decision.ALLOW
    assert policy.decide("modify") is Decision.DENY
    assert policy.decide("ocr_extract_text") is Decision.DENY


def test_an_explicit_rule_beats_the_default():
    policy = PermissionPolicy(rules={"read": "allow"}, default="deny")

    assert policy.decide("read") is Decision.ALLOW
    assert policy.source_for("read") == "rule"


# ── Built-in and MCP tools share one namespace ───────────────────────

def test_mcp_tool_names_resolve_exactly_like_builtin_ones():
    """8.4 asks for policies covering built-in *and* MCP tools. Both are
    flat names in the same namespace by the time they are callable, so
    the model needs no builtin/mcp branch -- this pins that claim."""
    policy = PermissionPolicy(rules={
        "modify": "deny",                            # built-in
        "generate_qr_code": "deny",                  # MCP (self-hosted QR server)
        "ocr_extract_text": "require-user-confirmation",  # MCP (OCR server)
        "check_product_recall": "allow",             # MCP (recall server)
    })

    assert policy.decide("modify") is Decision.DENY
    assert policy.decide("generate_qr_code") is Decision.DENY
    assert policy.decide("ocr_extract_text") is Decision.REQUIRE_USER_CONFIRMATION
    assert policy.decide("check_product_recall") is Decision.ALLOW


# ── Outcome parsing and normalization ────────────────────────────────

@pytest.mark.parametrize("spelling", [
    "require-user-confirmation",
    "require_user_confirmation",
    "REQUIRE-USER-CONFIRMATION",
    "  Require_User_Confirmation  ",
])
def test_outcome_spellings_all_normalize_to_one_decision(spelling):
    """Config is hand-written, so underscores, case and stray whitespace
    are all accepted and collapsed to the canonical form."""
    assert parse_decision(spelling) is Decision.REQUIRE_USER_CONFIRMATION


def test_parse_decision_passes_a_decision_through_unchanged():
    assert parse_decision(Decision.DENY) is Decision.DENY


def test_an_unknown_outcome_raises_instead_of_falling_back():
    """The important one. A typo silently becoming the default would read
    as 'allow' at runtime: a policy that looks configured and enforces
    nothing."""
    with pytest.raises(ValueError, match="unknown permission outcome"):
        parse_decision("deney")


def test_an_unknown_outcome_error_lists_the_valid_ones():
    with pytest.raises(ValueError) as exc:
        parse_decision("maybe")

    message = str(exc.value)
    assert "allow" in message
    assert "deny" in message
    assert "require-user-confirmation" in message


def test_a_non_string_outcome_raises():
    with pytest.raises(ValueError, match="must be a string"):
        parse_decision(True)


def test_a_bad_outcome_in_a_rule_names_the_offending_tool():
    """With a dozen rules, "unknown outcome 'deney'" alone does not say
    which line of config to go and fix."""
    with pytest.raises(ValueError, match="forget_fact"):
        PermissionPolicy(rules={"forget_fact": "deney"})


def test_a_bad_default_raises_at_construction_time():
    with pytest.raises(ValueError, match="unknown permission outcome"):
        PermissionPolicy(default="permit")


# ── Tool-name handling ───────────────────────────────────────────────

def test_tool_names_are_matched_after_stripping_whitespace():
    policy = PermissionPolicy(rules={"  modify  ": "deny"})

    assert policy.decide("modify") is Decision.DENY
    assert policy.decide("  modify  ") is Decision.DENY


def test_tool_names_are_matched_case_sensitively():
    """Tool names are case-sensitive identifiers the model emits verbatim
    from the schemas. Folding case would make a rule for 'modify' look
    like it covered 'Modify', which is not a real tool at all."""
    policy = PermissionPolicy(rules={"modify": "deny"}, default="allow")

    assert policy.decide("modify") is Decision.DENY
    assert policy.decide("Modify") is Decision.ALLOW


def test_an_empty_tool_name_raises():
    with pytest.raises(ValueError, match="empty tool name"):
        PermissionPolicy(rules={"   ": "deny"})


def test_names_colliding_only_on_whitespace_raise():
    """Both keys collapse to the same tool, so one would silently
    overwrite the other. Better to refuse than to pick a winner the
    config's author never chose."""
    with pytest.raises(ValueError, match="duplicate permission rule"):
        PermissionPolicy(rules={"modify": "deny", " modify ": "allow"})


def test_an_unknown_tool_name_gets_the_default_rather_than_raising():
    """Rejecting unknown tools is the registry's job -- it already
    reports them as "unknown tool". A policy that raised here would turn
    a model typo into a crashed run."""
    policy = PermissionPolicy(default="deny")

    assert policy.decide("no_such_tool") is Decision.DENY


def test_decide_rejects_a_non_string_tool_name():
    with pytest.raises(ValueError, match="tool name must be a string"):
        PermissionPolicy().decide(None)


# ── source_for: for the structured logs the handout asks for ─────────

def test_source_for_distinguishes_a_rule_from_the_default():
    policy = PermissionPolicy(rules={"modify": "deny"})

    assert policy.source_for("modify") == "rule"
    assert policy.source_for("read") == "default"


# ── from_config ──────────────────────────────────────────────────────

def test_from_config_reads_the_default_and_the_tool_rules():
    policy = PermissionPolicy.from_config({
        "default": "deny",
        "tools": {"read": "allow", "modify": "require-user-confirmation"},
    })

    assert policy.default is Decision.DENY
    assert policy.decide("read") is Decision.ALLOW
    assert policy.decide("modify") is Decision.REQUIRE_USER_CONFIRMATION
    assert policy.decide("anything_else") is Decision.DENY


@pytest.mark.parametrize("raw", [None, {}, {"tools": {}}, {"tools": None}])
def test_from_config_with_no_rules_is_an_all_allow_policy(raw):
    """A config.json predating this feature must keep working unchanged."""
    policy = PermissionPolicy.from_config(raw)

    assert policy.decide("modify") is Decision.ALLOW
    assert policy.rules() == {}


def test_from_config_rejects_an_unknown_key():
    """A "tool"/"tools" typo would otherwise make every rule vanish and
    leave the agent wide open while looking correctly configured."""
    with pytest.raises(ValueError, match="unknown key"):
        PermissionPolicy.from_config({"tool": {"modify": "deny"}})


def test_from_config_rejects_a_non_object_tools_section():
    with pytest.raises(ValueError, match="permissions.tools"):
        PermissionPolicy.from_config({"tools": ["modify"]})


def test_from_config_rejects_a_non_object_section():
    with pytest.raises(ValueError, match="must be an object"):
        PermissionPolicy.from_config("allow")


# ── Immutability of a validated policy ───────────────────────────────

def test_rules_returns_a_copy_so_a_validated_policy_cannot_be_mutated():
    policy = PermissionPolicy(rules={"modify": "deny"})

    policy.rules()["modify"] = Decision.ALLOW

    assert policy.decide("modify") is Decision.DENY


# ── summary() ────────────────────────────────────────────────────────

def test_summary_names_the_default_when_there_are_no_rules():
    summary = PermissionPolicy(default="deny").summary()

    assert "deny" in summary
    assert "no explicit rules" in summary


def test_summary_groups_tools_by_decision_in_a_stable_order():
    summary = PermissionPolicy(rules={
        "modify": "require-user-confirmation",
        "forget_fact": "deny",
        "read": "allow",
    }).summary()

    assert "3 rule(s)" in summary
    assert "read" in summary and "forget_fact" in summary and "modify" in summary
    # Grouped in Decision order, not dict order, so the line is stable
    # across runs and across however the config happened to be written.
    assert summary.index("allow:") < summary.index("deny:") < summary.index("require-user-confirmation:")


# ── Integration with config loading (no loop/registry involved) ──────

def test_load_config_exposes_the_permissions_section(tmp_path, monkeypatch):
    monkeypatch.setenv("INNKUBE_TOKEN", "test-token")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "llm": {"model": "m", "base_url": "b", "endpoint": "/e"},
        "permissions": {"default": "deny", "tools": {"read": "allow"}},
    }))

    config = load_config(str(config_file))
    policy = PermissionPolicy.from_config(config.permissions)

    assert policy.decide("read") is Decision.ALLOW
    assert policy.decide("modify") is Decision.DENY


def test_load_config_without_a_permissions_section_yields_an_allow_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("INNKUBE_TOKEN", "test-token")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"llm": {"model": "m"}}))

    config = load_config(str(config_file))
    policy = PermissionPolicy.from_config(config.permissions)

    assert policy.default is Decision.ALLOW
    assert policy.decide("modify") is Decision.ALLOW


# Tools that an end-to-end test drives on its happy path. A tool gated
# behind require-user-confirmation here would make that test block on
# stdin the moment enforcement reaches the loop, so the shipped policy
# must not confirm-gate any of them. Kept as a list of names rather than
# discovered automatically: the real-LLM tests choose their own tool
# calls, so this is a deliberate, reviewable statement of which tools the
# suite depends on being ungated.
TOOLS_ON_E2E_HAPPY_PATHS = frozenset({
    "create",              # all four e2e tests
    "modify",              # test_number_in_words, test_receipt_to_inventory
    "read",                # all four
    "read_many",           # reachable by the real LLM in the receipt test
    "navigate",            # reachable by the real LLM in the receipt test
    "remember_fact",       # test_agent_flow
    "recall_fact",         # test_agent_flow
    "generate_qr_code",    # test_agent_flow, test_qr_round_trip
    # The system prompt instructs the model to call these around any
    # receipt it adds to inventory, so the real-LLM receipt test reaches
    # them even though no test names them literally.
    "check_receipt_processed",
    "mark_receipt_processed",
})


def _shipped_policy():
    """The policy from config.json -- the file the agent actually runs
    on. config_example.json is only the template users copy from, so a
    mistake in it is a mistake in a starting point; a mistake in
    config.json is what actually misbehaves at runtime.

    config.json is listed in .gitignore but is in fact tracked, so a
    fresh clone and the GitLab unit-tests job both have it; the skip
    only fires if someone removed it locally. Absent means skip, not
    fail -- a permanently red pipeline over a file that cannot exist
    there reports nothing anyone can act on. The same
    skip-when-there-is-no-config pattern is already used by
    tests/e2e_tests/test_qr_round_trip.py. A config.json that exists
    but is broken still fails, which is the case worth catching.
    (The unit tests do NOT gate the Docker image -- that gate was
    removed with the multi-stage build; run them yourself before
    building if you want the same guarantee.)
    """
    # Resolved from this file, not the working directory, so the test
    # does not depend on pytest being invoked from the repository root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_root, "config.json")
    if not os.path.isfile(config_path):
        pytest.skip("config.json not found at repo root -- no live policy to check")
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return PermissionPolicy.from_config(raw.get("permissions"))


def test_the_projects_own_config_is_a_valid_policy():
    """A broken permissions section would stop the agent starting at all,
    since from_config() rejects a bad one rather than limping on."""
    assert _shipped_policy().rules(), "config.json defines no permission rules"


def test_the_shipped_policy_demonstrates_all_three_outcomes():
    """8.4 asks for allow, deny and require-user-confirmation. The demo
    only evidences that if the config the agent runs on actually
    exercises all three -- the default covers allow, so the rules must
    cover the other two."""
    policy = _shipped_policy()
    outcomes = set(policy.rules().values()) | {policy.default}

    assert outcomes == set(Decision), (
        f"shipped policy does not exercise every outcome; missing: "
        f"{sorted(d.value for d in set(Decision) - outcomes)}"
    )


def test_the_shipped_policy_covers_builtin_and_mcp_tools():
    """8.4 requires the mechanism to apply to built-in *and* MCP tools.
    Named explicitly rather than asked of the registry, so this test
    needs no MCP subprocess to run."""
    builtin_names = {"forget_fact", "create", "modify", "read", "remember_fact"}
    rules = _shipped_policy().rules()

    assert builtin_names & set(rules), "no built-in tool carries an explicit rule"
    assert set(rules) - builtin_names, "no MCP tool carries an explicit rule"


def test_no_tool_on_an_e2e_happy_path_is_confirm_gated():
    """The regression guard. Confirm-gating a tool the e2e suite drives
    would hang that test on stdin once enforcement is wired into the
    loop -- a failure that would show up far from this config change."""
    gated = {
        name
        for name, decision in _shipped_policy().rules().items()
        if decision is Decision.REQUIRE_USER_CONFIRMATION
    }

    offenders = gated & TOOLS_ON_E2E_HAPPY_PATHS
    assert not offenders, (
        f"these tools are confirm-gated but are driven by an end-to-end "
        f"test, which would block on stdin once permissions are enforced: "
        f"{sorted(offenders)}"
    )


def test_denied_tools_are_not_on_an_e2e_happy_path_either():
    """A deny does not block on stdin, but it would still fail the run."""
    denied = {
        name
        for name, decision in _shipped_policy().rules().items()
        if decision is Decision.DENY
    }

    assert not denied & TOOLS_ON_E2E_HAPPY_PATHS, (
        f"denied tools are driven by an end-to-end test: "
        f"{sorted(denied & TOOLS_ON_E2E_HAPPY_PATHS)}"
    )
