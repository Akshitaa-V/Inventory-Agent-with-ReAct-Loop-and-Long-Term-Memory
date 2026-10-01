"""
Unit tests for agent/agents.py -- sub-agent definitions from config.

This module is only definitions: it reads the "agents" section, validates
it, and hands back immutable specs. Nothing here runs an agent, so every
test is a plain in-memory call with no LLM, registry or subprocess.

Validation is tested harder than the happy path, for the same reason as
in test_permissions.py: a definition that is silently half-understood is
worse than one that stops the process. A role whose permissions were
dropped because of a mistyped key would run with the empty all-allow
default -- the opposite of the restriction it was written to express.
"""

import json
import os

import pytest

from agent.agents import (
    DEFAULT_MAX_DELEGATION_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    AgentCatalog,
    AgentSpec,
)
from agent.config import load_config
from agent.permissions import Decision


def _role(**overrides):
    definition = {
        "description": "Does one narrow thing.",
        "instructions": "You are a specialist. Do the one thing.",
    }
    definition.update(overrides)
    return definition


def _catalog(**roles):
    return AgentCatalog.from_config({"roles": roles})


# ── The happy path ───────────────────────────────────────────────────

def test_a_role_is_read_with_its_own_instructions_and_limit():
    catalog = _catalog(summarizer=_role(max_iterations=3))
    spec = catalog.get("summarizer")

    assert isinstance(spec, AgentSpec)
    assert spec.role == "summarizer"
    assert spec.instructions.startswith("You are a specialist")
    assert spec.max_iterations == 3


def test_a_role_gets_its_own_permission_policy():
    """8.4 requires a sub-agent to have its own permissions. Nested here
    and validated by PermissionPolicy, so sub-agent rules go through
    exactly the same code as the top-level ones."""
    catalog = _catalog(reader=_role(permissions={
        "default": "deny",
        "tools": {"ocr_extract_text": "allow"},
    }))
    policy = catalog.get("reader").policy

    assert policy.decide("ocr_extract_text") is Decision.ALLOW
    assert policy.decide("modify") is Decision.DENY


def test_two_roles_are_both_available():
    """8.4 asks for at least two specialized configurations."""
    catalog = _catalog(first=_role(), second=_role())

    assert len(catalog) == 2
    assert catalog.roles() == ["first", "second"]


def test_roles_are_listed_in_a_stable_order():
    catalog = _catalog(zeta=_role(), alpha=_role(), mid=_role())

    assert catalog.roles() == ["alpha", "mid", "zeta"]


def test_a_spec_is_immutable_once_validated():
    spec = _catalog(reader=_role()).get("reader")

    with pytest.raises(Exception):
        spec.max_iterations = 999


# ── Defaults ─────────────────────────────────────────────────────────

def test_a_role_without_a_limit_inherits_the_section_default():
    catalog = AgentCatalog.from_config({
        "max_iterations": 5,
        "roles": {"reader": _role()},
    })

    assert catalog.get("reader").max_iterations == 5


def test_a_role_limit_overrides_the_section_default():
    catalog = AgentCatalog.from_config({
        "max_iterations": 5,
        "roles": {"reader": _role(max_iterations=2)},
    })

    assert catalog.get("reader").max_iterations == 2


def test_with_no_section_default_the_module_default_applies():
    assert _catalog(reader=_role()).get("reader").max_iterations == DEFAULT_MAX_ITERATIONS


def test_model_and_temperature_default_to_inheriting():
    """None means "use the process LLM config", so a role that does not
    care about the model does not have to restate it."""
    spec = _catalog(reader=_role()).get("reader")

    assert spec.model is None
    assert spec.temperature is None


def test_model_and_temperature_can_be_overridden_per_role():
    spec = _catalog(reader=_role(model="small-model", temperature=0.1)).get("reader")

    assert spec.model == "small-model"
    assert spec.temperature == 0.1


def test_memory_injection_defaults_to_off():
    """A specialist works from the brief its parent hands it; injecting
    the whole fact store would dilute that and cost tokens every time."""
    assert _catalog(reader=_role()).get("reader").include_memory is False


def test_memory_injection_can_be_turned_on():
    assert _catalog(reader=_role(include_memory=True)).get("reader").include_memory is True


def test_the_delegation_depth_has_a_default_and_is_configurable():
    assert _catalog(reader=_role()).max_delegation_depth == DEFAULT_MAX_DELEGATION_DEPTH

    deeper = AgentCatalog.from_config({"max_delegation_depth": 3, "roles": {}})
    assert deeper.max_delegation_depth == 3


def test_a_depth_of_zero_is_allowed_and_means_no_delegation():
    """A way to switch delegation off without deleting the definitions."""
    assert AgentCatalog.from_config({"max_delegation_depth": 0, "roles": {}}).max_delegation_depth == 0


# ── An absent section keeps older configs working ────────────────────

@pytest.mark.parametrize("raw", [None, {}, {"roles": {}}, {"roles": None}])
def test_no_section_yields_an_empty_catalog(raw):
    catalog = AgentCatalog.from_config(raw)

    assert len(catalog) == 0
    assert catalog.roles() == []
    assert "anything" not in catalog


def test_an_empty_catalog_says_so_in_its_summary():
    assert "none configured" in AgentCatalog.from_config(None).summary()


# ── Lookup ───────────────────────────────────────────────────────────

def test_an_unknown_role_raises_and_names_what_is_available():
    """There is no default agent to fall back to, so this must error --
    but the message has to say what the caller could have asked for."""
    catalog = _catalog(reader=_role(), checker=_role())

    with pytest.raises(KeyError) as exc:
        catalog.get("nonexistent")

    message = str(exc.value)
    assert "reader" in message and "checker" in message


def test_membership_can_be_checked_without_raising():
    """A runtime caller handed a role name by the model checks first and
    reports an unknown one back as a tool result, rather than crashing."""
    catalog = _catalog(reader=_role())

    assert "reader" in catalog
    assert "nope" not in catalog
    assert None not in catalog


def test_lookup_tolerates_surrounding_whitespace():
    catalog = _catalog(reader=_role())

    assert catalog.get(" reader ").role == "reader"
    assert " reader " in catalog


# ── Validation: the section ──────────────────────────────────────────

def test_an_unknown_section_key_raises():
    """A "role"/"roles" typo would otherwise silently configure no agents
    at all while looking correct."""
    with pytest.raises(ValueError, match="unknown key"):
        AgentCatalog.from_config({"role": {"reader": _role()}})


def test_a_non_object_section_raises():
    with pytest.raises(ValueError, match="must be an object"):
        AgentCatalog.from_config("receipt-reader")


def test_a_non_object_roles_section_raises():
    with pytest.raises(ValueError, match="agents.roles"):
        AgentCatalog.from_config({"roles": ["reader"]})


@pytest.mark.parametrize("bad", [0, -1, 1.5, "8", True, None])
def test_a_bad_section_max_iterations_raises(bad):
    with pytest.raises(ValueError, match="agents.max_iterations"):
        AgentCatalog.from_config({"max_iterations": bad, "roles": {}})


@pytest.mark.parametrize("bad", [-1, 1.5, "1", True])
def test_a_bad_delegation_depth_raises(bad):
    with pytest.raises(ValueError, match="max_delegation_depth"):
        AgentCatalog.from_config({"max_delegation_depth": bad, "roles": {}})


# ── Validation: a role definition ────────────────────────────────────

def test_an_unknown_role_key_raises_and_names_the_role():
    """Mistyping "permissions" would leave the role on the all-allow
    default -- the opposite of the restriction it was written to express."""
    with pytest.raises(ValueError, match="reader"):
        _catalog(reader=_role(permission={"default": "deny"}))


@pytest.mark.parametrize("missing", ["description", "instructions"])
def test_a_missing_required_key_raises(missing):
    definition = _role()
    del definition[missing]

    with pytest.raises(ValueError, match=missing):
        _catalog(reader=definition)


@pytest.mark.parametrize("key", ["description", "instructions"])
@pytest.mark.parametrize("bad", ["", "   ", 5, None])
def test_an_empty_or_non_string_required_key_raises(key, bad):
    with pytest.raises(ValueError, match=key):
        _catalog(reader=_role(**{key: bad}))


def test_a_non_object_role_definition_raises():
    with pytest.raises(ValueError, match="must be an object"):
        _catalog(reader="just a string")


def test_a_bad_role_max_iterations_names_the_role():
    with pytest.raises(ValueError, match="reader"):
        _catalog(reader=_role(max_iterations=0))


def test_bad_nested_permissions_are_reported_against_the_role():
    """The nested policy is validated by PermissionPolicy, but the error
    has to say which agent's rules to go and fix."""
    with pytest.raises(ValueError) as exc:
        _catalog(reader=_role(permissions={"tools": {"read": "deney"}}))

    message = str(exc.value)
    assert "reader" in message
    assert "deney" in message


@pytest.mark.parametrize("bad", [-0.1, 2.5, "0.5", True])
def test_a_bad_temperature_raises(bad):
    with pytest.raises(ValueError, match="temperature"):
        _catalog(reader=_role(temperature=bad))


def test_a_non_string_model_raises():
    with pytest.raises(ValueError, match="model"):
        _catalog(reader=_role(model=5))


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_a_non_boolean_include_memory_raises(bad):
    with pytest.raises(ValueError, match="include_memory"):
        _catalog(reader=_role(include_memory=bad))


# ── Validation: role names ───────────────────────────────────────────

def test_an_empty_role_name_raises():
    with pytest.raises(ValueError, match="empty name"):
        _catalog(**{"   ": _role()})


@pytest.mark.parametrize("bad", ["has space", "has\nnewline", 'has"quote', "-leading", ""])
def test_an_unusable_role_name_raises(bad):
    """A role name becomes a Prometheus label value and a log field, so
    names that are legal JSON keys but unreadable in either are refused."""
    with pytest.raises(ValueError):
        AgentCatalog.from_config({"roles": {bad: _role()}})


@pytest.mark.parametrize("ok", ["receipt-reader", "recall_checker", "agent.v2", "a", "A1"])
def test_reasonable_role_names_are_accepted(ok):
    assert AgentCatalog.from_config({"roles": {ok: _role()}}).roles() == [ok]


def test_names_colliding_only_on_whitespace_raise():
    with pytest.raises(ValueError, match="duplicate agent role"):
        AgentCatalog.from_config({"roles": {"reader": _role(), " reader ": _role()}})


def test_a_duplicate_role_passed_directly_raises():
    spec = _catalog(reader=_role()).get("reader")

    with pytest.raises(ValueError, match="duplicate agent role"):
        AgentCatalog(specs=[spec, spec])


# ── Config loading, and the config this project ships ────────────────

def test_load_config_exposes_the_agents_section(tmp_path, monkeypatch):
    monkeypatch.setenv("INNKUBE_TOKEN", "test-token")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "llm": {"model": "m"},
        "agents": {"roles": {"reader": _role(max_iterations=2)}},
    }))

    config = load_config(str(config_file))
    catalog = AgentCatalog.from_config(config.agents)

    assert catalog.get("reader").max_iterations == 2


def test_load_config_without_an_agents_section_yields_an_empty_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("INNKUBE_TOKEN", "test-token")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"llm": {"model": "m"}}))

    config = load_config(str(config_file))

    assert len(AgentCatalog.from_config(config.agents)) == 0


def _shipped_catalog():
    """The catalog from config.json -- the file the agent actually runs
    on. Skipped when absent, as in test_permissions.py: config.json is
    gitignored, so it is not in a fresh clone or the CI unit-test job,
    but it IS in the Docker image where the build gate runs."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_root, "config.json")
    if not os.path.isfile(config_path):
        pytest.skip("config.json not found at repo root -- no live agents to check")
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return AgentCatalog.from_config(raw.get("agents"))


def test_the_projects_own_config_defines_at_least_two_agents():
    """8.4 requires at least two specialized sub-agent configurations."""
    catalog = _shipped_catalog()

    assert len(catalog) >= 2, f"only {len(catalog)} agent role(s) configured"


def test_every_shipped_agent_restricts_its_own_tools():
    """A sub-agent whose policy allowed everything would not be
    specialized in any enforceable sense -- the point of a narrow role is
    a narrow reach."""
    catalog = _shipped_catalog()

    for role in catalog.roles():
        policy = catalog.get(role).policy
        assert policy.default is Decision.DENY, (
            f"agent {role!r} defaults to '{policy.default.value}', so it can "
            f"reach every tool in the harness"
        )
        assert policy.rules(), f"agent {role!r} allows no tools at all"


def test_every_shipped_agent_has_a_description_aimed_at_its_parent():
    """The description is what the parent agent reads when deciding
    whether to delegate, so an empty or placeholder one makes the role
    undiscoverable in practice."""
    catalog = _shipped_catalog()

    for role in catalog.roles():
        description = catalog.get(role).description
        assert len(description) > 30, f"agent {role!r} has a too-thin description"


def test_the_shipped_delegation_depth_bounds_recursion():
    """Without a bound, handing a sub-agent the delegation tool makes
    unbounded recursion reachable from one user request."""
    assert _shipped_catalog().max_delegation_depth <= 2
