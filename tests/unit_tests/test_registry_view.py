"""
Unit tests for RegistryView -- a policy-scoped view onto a shared
ToolRegistry.

The view exists so a sub-agent can have its own permissions without its
own registry. Two tests below pin that directly by asserting the shared
registry is used rather than rebuilt, because the reasons are concrete:
a registry per sub-agent re-runs MCP discovery (a subprocess per server)
on every delegation, and each one builds its own MemoryStore, which
loads the receipt file into memory and writes it back whole -- two of
them over one file lose each other's writes.

The rest covers the thing easiest to get wrong: the view is an
enforcement layer, not a pass-through. A caller holding a view must be
bound by the policy it was given even when it never goes near the loop.
"""

from unittest.mock import MagicMock

import pytest

from agent.hooks import HookManager
from agent.permissions import Decision, PermissionPolicy
from agent.tool_registry import RegistryView


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _registry(names=("read", "modify", "forget_fact", "generate_qr_code")):
    registry = MagicMock()
    registry.schemas.return_value = [_schema(n) for n in names]
    registry.call.return_value = "tool ran"
    registry.is_builtin.return_value = True
    return registry


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


# ── It shares the registry rather than replacing it ──────────────────

def test_the_view_delegates_to_the_shared_registry(tmp_path):
    registry = _registry()
    view = RegistryView(registry, PermissionPolicy(default="allow"))

    view.call("read", {"path": "a.txt"})

    registry.call.assert_called_once_with("read", {"path": "a.txt"}, confirmed=None)


def test_two_views_share_one_registry():
    """The point of the view: two differently-permissioned agents, one
    registry, so MCP discovery and the receipt store happen once."""
    registry = _registry()
    reader = RegistryView(registry, PermissionPolicy(
        rules={"read": "allow"}, default="deny"))
    labeller = RegistryView(registry, PermissionPolicy(
        rules={"generate_qr_code": "allow"}, default="deny"))

    reader.call("read", {})
    labeller.call("generate_qr_code", {})

    assert registry.call.call_count == 2
    assert _names(reader.schemas()) == ["read"]
    assert _names(labeller.schemas()) == ["generate_qr_code"]


def test_is_builtin_is_passed_straight_through():
    registry = _registry()
    registry.is_builtin.return_value = False

    assert RegistryView(registry, PermissionPolicy()).is_builtin("ocr_extract_text") is False
    registry.is_builtin.assert_called_once_with("ocr_extract_text")


def test_the_view_exposes_the_three_methods_the_loop_uses():
    """The loop touches schemas(), is_builtin() and call() and nothing
    else, which is what makes a view able to stand in for a registry."""
    view = RegistryView(_registry(), PermissionPolicy())

    for method in ("schemas", "is_builtin", "call"):
        assert callable(getattr(view, method))


# ── Enforcement, not pass-through ────────────────────────────────────

def test_a_denied_tool_never_reaches_the_shared_registry():
    """The view refuses before the registry is touched at all."""
    registry = _registry()
    view = RegistryView(registry, PermissionPolicy(rules={"modify": "deny"}))

    result = view.call("modify", {"path": "a", "content": "b"})

    assert result.startswith("Permission denied:")
    registry.call.assert_not_called()


def test_a_confirmation_tool_is_refused_without_one():
    registry = _registry()
    view = RegistryView(registry, PermissionPolicy(
        rules={"modify": "require-user-confirmation"}))

    result = view.call("modify", {})

    assert "requires the user's confirmation" in result
    registry.call.assert_not_called()


def test_a_confirmed_call_passes_the_confirmation_through():
    """So the registry underneath does not ask a second time for a
    confirmation the loop already obtained."""
    registry = _registry()
    view = RegistryView(registry, PermissionPolicy(
        rules={"modify": "require-user-confirmation"}))

    view.call("modify", {"path": "a"}, confirmed=True)

    registry.call.assert_called_once_with("modify", {"path": "a"}, confirmed=True)


def test_the_view_never_prompts(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the view tried to prompt")

    monkeypatch.setattr("builtins.input", explode)
    view = RegistryView(_registry(), PermissionPolicy(
        rules={"modify": "require-user-confirmation"}))

    assert view.call("modify", {}).startswith("Permission denied:")


def test_there_is_no_way_to_override_the_registrys_own_policy():
    """The view can only refuse what the registry would have allowed,
    never the reverse -- so no parameter exists that would let a caller
    hand the registry a wider policy than its own."""
    view = RegistryView(_registry(), PermissionPolicy())

    with pytest.raises(TypeError):
        view.call("read", {}, policy=PermissionPolicy(default="allow"))


def test_a_non_string_tool_name_is_reported_rather_than_raising():
    assert RegistryView(_registry(), PermissionPolicy()).call(None, {}).startswith("Error:")


# ── Schema filtering: only for an allowlist policy ───────────────────

def test_an_allowlist_policy_hides_the_tools_it_denies():
    """A specialist should not be shown fifteen tools it cannot use --
    it would call them and spend an iteration finding out each time."""
    view = RegistryView(_registry(), PermissionPolicy(
        rules={"read": "allow", "generate_qr_code": "allow"}, default="deny"))

    assert sorted(_names(view.schemas())) == ["generate_qr_code", "read"]


def test_a_confirmation_gated_tool_is_still_shown_under_an_allowlist():
    """It can be granted, so hiding it would remove a capability the
    policy deliberately kept available."""
    view = RegistryView(_registry(), PermissionPolicy(
        rules={"modify": "require-user-confirmation"}, default="deny"))

    assert _names(view.schemas()) == ["modify"]


def test_a_denylist_policy_still_shows_every_tool():
    """Deliberately different. A broad agent's refused tools stay visible
    so the refusal actually happens and is observable -- a tool that is
    never offered is never denied either, and the deny outcome would stop
    being exercised."""
    view = RegistryView(_registry(), PermissionPolicy(
        rules={"forget_fact": "deny"}, default="allow"))

    assert "forget_fact" in _names(view.schemas())


def test_a_denied_tool_that_is_still_offered_is_still_refused():
    """The other half of the previous test: visible is not permitted."""
    registry = _registry()
    view = RegistryView(registry, PermissionPolicy(
        rules={"forget_fact": "deny"}, default="allow"))

    assert "forget_fact" in _names(view.schemas())
    assert view.call("forget_fact", {"id": "1"}).startswith("Permission denied:")
    registry.call.assert_not_called()


def test_an_all_allow_view_changes_nothing():
    registry = _registry()
    view = RegistryView(registry, PermissionPolicy(default="allow"))

    assert view.schemas() == registry.schemas.return_value


# ── Reporting ────────────────────────────────────────────────────────

def test_a_view_refusal_is_reported_on_the_hook_bus(monkeypatch):
    manager = HookManager()
    events = []
    manager.on("permission_decision", lambda name, **p: events.append(p))
    monkeypatch.setattr("agent.tool_registry.hooks", manager)

    view = RegistryView(_registry(), PermissionPolicy(rules={"modify": "deny"}))
    view.call("modify", {})

    assert len(events) == 1
    assert events[0]["tool_name"] == "modify"
    assert events[0]["decision"] == Decision.DENY.value
    assert events[0]["allowed"] is False


def test_an_allowed_call_is_not_reported_by_the_view(monkeypatch):
    """Only refusals, for the same reason as the registry: the loop
    already reported every decision and only reaches here for calls it
    allowed, so reporting allows again would double-count."""
    manager = HookManager()
    events = []
    manager.on("permission_decision", lambda name, **p: events.append(p))
    monkeypatch.setattr("agent.tool_registry.hooks", manager)

    RegistryView(_registry(), PermissionPolicy(default="allow")).call("read", {})

    assert events == []
