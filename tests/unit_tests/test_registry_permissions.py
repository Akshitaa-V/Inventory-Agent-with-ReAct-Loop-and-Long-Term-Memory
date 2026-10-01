"""
Unit tests for the permission boundary inside ToolRegistry.call().

The loop's enforcement (tests/unit_tests/test_loop_permissions.py) is the
layer that asks a human. This one is the layer that cannot be skipped,
so every test here calls registry.call() DIRECTLY, with no loop
involved -- that is the bypass this layer exists to close, and testing
it through react_step would test the wrong thing entirely.

The registry deliberately has no confirmation channel: it never prompts,
so it can neither double-ask after the loop already asked, nor block on
stdin somewhere there is no user. A tool needing confirmation that
arrives here without one is refused, not queried. Several tests below
pin that, because it is the property the whole design rests on.
"""

from unittest.mock import MagicMock

import pytest

from agent.hooks import HookManager
from agent.permissions import Decision, PermissionPolicy
from agent.tool_registry import ToolRegistry


def _registry(tmp_path, policy=None):
    """A registry over a temp workspace with every MCP server disabled,
    so these tests spawn no subprocesses and need no network."""
    return ToolRegistry(
        workspace_root=str(tmp_path),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers={
            "recall": {"enabled": False},
            "ocr": {"enabled": False},
            "qr": {"enabled": False},
        },
        policy=policy,
    )


# ── The bypass case: a direct call with no loop ──────────────────────

def test_a_denied_tool_is_refused_on_a_direct_call(tmp_path):
    """The whole point of this layer. Nothing here went through the loop,
    so nothing else could have stopped it."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "deny"}))

    result = registry.call("create", {"path": "x.txt", "type": "file"})

    assert result.startswith("Permission denied:")
    assert not (tmp_path / "x.txt").exists(), "the denied tool actually ran"


def test_a_denied_tool_returns_a_refusal_rather_than_raising(tmp_path):
    """call() reports every other failure as a result string, and the loop
    feeds that string back to the model. A raise here would be a different
    control path for no reason."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "deny"}))

    result = registry.call("create", {"path": "x.txt", "type": "file"})

    assert isinstance(result, str)


def test_an_allowed_tool_still_runs_on_a_direct_call(tmp_path):
    registry = _registry(tmp_path, PermissionPolicy(default="allow"))

    result = registry.call("create", {"path": "x.txt", "type": "file"})

    assert not result.startswith("Permission denied:")
    assert (tmp_path / "x.txt").exists()


def test_a_deny_by_default_policy_blocks_an_unlisted_tool(tmp_path):
    registry = _registry(tmp_path, PermissionPolicy(default="deny"))

    result = registry.call("create", {"path": "x.txt", "type": "file"})

    assert result.startswith("Permission denied:")
    assert not (tmp_path / "x.txt").exists()


def test_no_policy_means_a_registry_behaves_as_it_did_before(tmp_path):
    """Backwards compatibility: the existing tests build registries with
    no policy, and must keep working unchanged."""
    registry = _registry(tmp_path)

    result = registry.call("create", {"path": "x.txt", "type": "file"})

    assert (tmp_path / "x.txt").exists()
    assert not result.startswith("Permission denied:")


# ── Confirmation, without ever prompting ─────────────────────────────

def test_a_confirmation_tool_is_refused_when_no_confirmation_was_obtained(tmp_path):
    """Fail closed. This is the bypass that matters most: a caller with no
    way to ask a human must not get the tool run for free."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "require-user-confirmation"}))

    result = registry.call("create", {"path": "x.txt", "type": "file"})

    assert result.startswith("Permission denied:")
    assert "requires the user's confirmation" in result
    assert not (tmp_path / "x.txt").exists()


def test_a_confirmation_tool_runs_when_the_caller_confirms_it(tmp_path):
    """What the loop passes after a human said yes."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "require-user-confirmation"}))

    result = registry.call("create", {"path": "x.txt", "type": "file"}, confirmed=True)

    assert not result.startswith("Permission denied:")
    assert (tmp_path / "x.txt").exists()


@pytest.mark.parametrize("confirmed", [False, None, 0, "", "yes", 1])
def test_only_a_literal_true_counts_as_confirmation(tmp_path, confirmed):
    """Checked with `is not True`, so a truthy-but-not-True value cannot
    drift into counting as approval. A caller that means yes says True."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "require-user-confirmation"}))

    result = registry.call("create", {"path": "x.txt", "type": "file"}, confirmed=confirmed)

    assert result.startswith("Permission denied:")
    assert not (tmp_path / "x.txt").exists()


def test_confirmed_cannot_loosen_a_deny_rule(tmp_path):
    """Deny ignores `confirmed` entirely -- the one outcome with no
    caller-assertable escape hatch."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "deny"}))

    result = registry.call("create", {"path": "x.txt", "type": "file"}, confirmed=True)

    assert result.startswith("Permission denied:")
    assert not (tmp_path / "x.txt").exists()


def test_the_registry_never_reads_stdin(tmp_path, monkeypatch):
    """Structural, not incidental: the registry has no confirmation
    channel, so there is nothing that could prompt. If input() is ever
    reached from here, a run with no terminal would hang."""
    def explode(*args, **kwargs):
        raise AssertionError("the registry tried to prompt")

    monkeypatch.setattr("builtins.input", explode)
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "require-user-confirmation"}))

    result = registry.call("create", {"path": "x.txt", "type": "file"})

    assert result.startswith("Permission denied:")


def test_call_accepts_no_confirmation_callback_at_all(tmp_path):
    """There is deliberately no way to give the registry a prompt. If a
    `confirm=` parameter is ever added, the double-prompt guarantee and
    the never-block-on-stdin guarantee both stop being structural."""
    registry = _registry(tmp_path, PermissionPolicy())

    with pytest.raises(TypeError):
        registry.call("create", {"path": "x.txt", "type": "file"}, confirm=lambda *a: True)


# ── Built-in and MCP are covered by the same code path ───────────────

def test_an_mcp_tool_is_refused_exactly_like_a_builtin(tmp_path, monkeypatch):
    """8.4 requires the mechanism to cover both kinds. The check sits
    above the builtin/MCP dispatch, so this is one code path rather than
    two that could drift -- the MCP client must never be reached."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"ocr_extract_text": "deny"}))

    client = MagicMock()
    monkeypatch.setattr(registry, "_mcp_tool_owner", {"ocr_extract_text": client})

    result = registry.call("ocr_extract_text", {"image_path": "r.jpg"})

    assert result.startswith("Permission denied:")
    client.call_tool.assert_not_called()


def test_an_mcp_tool_needing_confirmation_is_refused_without_one(tmp_path, monkeypatch):
    registry = _registry(
        tmp_path, PermissionPolicy(rules={"ocr_extract_text": "require-user-confirmation"})
    )

    client = MagicMock()
    monkeypatch.setattr(registry, "_mcp_tool_owner", {"ocr_extract_text": client})

    result = registry.call("ocr_extract_text", {"image_path": "r.jpg"})

    assert result.startswith("Permission denied:")
    client.call_tool.assert_not_called()


def test_the_same_policy_applies_whether_a_tool_is_builtin_or_mcp(tmp_path, monkeypatch):
    """One policy, one namespace: a deny-by-default policy blocks both
    kinds without either being named."""
    registry = _registry(tmp_path, PermissionPolicy(default="deny"))

    client = MagicMock()
    monkeypatch.setattr(registry, "_mcp_tool_owner", {"ocr_extract_text": client})

    assert registry.call("create", {"path": "x.txt", "type": "file"}).startswith("Permission denied:")
    assert registry.call("ocr_extract_text", {}).startswith("Permission denied:")
    client.call_tool.assert_not_called()


# ── Ordering of the check ────────────────────────────────────────────

def test_a_denied_tool_with_unparseable_arguments_reports_the_refusal(tmp_path):
    """The policy check runs before argument parsing, so a blocked call is
    refused on the grounds that it is blocked rather than reporting a JSON
    error that sends the model off fixing the wrong thing."""
    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "deny"}))

    result = registry.call("create", "{not valid json")

    assert result.startswith("Permission denied:")
    assert "could not parse" not in result


def test_an_unknown_tool_is_refused_under_a_deny_by_default_policy(tmp_path):
    """The check runs before the unknown-tool branch, so the registry does
    not disclose which tools exist to a caller allowed to run none."""
    registry = _registry(tmp_path, PermissionPolicy(default="deny"))

    result = registry.call("no_such_tool", {})

    assert result.startswith("Permission denied:")


def test_an_unknown_tool_still_reports_itself_as_unknown_when_allowed(tmp_path):
    """The pre-existing behaviour must survive: with nothing blocked, an
    unknown name is still a clear "unknown tool", not a permission error."""
    registry = _registry(tmp_path, PermissionPolicy(default="allow"))

    result = registry.call("no_such_tool", {})

    assert "unknown tool" in result


def test_a_non_string_tool_name_is_reported_rather_than_raising(tmp_path):
    """policy.decide() raises on a non-string, but call() is documented
    never to raise, so it must be caught before it escapes."""
    registry = _registry(tmp_path, PermissionPolicy())

    result = registry.call(None, {})

    assert result.startswith("Error:")


# ── Reporting ────────────────────────────────────────────────────────

def test_a_registry_refusal_is_reported_on_the_hook_bus(tmp_path, monkeypatch):
    """A refusal here means someone reached the registry without going
    through the loop, which is exactly the thing worth seeing."""
    manager = HookManager()
    events = []
    manager.on("permission_decision", lambda name, **p: events.append(p))
    monkeypatch.setattr("agent.tool_registry.hooks", manager)

    registry = _registry(tmp_path, PermissionPolicy(rules={"create": "deny"}))
    registry.call("create", {"path": "x.txt", "type": "file"})

    assert len(events) == 1
    assert events[0]["tool_name"] == "create"
    assert events[0]["decision"] == Decision.DENY.value
    assert events[0]["source"] == "rule"
    assert events[0]["allowed"] is False


def test_an_allowed_call_is_not_reported_by_the_registry(tmp_path, monkeypatch):
    """Only refusals. The loop already reports every decision it makes,
    and it only reaches the registry for calls it allowed -- reporting
    allows here too would double-count the ordinary path."""
    manager = HookManager()
    events = []
    manager.on("permission_decision", lambda name, **p: events.append(p))
    monkeypatch.setattr("agent.tool_registry.hooks", manager)

    registry = _registry(tmp_path, PermissionPolicy(default="allow"))
    registry.call("create", {"path": "x.txt", "type": "file"})

    assert events == []
