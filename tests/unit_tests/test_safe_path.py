"""
Unit tests for ToolRegistry's case-insensitive path resolution.

Windows filesystems are case-insensitive, so 'receipt 1.jpg' silently
matched 'Receipt 1.jpg' in every local test on every team member's
machine -- but the Docker image runs on Linux, which is case-sensitive,
so the exact same request would fail there with no local test ever
catching it. These tests run on whatever filesystem the test suite
executes on and exercise the resolution logic directly, so they catch
this class of bug regardless of which OS runs them.
"""

import pytest

from tool_registry import ToolRegistry


@pytest.fixture
def registry(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "receipts").mkdir()
    (workspace / "receipts" / "Receipt 1.jpg").write_text("fake image content", encoding="utf-8")
    (workspace / "inventory.md").write_text("# Inventory\n", encoding="utf-8")

    return ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers={"recall": {"enabled": False}, "ocr": {"enabled": False}},
    )


def test_exact_case_match_still_works(registry):
    result = registry.call("read", {"path": "receipts/Receipt 1.jpg"})
    assert "Error" not in result
    assert "fake image content" in result


def test_wrong_case_filename_still_resolves(registry):
    """The core bug: a request differing only in case must still find
    the real file, instead of a false 'does not exist' error."""
    result = registry.call("read", {"path": "receipts/receipt 1.jpg"})
    assert "Error" not in result
    assert "fake image content" in result


def test_wrong_case_folder_and_filename_both_resolve(registry):
    """Case mismatches can occur at any path component, not just the
    filename -- both the folder and file names are wrong-case here."""
    result = registry.call("read", {"path": "RECEIPTS/RECEIPT 1.JPG"})
    assert "Error" not in result
    assert "fake image content" in result


def test_genuinely_missing_file_still_reports_not_found(registry):
    """The fix must not paper over real 404s -- a file that truly
    doesn't exist, under any casing, must still fail clearly."""
    result = registry.call("read", {"path": "receipts/this_file_was_never_created.jpg"})
    assert "Error" in result
    assert "does not exist" in result


def test_case_insensitive_resolution_does_not_bypass_workspace_sandbox(registry):
    """The fallback walks path components looking for case-insensitive
    matches -- it must never be tricked into escaping the workspace root."""
    result = registry.call("read", {"path": "../../../etc/passwd"})
    assert "Error" in result
    assert "escapes the workspace" in result


def test_wrong_case_works_for_write_operations_too(registry):
    """modify() also goes through _safe_path -- confirm the fix applies
    there too, not just to read-only lookups."""
    result = registry.call("modify", {"path": "INVENTORY.MD", "content": "New line"})
    assert "Error" not in result
