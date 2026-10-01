import os
import shutil

from tool_registry import ToolRegistry

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")


class TestToolNavigate:
    """Tests for the _navigate tool."""

    def test_navigate_root(self):
        registry = ToolRegistry(workspace_root=WORKSPACE)
        nav_dir = os.path.join(WORKSPACE, "test_nav_dir")
        os.makedirs(nav_dir)
        open(os.path.join(nav_dir, "file.txt"), "w").close()

        result = registry.call("navigate", {"path": "test_nav_dir"})

        assert "file.txt (file)" in result
        shutil.rmtree(nav_dir)

    def test_navigate_empty_dir(self):
        registry = ToolRegistry(workspace_root=WORKSPACE)
        empty_dir = os.path.join(WORKSPACE, "test_empty_dir")
        os.makedirs(empty_dir)

        result = registry.call("navigate", {"path": "test_empty_dir"})

        assert "Empty directory" in result
        shutil.rmtree(empty_dir)

    def test_navigate_nonexistent(self):
        registry = ToolRegistry(workspace_root=WORKSPACE)

        result = registry.call("navigate", {"path": "no_such_dir"})

        assert "Error" in result

    def test_navigate_nested(self):
        registry = ToolRegistry(workspace_root=WORKSPACE)
        nested = os.path.join(WORKSPACE, "test_nested", "sub")
        os.makedirs(nested)
        open(os.path.join(nested, "data.csv"), "w").close()

        result_parent = registry.call("navigate", {"path": "test_nested"})
        assert "sub/ (folder)" in result_parent

        result_sub = registry.call("navigate", {"path": "test_nested/sub"})
        assert "data.csv (file)" in result_sub

        shutil.rmtree(os.path.join(WORKSPACE, "test_nested"))
