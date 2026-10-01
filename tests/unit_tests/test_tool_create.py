import os
import shutil

from tool_registry import ToolRegistry

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")


class TestToolCreate:
    """Tests for the _create tool: file, folder, and nested file creation."""

    def test_create_file(self):
        registry = ToolRegistry(workspace_root=WORKSPACE)
        test_path = "test_hello.txt"

        result = registry.call("create", {"path": test_path, "type": "file"})

        assert "Created file:" in result
        assert os.path.isfile(os.path.join(WORKSPACE, test_path))

        os.remove(os.path.join(WORKSPACE, test_path))

    def test_create_folder(self):
        registry = ToolRegistry(workspace_root=WORKSPACE)
        test_path = "test_my_folder"

        result = registry.call("create", {"path": test_path, "type": "folder"})

        assert "Created folder:" in result
        assert os.path.isdir(os.path.join(WORKSPACE, test_path))

        shutil.rmtree(os.path.join(WORKSPACE, test_path))

    def test_create_nested_file(self):
        registry = ToolRegistry(workspace_root=WORKSPACE)
        test_path = "test_docs/test_notes.txt"

        result = registry.call("create", {"path": test_path, "type": "file"})

        assert "Created file:" in result
        assert os.path.isfile(os.path.join(WORKSPACE, test_path))

        shutil.rmtree(os.path.join(WORKSPACE, "test_docs"))
