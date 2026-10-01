"""
End-to-end test: asks the real agent to create a receipt and build an inventory.

Requires:
    - INNKUBE_TOKEN environment variable set
    - A valid config.json at the repo root with LLM settings

Run from the repo root:
    pytest tests/e2e_tests/test_add_receipt_and_inventory.py -v
"""

import os
import shutil

import pytest

from config import Config, load_config
from context import Context
from llm_client import LLMClient
from loop import react_step
from permissions import auto_approve
from main import SYSTEM_PROMPT
from tool_registry import ToolRegistry

RECEIPT_PROMPT = (
    "Create a new empty file at receipts/receipt1000.txt using the 'create' "
    "tool. Then, in a separate step, use the 'modify' tool to write exactly "
    "this content into it:\n"
    "Best Buy\n"
    "Dell XPS 13 Laptop\n"
    "Serial No: DXP-13-45927\n"
    "Condition: New\n"
    "EUR 1,199.99\n"
    "Date: 15 Aug 2026\n\n"
    "Do this before anything else, and do not process, read, or reference "
    "any other files in the receipts folder for this task.\n\n"
    "Once receipt1000.txt has that content, add it to inventory.md. If "
    "inventory.md does not exist yet, create it first with a header row "
    "before appending the entry."
)

# Repo root is two levels up from this test file
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_receipt_to_inventory(tmp_path, shipped_policy):
    """Agent creates receipt1000.txt and adds it to inventory.md."""
    config_path = os.path.join(REPO_ROOT, "config.json")
    if not os.path.exists(config_path):
        pytest.skip("config.json not found at repo root — cannot run E2E test")

    if not os.environ.get("INNKUBE_TOKEN"):
        pytest.skip("INNKUBE_TOKEN environment variable not set — skipping E2E test")

    # Use production config loader, override workspace_root
    real_config = load_config(config_path)

    # Set up workspace with existing sample receipts
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipts_dir = workspace / "receipts"
    receipts_dir.mkdir()

    samples_dir = os.path.join(REPO_ROOT, "samples", "receipts")
    if os.path.isdir(samples_dir):
        for fname in os.listdir(samples_dir):
            src = os.path.join(samples_dir, fname)
            if os.path.isfile(src):
                shutil.copy(src, receipts_dir / fname)

    # Build a config pointing to our temp workspace
    config = Config(
        model=real_config.model,
        temperature=real_config.temperature,
        base_url=real_config.base_url,
        endpoint=real_config.endpoint,
        max_iterations=real_config.max_iterations,
        workspace_root=str(workspace),
        ltm_db_path=real_config.ltm_db_path,
        api_key=real_config.api_key,
        permissions=real_config.permissions,
    )

    llm = LLMClient(config)
    context = Context(SYSTEM_PROMPT)
    tools = ToolRegistry(workspace_root=str(workspace))

    context.add_user_message(RECEIPT_PROMPT)
    answer = react_step(
        llm, context, tools, config.max_iterations,
        policy=shipped_policy, confirm=auto_approve,
    )

    # --- Assertions ---

    # 1. receipt1000.txt was created
    receipt_path = receipts_dir / "receipt1000.txt"
    assert receipt_path.exists(), f"receipt1000.txt was not created. Agent said: {answer}"

    # 2. receipt1000.txt contains the expected content
    receipt_content = receipt_path.read_text(encoding="utf-8")
    assert "Best Buy" in receipt_content, "Receipt missing store name"
    assert "Dell XPS 13" in receipt_content, "Receipt missing item name"
    assert "DXP-13-45927" in receipt_content, "Receipt missing serial number"
    assert "1,199.99" in receipt_content, "Receipt missing price"
    assert "15 Aug 2026" in receipt_content, "Receipt missing date"
    assert "New" in receipt_content, "Receipt missing condition"

    # 3. inventory.md was created
    inventory_path = workspace / "inventory.md"
    assert inventory_path.exists(), f"inventory.md was not created. Agent said: {answer}"

    # 4. inventory.md references the Dell XPS 13
    inventory_content = inventory_path.read_text(encoding="utf-8")
    assert "Dell XPS 13" in inventory_content or "DXP-13-45927" in inventory_content, (
        f"inventory.md does not reference the new item. Agent said: {answer}"
    )

    # Accept either "1,199.99" or "1199.99" — the model may or may not
    # include a thousands separator; what matters is the numeric value,
    # not the exact formatting, which can vary by model.
    price_found = "1,199.99" in inventory_content or "1199.99" in inventory_content
    assert price_found, (
        f"inventory.md missing price (checked both '1,199.99' and '1199.99'). "
        f"Agent said: {answer}"
    )