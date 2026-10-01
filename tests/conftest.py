import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))


# Load INNKUBE_TOKEN (and anything else) from a .env file in the repository
# root if it is not already exported. The end-to-end tests need a real token
# and otherwise skip themselves; .env is gitignored and is the mechanism the
# README already documents for docker compose, so this makes `pytest
# tests/e2e_tests/` work without having to remember to export the variable
# first. An exported value always wins over the file.
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()


@pytest.fixture
def shipped_policy():
    """The permission policy from config_example.json -- the tracked,
    canonical template. config.json is listed in .gitignore but is in
    fact tracked too, so it can carry machine-specific edits; reading
    config_example.json keeps this fixture independent of those.

    End-to-end tests run under the real policy rather than the all-allow
    default so that a mistake in the shipped rules actually surfaces in
    the suite. They pair it with permissions.auto_approve, because they
    have no human to answer a confirmation prompt and must never block
    on stdin. auto_approve cannot loosen a deny rule, so a tool denied
    by the shipped policy stays denied even here.
    """
    from agent.permissions import PermissionPolicy

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(repo_root, "config_example.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return PermissionPolicy.from_config(raw.get("permissions"))
