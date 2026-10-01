"""Startup-path checks: the preflight that turns opaque container
permission failures into messages naming the path and the fix.

Three things are pinned here because each has silently broken before:
  - check_state_paths passes on a writable tree and fails (actionably)
    on a non-writable one;
  - Memory re-raises ChromaDB's pathless "Permission denied" with the
    db_path and the docker compose down -v remedy attached;
  - METRICS_PORT stays hardcoded: prometheus.yml reads plain YAML and
    cannot follow an env var, so a configurable port is a knob that
    only lets main.py, docker-compose.yml and prometheus.yml drift.
"""

import os
from types import SimpleNamespace

import pytest

import agent.ltm as ltm
from agent.main import check_state_paths


def _config(tmp_path, **overrides):
    cfg = SimpleNamespace(
        workspace_root=str(tmp_path / "workspace"),
        ltm_db_path=str(tmp_path / "data" / "chroma"),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_writable_paths_pass_preflight(tmp_path):
    # Fresh checkout: the leaves do not exist yet, only their writable
    # ancestors are probed -- the libraries create the leaves themselves.
    check_state_paths(_config(tmp_path))


def test_unwritable_path_names_path_and_fix(tmp_path, monkeypatch):
    target = str(tmp_path)
    real_access = os.access

    def fake_access(path, mode):
        if os.path.normpath(str(path)) == os.path.normpath(target):
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", fake_access)

    with pytest.raises(RuntimeError) as excinfo:
        check_state_paths(_config(tmp_path))

    message = str(excinfo.value)
    assert target in message                      # says WHERE
    assert "uid" in message                       # says WHO it runs as
    assert "docker compose down -v" in message    # says HOW to fix it
    assert "not a missing database" in message    # says what it is NOT


def test_workspace_and_db_path_both_checked(tmp_path, monkeypatch):
    # Only the workspace is unwritable -- the db path alone passing must
    # not let the workspace failure slip through. Both leaves must exist:
    # a missing leaf is probed at its (writable) parent, which is the
    # correct behavior, not a failure.
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    (tmp_path / "data").mkdir()
    real_access = os.access

    def fake_access(path, mode):
        if os.path.normpath(str(path)) == os.path.normpath(workspace):
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", fake_access)

    with pytest.raises(RuntimeError, match="workspace_root"):
        check_state_paths(_config(tmp_path))


def test_memory_reraises_chromadb_failure_with_path(tmp_path, monkeypatch):
    # Chroma's Rust core raises a bare OSError/RuntimeError with no path;
    # Memory must attach the path and the remedy before it escapes.
    db_path = str(tmp_path / "chroma")

    def boom(path=None, *args, **kwargs):
        raise RuntimeError("Permission denied (os error 13)")

    monkeypatch.setattr(ltm.chromadb, "PersistentClient", boom)

    with pytest.raises(RuntimeError) as excinfo:
        ltm.Memory(db_path=db_path)

    message = str(excinfo.value)
    assert db_path in message
    assert "docker compose down -v" in message


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_metrics_port_is_hardcoded_not_env_configurable():
    # main.py must read a literal port, and .env_example must not offer
    # METRICS_PORT at all: prometheus.yml cannot follow an env var, so
    # exposing the knob only invites the three copies to drift apart and
    # turns a typo into a silently empty dashboard.
    with open(os.path.join(_repo_root(), "agent", "main.py"), encoding="utf-8") as f:
        main_src = f.read()
    assert "METRICS_PORT" not in main_src
    assert "start_metrics_server(port=9000)" in main_src

    with open(os.path.join(_repo_root(), ".env_example"), encoding="utf-8") as f:
        env_example = f.read()
    assert "METRICS_PORT" not in env_example
