"""
Unit tests for LTM (Long-Term Memory) module.

Run from the repo root:
    pytest tests/unit_tests/test_ltm.py
"""

import pytest

from agent.ltm import Memory


@pytest.fixture
def memory(tmp_path):
    db_path = str(tmp_path / "test_chroma")
    mem = Memory(db_path=db_path)
    yield mem


def test_remember_and_recall(memory):
    memory.remember("user", "prefers", "Excel over CSV")
    result = memory.recall("Excel")
    assert "Excel over CSV" in result
    assert "user" in result
    assert "prefers" in result
    assert "[" in result


def test_recall_semantic(memory):
    memory.remember("user", "prefers", "Excel over CSV")
    result = memory.recall("spreadsheet software")
    assert "Excel over CSV" in result


def test_recall_all_returns_all_facts(memory):
    memory.remember("user", "prefers", "dark mode")
    memory.remember("user", "works from", "home")
    memory.remember("project", "uses", "SQLite")
    result = memory.recall_all()
    assert "Known Facts:" in result
    assert "dark mode" in result
    assert "works from" in result
    assert "SQLite" in result
    assert result.count("[") == 3


def test_recall_all_empty_when_no_facts(memory):
    result = memory.recall_all()
    assert result == ""


def test_recall_all_filters_by_subject(memory):
    """subject='user' returns only user facts; no subject returns all.

    Also guards the Chroma `where` clause: a plain two-key dict
    ({"active": True, "subject": ...}) raises ValueError, so this test
    fails loudly if the $and form is ever simplified away.
    """
    memory.remember("user", "prefers", "metric units")
    memory.remember("laptop", "brand", "Dell")

    only_user = memory.recall_all(subject="user")
    assert "Known Facts:" in only_user
    assert "metric units" in only_user
    assert "Dell" not in only_user
    assert only_user.count("[") == 1

    everything = memory.recall_all()
    assert "metric units" in everything
    assert "Dell" in everything
    assert everything.count("[") == 2


def test_recall_all_subject_with_no_matches_returns_empty(memory):
    memory.remember("laptop", "brand", "Dell")
    assert memory.recall_all(subject="user") == ""


def test_recall_all_summaries(memory):
    memory.save_summary("Discussed project setup and tool choices.", date="2025-01-15", key_topics="project setup, tools")
    memory.save_summary("Reviewed inventory and resolved duplicates.", date="2025-01-20", key_topics="inventory, duplicates")
    result = memory.recall_all_summaries()
    assert "Stored Summaries:" in result
    assert "project setup" in result
    assert "inventory" in result
    assert "2025-01-15" in result
    assert "2025-01-20" in result


def test_recall_all_summaries_empty(memory):
    result = memory.recall_all_summaries()
    assert result == ""


def test_forget_excludes_from_recall(memory):
    entry_id = _remember_and_get_id(memory, "user", "prefers", "dark mode")
    result = memory.forget(entry_id)
    assert "Forgotten" in result
    recall_result = memory.recall("dark mode")
    assert entry_id not in recall_result


def test_forget_soft_delete(memory, tmp_path):
    entry_id = _remember_and_get_id(memory, "user", "prefers", "dark mode")
    memory.forget(entry_id)

    client = __import__("chromadb").PersistentClient(path=str(tmp_path / "test_chroma"))
    collection = client.get_collection("long_term_memory")
    raw = collection.get(ids=[entry_id], include=["metadatas"])
    assert raw["ids"]
    assert raw["metadatas"][0]["active"] is False


def test_remember_returns_confirmation(memory):
    result = memory.remember("subject", "predicate", "value")
    assert "Remembered" in result
    assert "subject" in result
    assert "predicate" in result
    assert "value" in result


def test_persistence(tmp_path):
    db_path = str(tmp_path / "persist_chroma")
    mem1 = Memory(db_path=db_path)
    mem1.remember("user", "name", "Alice")

    mem2 = Memory(db_path=db_path)
    result = mem2.recall("Alice")
    assert "Alice" in result


def _remember_and_get_id(memory, subject, predicate, value):
    result = memory.remember(subject, predicate, value)
    start = result.find("[") + 1
    end = result.find("]")
    return result[start:end]


def test_remember_id_in_confirmation(memory):
    result = memory.remember("subject", "predicate", "value")
    assert "[" in result
    assert "]" in result
