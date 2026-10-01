"""
Unit tests for the receipt long-term memory store (agent/memory.py).

Covers: dedup by content hash (not filename), running totals, persistence
across separate MemoryStore instances (simulating an app restart), the
startup summary text, and graceful handling of a missing/corrupt file.
"""

import json
import os

from memory import MemoryStore


def _write_file(path, content=b"receipt content"):
    with open(path, "wb") as f:
        f.write(content)
    return str(path)


def test_fresh_store_reports_no_receipts(tmp_path):
    store = MemoryStore(path=str(tmp_path / "memory.json"))
    assert "No receipts processed yet" in store.summary()


def test_is_processed_false_for_new_receipt(tmp_path):
    receipt = _write_file(tmp_path / "r1.jpg")
    store = MemoryStore(path=str(tmp_path / "memory.json"))
    assert store.is_processed(receipt) is False


def test_is_processed_false_for_nonexistent_file(tmp_path):
    store = MemoryStore(path=str(tmp_path / "memory.json"))
    assert store.is_processed(str(tmp_path / "does_not_exist.jpg")) is False


def test_record_then_is_processed_true(tmp_path):
    receipt = _write_file(tmp_path / "r1.jpg")
    store = MemoryStore(path=str(tmp_path / "memory.json"))
    store.record_receipt(receipt, item_count=2, total_value=45.50)
    assert store.is_processed(receipt) is True


def test_dedup_is_by_content_not_filename(tmp_path):
    """A renamed copy of the exact same bytes must still be recognized
    as a duplicate -- this is the whole point of hashing content instead
    of tracking paths."""
    original = _write_file(tmp_path / "Receipt 1.jpg", content=b"identical bytes")
    renamed = _write_file(tmp_path / "totally_different_name.jpg", content=b"identical bytes")

    store = MemoryStore(path=str(tmp_path / "memory.json"))
    store.record_receipt(original, item_count=1, total_value=10.0)

    assert store.is_processed(renamed) is True


def test_different_content_same_filename_not_treated_as_duplicate(tmp_path):
    """The inverse case: two different files that happen to share a name
    pattern (e.g. 'receipt1.txt' vs 'Receipt 1.jpg' in the real app) must
    NOT be treated as the same receipt just because the names look similar."""
    receipt_a = _write_file(tmp_path / "receipt_a.jpg", content=b"content A")
    receipt_b = _write_file(tmp_path / "receipt_b.jpg", content=b"content B")

    store = MemoryStore(path=str(tmp_path / "memory.json"))
    store.record_receipt(receipt_a, item_count=1, total_value=10.0)

    assert store.is_processed(receipt_b) is False


def test_running_totals_accumulate_across_receipts(tmp_path):
    r1 = _write_file(tmp_path / "r1.jpg", content=b"one")
    r2 = _write_file(tmp_path / "r2.jpg", content=b"two")

    store = MemoryStore(path=str(tmp_path / "memory.json"))
    store.record_receipt(r1, item_count=2, total_value=100.0)
    store.record_receipt(r2, item_count=3, total_value=50.0)

    summary = store.summary()
    assert "2 receipt(s) processed" in summary
    assert "5 item(s) tracked" in summary
    assert "150.00" in summary


def test_persists_across_separate_instances(tmp_path):
    """Simulates an app restart: a brand-new MemoryStore instance pointed
    at the same file must see what a previous instance recorded."""
    receipt = _write_file(tmp_path / "r1.jpg")
    memory_path = str(tmp_path / "memory.json")

    first_session = MemoryStore(path=memory_path)
    first_session.record_receipt(receipt, item_count=1, total_value=25.0)

    second_session = MemoryStore(path=memory_path)  # fresh instance, same file
    assert second_session.is_processed(receipt) is True
    assert "1 receipt(s) processed" in second_session.summary()


def test_missing_file_starts_with_empty_state(tmp_path):
    memory_path = str(tmp_path / "does_not_exist_yet.json")
    store = MemoryStore(path=memory_path)
    assert "No receipts processed yet" in store.summary()


def test_corrupt_file_falls_back_to_empty_state_instead_of_crashing(tmp_path):
    memory_path = tmp_path / "memory.json"
    memory_path.write_text("{ this is not valid json !!!", encoding="utf-8")

    store = MemoryStore(path=str(memory_path))  # must not raise
    assert "No receipts processed yet" in store.summary()


def test_record_receipt_writes_valid_json_to_disk(tmp_path):
    receipt = _write_file(tmp_path / "r1.jpg")
    memory_path = tmp_path / "memory.json"

    store = MemoryStore(path=str(memory_path))
    store.record_receipt(receipt, item_count=1, total_value=19.99)

    with open(memory_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # must not raise -- proves valid JSON was written

    assert data["item_count"] == 1
    assert data["total_value"] == 19.99
    assert len(data["processed_receipts"]) == 1


def test_no_leftover_tmp_file_after_write(tmp_path):
    """The atomic-write implementation writes to a .tmp file then renames
    it -- confirms the .tmp file doesn't get left behind."""
    receipt = _write_file(tmp_path / "r1.jpg")
    memory_path = tmp_path / "memory.json"

    store = MemoryStore(path=str(memory_path))
    store.record_receipt(receipt, item_count=1, total_value=10.0)

    assert not os.path.exists(str(memory_path) + ".tmp")
