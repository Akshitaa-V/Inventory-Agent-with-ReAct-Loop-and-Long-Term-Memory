"""
memory.py — small persistent JSON-backed memory for the inventory agent.

Survives restarts. Tracks which receipts have already been processed
(keyed by file content hash, not filename, so a renamed-but-identical
receipt still gets recognized as a duplicate) plus a running item count
and total value the agent can greet the user with on a new session.

Trigger contract (matches the handout's requirement):
  - READ once at startup (MemoryStore.__init__ loads the file).
  - WRITE after every successful inventory update (record_receipt()).
"""

import hashlib
import json
import os

from agent.hooks import hooks, timed

DEFAULT_MEMORY_PATH = "memory.json"


class MemoryStore:
    def __init__(self, path: str = DEFAULT_MEMORY_PATH):
        self._path = os.path.abspath(path)
        self._data = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self._path):
            return {"processed_receipts": {}, "item_count": 0, "total_value": 0.0}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable memory file -- start fresh rather than
            # crashing the whole agent over a broken cache file.
            return {"processed_receipts": {}, "item_count": 0, "total_value": 0.0}

        data.setdefault("processed_receipts", {})
        data.setdefault("item_count", 0)
        data.setdefault("total_value", 0.0)
        return data

    def _save(self) -> None:
        # Write to a temp file then atomically replace, so a crash mid-write
        # can't leave memory.json truncated/corrupted.
        tmp_path = self._path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp_path, self._path)

    @staticmethod
    def _hash_file(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_processed(self, receipt_path: str) -> bool:
        """True if this exact file's content has already been recorded,
        regardless of what it's currently named or where it lives."""
        with timed(hooks, "memory_op", operation="is_processed"):
            if not os.path.isfile(receipt_path):
                return False
            return self._hash_file(receipt_path) in self._data["processed_receipts"]

    def record_receipt(self, receipt_path: str, item_count: int = 1, total_value: float = 0.0) -> None:
        """Marks a receipt as processed and updates running totals.
        Safe to call once per receipt -- calling it again on the same file
        content will double-count, so callers should check is_processed()
        first (the agent is instructed to do this)."""
        with timed(hooks, "memory_op", operation="record_receipt"):
            file_hash = self._hash_file(receipt_path)
            self._data["processed_receipts"][file_hash] = {
                "source_path": receipt_path,
            }
            self._data["item_count"] += item_count
            self._data["total_value"] += total_value
            self._save()

    def summary(self) -> str:
        """A short, human-readable greeting for session startup."""
        count = len(self._data["processed_receipts"])
        if count == 0:
            return "No receipts processed yet. Ready when you are."
        items = self._data["item_count"]
        total = self._data["total_value"]
        return (
            f"{count} receipt(s) processed so far, {items} item(s) tracked, "
            f"total value {total:.2f}."
        )