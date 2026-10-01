"""
Long-Term Memory (LTM) — ChromaDB-backed persistent store.

Two collections:
  - long_term_memory: (subject, predicate, value) triples for general facts.
  - conversation_summaries: session summaries saved at exit.
"""

import os
from datetime import datetime, timezone
from uuid import uuid4

import chromadb

from agent.hooks import hooks, timed


class Memory:
    def __init__(self, db_path: str = "data/chroma"):
        # ChromaDB's PersistentClient manages db_path as a directory it
        # creates and controls itself. If an old file happens to already
        # exist at that exact path -- e.g. a leftover data/memory.db from
        # before this project switched from SQLite to ChromaDB -- Chroma's
        # Rust backend fails with an opaque "Cannot create a file when
        # that file already exists" error instead of a clear message.
        # Move any such file out of the way automatically rather than
        # crashing or silently deleting someone's old data.
        if os.path.isfile(db_path):
            backup_path = db_path + ".pre-chromadb-backup"
            os.rename(db_path, backup_path)
            print(
                f"Note: found an old file at '{db_path}' (from a previous "
                f"memory system) -- moved it to '{backup_path}' so the "
                f"ChromaDB store could be created fresh."
            )

        # Chroma's Rust backend reports a failed open as an opaque
        # "Permission denied (os error 13)" with no path attached, which
        # reads like a missing database when it never is. Re-raise the
        # same failure with the path and the actual fix attached, so the
        # first line anyone sees is actionable.
        try:
            self._client = chromadb.PersistentClient(path=db_path)
            self._collection = self._client.get_or_create_collection(
                name="long_term_memory"
            )
            self._summaries = self._client.get_or_create_collection(
                name="conversation_summaries"
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Could not open the memory database at '{db_path}' "
                f"({type(exc).__name__}: {exc}).\n"
                "This is a permissions problem on the data directory, not "
                "a missing database -- it is created automatically.\n"
                "  * Docker/Podman: use the named volume from "
                "docker-compose.yml; do not bind-mount ./data.\n"
                "  * Stale volume from an older image: "
                "'docker compose down -v' resets it.\n"
                "  * Rootless Podman: add --userns=keep-id; "
                "on SELinux add :Z to the mount."
            ) from exc

    def remember(self, subject: str, predicate: str, value: str) -> str:
        with timed(hooks, "memory_op", operation="remember"):
            entry_id = str(uuid4())
            document = f"{subject} {predicate}: {value}"
            metadata = {
                "subject": subject,
                "predicate": predicate,
                "value": value,
                "active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._collection.add(ids=[entry_id], documents=[document], metadatas=[metadata])
            return f"Remembered [{entry_id}]: {subject} | {predicate} | {value}"

    def recall(self, query: str) -> str:
        with timed(hooks, "memory_op", operation="recall"):
            results = self._collection.query(
                query_texts=[query], where={"active": True}
            )

            if not results["ids"] or not results["ids"][0]:
                return "No matching facts found."

            lines = []
            for i, entry_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                lines.append(f"[{entry_id}] {i + 1}. {meta['subject']} | {meta['predicate']} | {meta['value']}")
            return "Matching facts:\n" + "\n".join(lines)

    def recall_all(self, subject: str = None) -> str:
        """Return active facts, optionally narrowed to one subject.

        The CLI calls this with subject='user' at startup so only user
        facts get folded into the system prompt. Sub-agents
        (include_memory) and the dashboard pass no subject and receive
        the whole store.

        Chroma requires exactly one top-level key in `where`, so the
        two-condition form has to go through `$and` -- a plain
        {"active": True, "subject": subject} raises ValueError.
        """
        with timed(hooks, "memory_op", operation="recall_all"):
            where = {"active": True}
            if subject is not None:
                where = {"$and": [{"active": True}, {"subject": subject}]}
            results = self._collection.get(where=where)

            if not results["ids"]:
                return ""

            lines = []
            for i, entry_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                lines.append(f"[{entry_id}] {i + 1}. {meta['subject']} | {meta['predicate']} | {meta['value']}")
            return "Known Facts:\n" + "\n".join(lines)

    def forget(self, id: str) -> str:
        with timed(hooks, "memory_op", operation="forget") as ctx:
            existing = self._collection.get(ids=[id], include=["metadatas"])
            if not existing["ids"]:
                ctx["success"] = False
                ctx["error"] = f"fact '{id}' not found"
                return f"Error: fact with id '{id}' not found."

            meta = existing["metadatas"][0]
            meta["active"] = False
            self._collection.update(ids=[id], metadatas=[meta])
            return f"Forgotten: {meta['subject']} | {meta['predicate']} | {meta['value']}"

    # ── Conversation summaries ──────────────────────────────────────

    def save_summary(self, summary: str, date: str, key_topics: str) -> str:
        with timed(hooks, "memory_op", operation="save_summary"):
            entry_id = str(uuid4())
            document = f"Summary ({date}): {summary}"
            metadata = {
                "date": date,
                "key_topics": key_topics,
                "active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._summaries.add(ids=[entry_id], documents=[document], metadatas=[metadata])
            return entry_id

    def recall_summaries(self, query: str) -> str:
        with timed(hooks, "memory_op", operation="recall_summaries"):
            results = self._summaries.query(
                query_texts=[query], where={"active": True}, n_results=5
            )
            if not results["ids"] or not results["ids"][0]:
                return "No matching conversation summaries found."

            lines = []
            for i, entry_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i]
                lines.append(f"[{entry_id}] {doc}")
            return "Matching summaries:\n" + "\n".join(lines)

    def recall_all_summaries(self) -> str:
        """Return all stored session summaries (full text)."""
        with timed(hooks, "memory_op", operation="recall_all_summaries"):
            results = self._summaries.get(where={"active": True})

            if not results["ids"]:
                return ""

            lines = []
            for i, entry_id in enumerate(results["ids"]):
                doc = results["documents"][i]
                lines.append(f"[{entry_id}] {doc}")
            return "Stored Summaries:\n" + "\n".join(lines)
