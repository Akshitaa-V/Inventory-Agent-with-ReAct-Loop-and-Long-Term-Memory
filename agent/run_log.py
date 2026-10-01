"""
agent/run_log.py -- structured JSON-lines log of every lifecycle event.

Week 3 (handout 8.4) asks for the agent's monitoring data to include a
readable log, not just Prometheus counters. This module is the file-based
listener side of the same hooks bus metrics.py already uses -- it adds a
second, independent subscriber to every event, so nothing in loop.py,
tool_registry.py or ltm.py needs to change or even know this exists.

Each event becomes exactly one JSON line, in the order it fired, with an
ISO-8601 UTC timestamp added. That shape -- one line per event, appended
as it happens -- is what makes the file greppable and safe to tail live
while the agent runs, and what makes `python -m agent.run_log <run_id>`
below able to reconstruct one run's full story without loading the whole
file into memory.
"""

import json
import os
import sys
import threading
from datetime import datetime, timezone

from agent.hooks import HookManager

# Every event name the loop, registry and memory layer emit. Listed
# explicitly, not "everything hooks.emit ever sees", so a future typo'd
# event name fails loudly (nothing recorded) rather than being logged
# under whatever name someone accidentally shipped.
_LOGGED_EVENTS = (
    "run_start", "run_end",
    "llm_call_start", "llm_call_end",
    "tool_call_start", "tool_call_end",
    "memory_op_start", "memory_op_end",
    "permission_decision",
)

# Cheap insurance against interleaved partial writes if two runs ever
# emit concurrently -- the entire point of this module is not to lose
# or corrupt events, so a lock around each individual write is worth it.
_write_lock = threading.Lock()


def init_run_log(hook_manager: HookManager, path: str) -> None:
    """Subscribes a JSON-lines writer at `path` to every event in
    _LOGGED_EVENTS.

    Mirrors metrics.init_metrics(): called once at startup, after which
    core code does nothing differently to be logged. A listener that
    cannot open its file crashes at startup, in the same spirit as a bad
    permissions config -- observability that silently never wrote
    anything would be worse than one that refuses to start.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path, "a", encoding="utf-8")

    def _write(event_name: str, **payload) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            **payload,
        }
        line = json.dumps(record, default=str)
        with _write_lock:
            handle.write(line + "\n")
            handle.flush()

    for event_name in _LOGGED_EVENTS:
        hook_manager.on(event_name, _write)


def _print_run(path: str, run_id: str) -> None:
    """Prints every logged line for one run_id, in file order."""
    found = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("run_id") == run_id:
                    found = True
                    print(json.dumps(record, indent=2))
    except FileNotFoundError:
        print(f"No log file at {path!r} -- has the agent been run yet?")
        return

    if not found:
        print(f"No entries found for run_id {run_id!r} in {path!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agent.run_log <run_id> [path]")
        sys.exit(1)
    _print_run(sys.argv[2] if len(sys.argv) > 2 else "data/run_log.jsonl", sys.argv[1])