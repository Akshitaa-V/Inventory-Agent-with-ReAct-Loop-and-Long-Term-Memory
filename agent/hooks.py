"""
agent/hooks.py -- reusable lifecycle hooks for instrumentation.

This is the decoupled mechanism the Week 3 handout asks for: a single,
generic pub/sub system that anything (metrics, logging, tracing) can
subscribe to, without the agent's core code (loop.py, tool_registry.py,
ltm.py, llm_client.py) needing to know or care who's listening.

The core code only ever does two things:
  1. `hooks.on(event_name, callback)` -- register interest (done once,
     at startup, by whatever subsystem cares -- e.g. metrics.py).
  2. `with timed(hooks, "llm_call", **labels): ...` -- wrap an existing
     block of code to emit start/end events around it automatically,
     including duration and success/failure, with zero changes to the
     wrapped code itself.

No component ever imports Prometheus, logging, or anything else directly
-- they only ever touch this module. That's what makes it decoupled:
swapping "log to Prometheus" for "log to a file" later means changing
metrics.py, not touching loop.py or tool_registry.py at all.
"""

import time
from contextlib import contextmanager


class HookManager:
    """A minimal synchronous pub/sub system. Deliberately dependency-free
    and framework-agnostic -- this is the "equivalent decoupled
    instrumentation mechanism" the handout allows as an alternative to a
    specific library."""

    def __init__(self):
        self._listeners = {}

    def on(self, event_name: str, callback) -> None:
        """Register `callback(event_name, **payload)` to run whenever
        `event_name` fires. Multiple listeners per event are supported --
        e.g. metrics.py and a future logging module can both listen to
        the same "tool_call_end" event independently."""
        self._listeners.setdefault(event_name, []).append(callback)

    def emit(self, event_name: str, **payload) -> None:
        """Fire `event_name` to every registered listener. A listener
        raising an exception must never break the agent -- instrumentation
        is observability, not a load-bearing part of the request path --
        so failures here are swallowed (not re-raised), after printing a
        one-line warning so a broken listener is still visible."""
        for callback in self._listeners.get(event_name, []):
            try:
                callback(event_name, **payload)
            except Exception as exc:
                print(f"(hook listener for '{event_name}' failed: {exc})")


# Module-level singleton -- one shared hook manager for the whole process,
# so any file can just `from agent.hooks import hooks` and use it.
hooks = HookManager()


@contextmanager
def timed(hook_manager: HookManager, event_prefix: str, **labels):
    """Wraps an existing block of code with automatic start/end events,
    duration timing, and success/failure detection -- without requiring
    any changes to the code inside the `with` block.

    Emits:
      - "{event_prefix}_start" before the block runs
      - "{event_prefix}_end" after it finishes, with:
          duration_seconds: float
          success: bool
          error: str | None  -- the exception's message, if it failed
          ...plus anything added to the yielded context dict (see below)

    Usage (drop this around ANY existing call, unchanged):
        with timed(hooks, "tool_call", tool_name=name):
            result = registry.call(name, args)   # existing code, untouched

    Some data (like LLM token counts) is only known AFTER the wrapped
    call returns. `timed` yields a small mutable dict for exactly this --
    add to it inside the block, and its contents are merged into the
    single end-event payload. This avoids ever needing a second, manual
    emit() call alongside `timed`, which would double-count the event:
        with timed(hooks, "llm_call") as ctx:
            response = llm.get_next_step(...)   # existing code, untouched
            ctx["input_tokens"] = response.usage.input_tokens
            ctx["output_tokens"] = response.usage.output_tokens

    Some code (like loop.py's tool-call handling) already catches its own
    exceptions and turns them into a result string, rather than letting
    them propagate. In that case `timed`'s automatic exception-based
    failure detection never triggers -- so the context dict can override
    it explicitly:
        with timed(hooks, "tool_call", tool_name=name) as ctx:
            try:
                observation = registry.call(name, args)   # existing code, untouched
            except Exception as exc:
                observation = f"Error: {exc}"
                ctx["success"] = False
                ctx["error"] = str(exc)

    A failure inside the block is still raised to the caller as normal --
    this context manager observes, it never swallows or changes behavior.
    """
    hook_manager.emit(f"{event_prefix}_start", **labels)
    start = time.monotonic()
    success = True
    error = None
    context = {}
    try:
        yield context
    except Exception as exc:
        success = False
        error = str(exc)
        raise
    finally:
        duration = time.monotonic() - start
        # Build the payload with labels/context able to override the
        # auto-detected duration/success/error -- e.g. when the caller
        # already caught its own exception and wants to report failure
        # without letting timed() see the exception itself.
        payload = {"duration_seconds": duration, "success": success, "error": error}
        payload.update(labels)
        payload.update(context)
        hook_manager.emit(f"{event_prefix}_end", **payload)
