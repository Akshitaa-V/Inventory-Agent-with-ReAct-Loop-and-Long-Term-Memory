"""
CLI entrypoint for the Common Inventory Agent.

Run from the repository root:
    python -m agent.main
"""

import os
import shutil
import sys
from datetime import datetime, timezone

from agent.config import load_config
from agent.context import Context
from agent.ltm import Memory
from agent.llm_client import LLMClient
from agent.cli_format import pretty_print_tables
from agent.tool_registry import ToolRegistry
from agent.hooks import hooks
from agent.agents import AgentCatalog, AgentSpec
from agent.permissions import PermissionPolicy
from agent.runner import AgentRunner
from agent.metrics import init_metrics, start_metrics_server
from agent.run_log import init_run_log

SYSTEM_PROMPT = """You are a general-purpose inventory agent. Your job is to help the user build and maintain a record of items they own -- at home, in an office, medical equipment, or any other context -- so the record can support an insurance claim or other formal proof of ownership.

## Available Tools
- File ops: create, navigate, search, read, read_many, modify
- Memory: remember_fact, recall_fact, recall_summaries, recall_all_summaries
- No delete tool. Never overwrite existing content; only append via 'modify'.

## Item Record Format
For every item, capture: name/description, category (electronics, furniture, medical, office, jewelry, appliance, other), purchase price + currency, purchase date, store/vendor, serial/model number, condition, location, source. Note missing fields as 'not provided' rather than guessing. If inventory.md does not exist yet, create it first with a header row before appending entries. For items described directly in chat (not from a receipt file), read the existing inventory first to avoid duplicates.

## Receipt Processing
**OCR:** For images/scanned PDFs, use ocr_extract_text or ocr_extract_text_from_pdf before extracting items.

**Quantities:** Record the FULL LINE TOTAL (qty x unit price) as purchase price, not unit price. Note quantity in item name (e.g. 'Console (Qty: 5)').

**Dedup:** Receipts are deduplicated by content hash, not filename. Before adding, always call check_receipt_processed on the exact file path. If already processed, search inventory.md for that receipt's entries and show them. Only call mark_receipt_processed after inventory.md update succeeds.

**View vs Add:** If user just wants to VIEW a receipt (not add it), run OCR and show content regardless of processing status. Do not call check_receipt_processed for viewing requests.

## File Operations
- Use 'read_many' instead of multiple 'read' calls for same folder
- If 'create' fails due to name conflict, report it and ask user how to proceed
- Minimize tool calls: each costs a full LLM round-trip

## Context Compaction
When you see '[Summary of earlier conversation: ...]' in your messages, older conversation was compressed to save tokens. The summary contains key topics and decisions from earlier. Treat it as background context and continue with the current task. Do not re-summarize or ask about it.

## Long-Term Memory
You have persistent memory across sessions for facts about items, places, people, preferences, etc.

**Remember a fact when:** User states a preference, corrects you with a generalizable rule, or you learn something actionable about any subject.

**Recall facts when:** Before making assumptions, when user asks about past actions, or when previous context would help.

**Recall summaries when:** User asks about past conversations, or you need to understand what topics have been covered.

Only user facts (subject='user') are pre-loaded at startup. Use recall_fact for other subjects on demand. Do NOT use remember_fact/recall_fact for receipts -- use check_receipt_processed/mark_receipt_processed instead."""




# The end-of-session summariser, as an agent like any other rather than a
# hand-rolled LLM call. Its deny-everything policy makes the tool view
# expose no schemas, which makes llm_client omit "tools" from the request
# entirely -- the same single-shot request this used to build by hand. One
# iteration is all it can take, and with no tools available the model has
# nothing to do but answer.
#
# Routing it through the runner is also what fixed two gaps: its LLM call
# used to reach the metrics with no model label and no run identity,
# because it was assembled outside react_step. It now gets both for the
# same reason every other call does.
SUMMARY_SPEC = AgentSpec(
    role="session-summariser",
    description="Summarizes a finished session for long-term memory.",
    instructions=(
        "You summarize agent sessions for long-term memory. Given a "
        "conversation, reply with a 3-5 sentence summary covering what the "
        "user wanted, what was accomplished, and any key decisions or "
        "findings. Reply with the summary text and nothing else."
    ),
    max_iterations=1,
    policy=PermissionPolicy(default="deny"),
)


def _confirm_on_stdin(tool_name: str, arguments) -> bool:
    """Asks the user to approve one tool call, for the CLI.

    This is the interactive half of the permission mechanism, and it
    lives here rather than in agent/permissions.py because this module
    already owns the REPL and the only input() in the codebase --
    keeping the policy model itself free of I/O is what lets it be
    tested without a terminal.

    Anything other than an explicit yes is a refusal, including a closed
    or piped stdin: if the question cannot actually be put to a human,
    the answer is no. `arguments` is shown as the model supplied it so
    the user can see what the tool would have been asked to do.
    """
    print(f"\nThis action needs your permission: {tool_name}")
    if arguments:
        print(f"  arguments: {arguments}")
    try:
        answer = input("  Allow it? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  (no answer available -- treating as denied)")
        return False

    approved = answer in {"y", "yes"}
    print("  -> allowed" if approved else "  -> denied")
    return approved


def _ensure_sample_receipts(workspace_root: str) -> None:
    """Self-healing: if workspace/receipts is missing or empty -- e.g. a
    Docker volume mount hid the image-baked samples, or a fresh clone
    hasn't had the manual setup step run yet -- repopulate it from the
    samples/receipts/ copy that ships alongside the code. Located
    relative to this file, so it works the same locally and in Docker.
    Never touches an already-populated workspace, so it can't clobber
    real receipts someone has already added."""
    receipts_dir = os.path.join(workspace_root, "receipts")
    has_files = os.path.isdir(receipts_dir) and any(
        os.path.isfile(os.path.join(receipts_dir, f)) for f in os.listdir(receipts_dir)
    )
    if has_files:
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(repo_root, "samples", "receipts")
    if not os.path.isdir(samples_dir):
        return  # no fallback available either -- nothing more we can do

    os.makedirs(receipts_dir, exist_ok=True)
    for fname in os.listdir(samples_dir):
        src = os.path.join(samples_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(receipts_dir, fname))
    print(f"(No receipts found in workspace -- copied sample receipts from {samples_dir})\n")


def check_state_paths(config) -> None:
    """Startup check: verify the directories this process must write to are
    actually writable, and fail with an actionable message if not.

    ChromaDB and the file tools both need to write under the paths in
    config.json. When that goes wrong -- a bind-mounted ./data owned by
    another uid, a stale volume seeded by an older image, a read-only
    mount -- the underlying errors are opaque: Chroma's Rust core reports
    "Permission denied (os error 13)" with no path, and a file tool raises
    an unhandled OSError mid-conversation. Checking the nearest existing
    ancestor of each configured path up front turns either into one
    message that names the path, the uid, and the fix.

    Raises RuntimeError (rather than exiting) so callers that are not a
    CLI -- the Streamlit dashboard, tests -- can render it themselves.
    """
    uid = os.getuid() if hasattr(os, "getuid") else "unknown (Windows)"
    for name, path in (
        ("workspace_root", config.workspace_root),
        ("ltm_db_path", config.ltm_db_path),
    ):
        # Probe the deepest ancestor that exists: the library will create
        # the leaf itself, but writability can only be tested where the
        # directory (or its parent) already does.
        probe = path
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if os.path.isfile(probe):
            probe = os.path.dirname(probe)

        if not os.access(probe, os.W_OK):
            raise RuntimeError(
                f"Startup check failed: '{probe}' (where config '{name}' "
                f"= '{path}' would be written) is not writable by uid {uid}.\n"
                "This is a permissions problem, not a missing database -- "
                "the directory is created automatically.\n"
                "  * Docker/Podman: use the named volume from "
                "docker-compose.yml; do not bind-mount ./data.\n"
                "  * Stale volume from an older image: "
                "'docker compose down -v' resets it.\n"
                "  * Rootless Podman: add --userns=keep-id; "
                "on SELinux add :Z to the mount."
            )


def _save_session_summary(runner, context, memory):
    """At session end, summarize the conversation into long-term memory.

    The summarizing itself is an ordinary agent run (see SUMMARY_SPEC),
    so this function is only the part that is genuinely its own concern:
    deciding there is something worth summarizing, and persisting the
    result. A failure here must not stop the process exiting, so it is
    reported and swallowed.
    """
    messages = context.as_list()
    if len(messages) <= 1:
        return  # no user messages, nothing to summarize

    try:
        result = runner.run(SUMMARY_SPEC, str(messages))
        if not result.success:
            print(f"\n(Could not save session summary: {result.reason})")
            return
        summary_text = str(result).strip()
        if summary_text:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key_topics = summary_text[:100].replace("\n", " ")
            memory.save_summary(summary_text, date=now, key_topics=key_topics)
            print("\n(Session summary saved.)")
    except Exception as exc:
        print(f"\n(Could not save session summary: {exc})")


def main():
    config = load_config("config.json")

    # Preflight before anything opens a database or writes a file: seed
    # the sample receipts, then verify the state paths are writable, so
    # the first failure anyone sees names the path and the fix instead of
    # coming out of ChromaDB as an opaque permission error.
    _ensure_sample_receipts(config.workspace_root)
    try:
        check_state_paths(config)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    # Metrics: registers listeners and starts the Prometheus-compatible
    # /metrics endpoint. Must happen before any run/llm_call/tool_call/
    # memory_op events fire, so it's the first thing after config loads.
    init_metrics()
    # Port is deliberately fixed at 9000: prometheus.yml reads plain YAML and
    # cannot follow an environment variable, so a port configurable in one
    # place only turns a mismatch into a silently empty dashboard instead of
    # an error. Docker-compose.yml and prometheus.yml both hardcode 9000.
    start_metrics_server(port=9000)

    # Structured JSON-lines observability log (handout 8.4's "log part").
    # A second, independent listener on the same hooks bus as metrics
    # above -- neither knows the other exists. Must also happen before
    # any events fire, for the same reason.
    if config.run_log.get("enabled", True):
        init_run_log(hooks, config.run_log.get("path", "data/run_log.jsonl"))

    memory = Memory(db_path=config.ltm_db_path)
    # Tool permissions. Built here and injected into react_step rather
    # than read from config inside the loop, so a caller that wants a
    # different policy -- a test, or a sub-agent later -- supplies its
    # own instead of having to rewrite the config file.
    policy = PermissionPolicy.from_config(config.permissions)
    # Sub-agent definitions, checked against the policy above before
    # anything runs. A sub-agent that could hold permissions the main
    # agent lacks would make delegating a way around the main agent's
    # own policy, so an escalating definition is a startup error, not a
    # surprise at the moment someone delegates. The same check runs again
    # when a sub-run is actually constructed, against its real parent --
    # this one catches it where it can be read and fixed.
    agent_catalog = AgentCatalog.from_config(config.agents)
    agent_catalog.require_narrower_than(policy)
    # The policy goes to the registry as well as to react_step below:
    # the loop is where the user gets asked, the registry is the
    # boundary that still holds for a caller that skips the loop.
    tools = ToolRegistry(
        workspace_root=config.workspace_root, memory=memory,
        mcp_servers=config.mcp_servers, policy=policy,
    )

    # Only user facts are folded into the system prompt, as SYSTEM_PROMPT
    # promises; other subjects come in on demand via recall_fact. The
    # filter must be optional so runner.py and the dashboard still get
    # the full store.
    facts_str = memory.recall_all(subject="user")
    effective_prompt = SYSTEM_PROMPT
    if facts_str:
        effective_prompt += "\n\n" + facts_str

    # Create LLM client for compaction summarization
    llm_client = LLMClient(config)

    context = Context(
        effective_prompt,
        max_tokens=config.context_window,
        llm_client=llm_client,
    )

    # The main agent is an AgentSpec too, run through the same runner as
    # any sub-agent. That is the handout's "a sub-agent must run on the
    # same harness that its parent agent", made literal rather than
    # asserted. include_memory is False here only because the prompt
    # above already has memory folded in for the CLI's own greeting.
    main_spec = AgentSpec(
        role="main",
        description="The inventory agent the user talks to directly.",
        instructions=effective_prompt,
        max_iterations=config.max_iterations,
        policy=policy,
    )
    runner = AgentRunner(
        config, tools, memory=memory, confirm=_confirm_on_stdin,
        catalog=agent_catalog,
    )

    print("Inventory agent ready. Type 'exit' to quit.\n")
    print(tools.startup_summary())
    print(policy.summary())
    print(agent_catalog.summary() + "\n")

    while True:
        try:
            task = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if task.lower() in {"exit", "quit"}:
            break
        if not task:
            continue

        try:
            # The persistent context is handed in, so a session's turns
            # accumulate in one history exactly as they did before.
            answer = runner.run(main_spec, task, context=context)
        except Exception as exc:
            print(f"Error: could not complete this turn ({type(exc).__name__}: {exc})")
            print("Try again, or type 'exit' to quit.")
            continue

        print(pretty_print_tables(answer))

    _save_session_summary(runner, context, memory)


if __name__ == "__main__":
    main()