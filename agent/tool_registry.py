import asyncio
import json
import os

from mcp_servers.recall.recall_mcp_client import RecallMCPClient
from mcp_servers.ocr.ocr_mcp_client import OCRMCPClient
from mcp_servers.qr.qr_mcp_client import QRMCPClient
from agent.memory import MemoryStore
from agent.hooks import hooks
from agent.delegation import Delegation
from agent.permissions import PolicyEscalation
from agent.permissions import Decision, PermissionPolicy, refusal_text


class ToolRegistry:
    def __init__(self, workspace_root: str = "./workspace", memory=None,
                 receipt_memory_path: str = None, mcp_servers: dict = None,
                 policy: PermissionPolicy = None):
        self._workspace_root = os.path.abspath(workspace_root)

        # The hard permission boundary. Every call() goes through this,
        # built-in and MCP alike, whether or not the caller came via the
        # ReAct loop -- the loop is where a human gets asked, this is what
        # cannot be bypassed by not going through the loop.
        #
        # Defaults to an all-allow policy so that a registry built without
        # one behaves exactly as it did before permissions existed, which
        # is what the existing tests construct. Real enforcement comes from
        # main.py injecting the configured policy.
        self._policy = policy if policy is not None else PermissionPolicy()

        # `memory` = teammate's ChromaDB-backed general fact store (Memory
        # from agent/ltm.py) -- freeform (subject, predicate, value) facts
        # the LLM decides to remember mid-conversation.
        self._memory = memory

        # `self._receipt_memory` = my JSON-backed receipt dedup/totals
        # store -- a different concern (specific to processing receipts),
        # kept as a separate file so neither memory system has to know
        # about the other's internals.
        if receipt_memory_path is None:
            receipt_memory_path = os.path.join(self._workspace_root, "memory.json")
        self._receipt_memory = MemoryStore(path=receipt_memory_path)

        self._tools = {
            "create": self._create,
            "navigate": self._navigate,
            "search": self._search,
            "read": self._read,
            "read_many": self._read_many,
            "modify": self._modify,
            "remember_fact": self._remember_fact,
            "recall_fact": self._recall_fact,
            "recall_summaries": self._recall_summaries,
            "recall_all_summaries": self._recall_all_summaries,
            "check_receipt_processed": self._check_receipt_processed,
            "mark_receipt_processed": self._mark_receipt_processed,
            "forget_fact": self._forget_fact,
        }

        # MCP servers are configured, not hardcoded: which servers run at
        # all, and (for OCR) which script implements it, come from
        # config.json's "mcp_servers" section -- passed in by main.py.
        # Defaults (both enabled, default OCR script) apply if the caller
        # doesn't supply this.
        mcp_servers = mcp_servers or {
            "recall": {"enabled": True},
            "ocr": {"enabled": True, "server_script": None},
            "qr": {"enabled": True},
        }

        self._mcp_tool_owner = {}  # tool name -> client instance that serves it
        self._mcp_schemas = []

        recall_cfg = mcp_servers.get("recall", {"enabled": True})
        if recall_cfg.get("enabled", True):
            self._recall_client = RecallMCPClient()
            self._load_mcp_tools(self._recall_client, "Product Recall")
        else:
            self._recall_client = None

        ocr_cfg = mcp_servers.get("ocr", {"enabled": True, "server_script": None})
        if ocr_cfg.get("enabled", True):
            ocr_script = ocr_cfg.get("server_script")
            self._ocr_client = OCRMCPClient(server_script=ocr_script) if ocr_script else OCRMCPClient()
            self._load_mcp_tools(self._ocr_client, "OCR")
        else:
            self._ocr_client = None

        # The QR client is the only one that needs to know where the
        # workspace is: its server writes files, and the directory it writes
        # them into is derived from workspace_root and passed to the
        # subprocess as an environment variable. Recall and OCR only read,
        # so neither needs any of this.
        qr_cfg = mcp_servers.get("qr", {"enabled": True})
        if qr_cfg.get("enabled", True):
            self._qr_client = QRMCPClient(workspace_root=self._workspace_root)
            self._load_mcp_tools(self._qr_client, "QR")
        else:
            self._qr_client = None

    def _load_mcp_tools(self, client, label: str) -> None:
        try:
            tools = asyncio.run(client.list_tools())
            for tool in tools:
                self._mcp_tool_owner[tool.name] = client
                self._mcp_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                })
        except Exception as e:
            print(f"Warning: {label} MCP server unavailable, continuing without it ({e})")

    def startup_summary(self) -> str:
        """Human-readable greeting drawn from receipt memory. Call this
        once at session startup -- receipt memory itself is already
        loaded at __init__ time."""
        return self._receipt_memory.summary()

    def is_builtin(self, name: str) -> bool:
        """Returns True if `name` is a built-in tool (not MCP)."""
        return name in self._tools

    def _safe_path(self, path: str) -> str:
        """Resolves a relative path against the workspace root and ensures
        it cannot escape the workspace (blocks '../' traversal, absolute
        paths outside the sandbox, etc.).

        Also case-insensitively resolves the path against what's actually
        on disk if the exact-case path doesn't exist. This matters because
        Windows filesystems are case-insensitive but the Docker image's
        Linux filesystem is not -- a request for 'receipt 1.jpg' silently
        found 'Receipt 1.jpg' in every local test on Windows, but fails
        outright inside the container without this fallback."""
        full_path = os.path.abspath(os.path.join(self._workspace_root, path))
        if os.path.commonpath([full_path, self._workspace_root]) != self._workspace_root:
            raise ValueError(f"Path '{path}' escapes the workspace and is not allowed.")

        if not os.path.exists(full_path):
            resolved = self._resolve_case_insensitive(full_path)
            if resolved is not None:
                full_path = resolved

        return full_path

    def _resolve_case_insensitive(self, full_path: str) -> str:
        """If `full_path` doesn't exist with its exact casing, walks it
        component by component from workspace_root, matching each part
        case-insensitively against real directory entries. Returns the
        corrected, real-casing path if every component is found this way,
        or None if any component genuinely doesn't exist (in which case
        the caller reports the original not-found error, unchanged)."""
        relative = os.path.relpath(full_path, self._workspace_root)
        if relative.startswith(".."):
            return None  # shouldn't happen, _safe_path already blocked this

        parts = relative.split(os.sep)
        current = self._workspace_root

        for part in parts:
            candidate = os.path.join(current, part)
            if os.path.exists(candidate):
                current = candidate
                continue

            if not os.path.isdir(current):
                return None

            try:
                entries = os.listdir(current)
            except OSError:
                return None

            match = next((e for e in entries if e.lower() == part.lower()), None)
            if match is None:
                return None
            current = os.path.join(current, match)

        return current

    def schemas(self) -> list:
        """JSON schemas describing each tool, sent to the LLM. Combines
        built-in file-system tools, general fact memory, receipt memory,
        and discovered MCP tools (Product Recall and OCR)."""
        built_in_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "create",
                    "description": "Creates a file or folder at the given path inside the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path (e.g. 'docs/notes.txt' or 'src')",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["file", "folder"],
                                "description": "Whether to create a file or folder.",
                            },
                        },
                        "required": ["path", "type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "navigate",
                    "description": "Lists files and folders at the given path inside the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path (e.g. 'src' or 'docs'). Empty for workspace root.",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for a text string inside files in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The text to search for.",
                            },
                            "path": {
                                "type": "string",
                                "description": "Optional: restrict search to this directory (relative path).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Reads the entire contents of a single file. If you need to read every file in a folder, prefer 'read_many' instead to save round trips.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file (e.g. 'docs/notes.txt')",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_many",
                    "description": "Reads every file inside a folder in a single call, returning each file's content labeled by filename. Prefer this over calling 'read' repeatedly when you need to read multiple files in the same folder.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative folder path whose files should all be read, e.g. 'receipts'",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "modify",
                    "description": "Appends text to the end of an existing file. This cannot overwrite or delete files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file to modify.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The text content to append to the file.",
                            }
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remember_fact",
                    "description": "Store a fact in long-term memory that persists across sessions. Use for user preferences, learned information, or important details worth remembering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "What the fact is about (e.g. 'user', 'laptop', 'project').",
                            },
                            "predicate": {
                                "type": "string",
                                "description": "The attribute or relationship (e.g. 'prefers', 'brand', 'uses').",
                            },
                            "value": {
                                "type": "string",
                                "description": "The value of the fact (e.g. 'dark mode', 'Dell', 'SQLite').",
                            },
                        },
                        "required": ["subject", "predicate", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_fact",
                    "description": "Search long-term memory for facts matching a query. Use when you need to recall user preferences or previously stored information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search term to match against stored facts.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_summaries",
                    "description": "Semantic search over stored session summaries, returning only those relevant to a query. Prefer this over recall_all_summaries when you need summaries about a specific topic, to avoid loading every stored summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The topic or question to match against stored summaries.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_all_summaries",
                    "description": "Retrieve all stored session summaries with their full text. Use when the user asks about past conversations broadly, or you need a complete overview of what has been discussed across sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_receipt_processed",
                    "description": "Checks whether a receipt file has already been processed and recorded in memory, based on its content (not just its filename). Call this BEFORE running OCR or adding a receipt's items to inventory.md, to avoid creating duplicate entries across sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the receipt file (e.g. 'receipts/receipt1.jpg')",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_receipt_processed",
                    "description": "Records a receipt as processed in persistent memory, updating the running item count and total value. Call this AFTER successfully adding a receipt's items to inventory.md -- never before, and never for a receipt that check_receipt_processed already reported as processed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the receipt file that was just processed.",
                            },
                            "item_count": {
                                "type": "integer",
                                "description": "How many items from this receipt were added to inventory.md.",
                            },
                            "total_value": {
                                "type": "number",
                                "description": "Combined purchase price of the items added from this receipt (0 if unknown).",
                            },
                        },
                        "required": ["path", "item_count", "total_value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "forget_fact",
                    "description": "Soft-delete a fact from long-term memory by its ID. The fact is marked inactive and will no longer appear in recall results, but is not permanently removed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The ID of the fact to forget (as shown in recall results).",
                            },
                        },
                        "required": ["id"],
                    },
                },
            },
        ]

        return built_in_schemas + self._mcp_schemas

    def _refuse(self, name: str, outcome, reason: str) -> str:
        """Reports a refusal and returns the text for the tool result.

        Only refusals are reported, never allows. A call that came via the
        loop was already checked and reported there, and the loop does not
        reach this method unless it allowed the call -- so this layer can
        only ever refuse something the loop did not handle. That makes a
        permission_decision event from here precisely the signal that a
        caller bypassed the loop, with no double-counting of the ordinary
        path. Nothing listens yet; emit with no listeners is a no-op.
        """
        hooks.emit(
            "permission_decision",
            tool_name=name,
            decision=outcome.value,
            source=self._policy.source_for(name),
            allowed=False,
        )
        return refusal_text(name, reason)

    def call(self, name: str, arguments, *, confirmed: bool = None) -> str:
        """Runs a tool, subject to the permission policy.

        This is the enforcement boundary: the check below sits above the
        built-in/MCP dispatch, so both kinds are covered by one code path
        rather than two that have to be kept in step, and a caller that
        reaches here without going through the loop is still bound by it.

        confirmed: whether the caller already obtained the user's
            confirmation for this specific call. The loop passes True
            after a human said yes. This registry never prompts -- it has
            no confirmation channel at all, which is what makes it
            impossible for it to ask a second time after the loop already
            has, and impossible for it to block on stdin in a context
            with no user (a test, the dashboard, a sub-agent). A
            require-user-confirmation tool reached with no confirmation
            is therefore refused, not queried.

            Note this is an assertion by the caller, not a capability:
            passing confirmed=True without having asked anyone does get
            past a confirmation gate. What the boundary guarantees is
            that a caller who does nothing gets the safe outcome. A deny
            rule has no such gap -- it ignores `confirmed` entirely.

        Refusals are returned as a result string, never raised, matching
        how every other failure in this method is reported: the loop
        feeds the string back to the model as the tool result.
        """
        if not isinstance(name, str):
            return f"Error: tool name must be a string, got {type(name).__name__}"

        # Deliberately before argument parsing and before the unknown-tool
        # branch below. A blocked tool should be refused on the grounds it
        # is blocked, not report a JSON parse error first; and under a
        # deny-by-default policy an unknown name should be refused rather
        # than have the registry disclose which tools exist.
        outcome = self._policy.decide(name)
        if outcome is Decision.DENY:
            return self._refuse(name, outcome, "denied")
        if outcome is Decision.REQUIRE_USER_CONFIRMATION and confirmed is not True:
            return self._refuse(name, outcome, "unconfirmed")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: could not parse arguments for '{name}'"

        if name in self._tools:
            try:
                return self._tools[name](**arguments)
            except ValueError as e:
                return f"Error: {e}"

        if name in self._mcp_tool_owner:
            client = self._mcp_tool_owner[name]
            # Maps an MCP tool to the argument of its that names a file, so
            # that argument can be forced inside the workspace before the
            # call goes out. Only tools that accept a *path* belong here.
            #
            # generate_qr_code is deliberately absent. It takes a bare
            # filename, refuses anything containing a separator or '..', and
            # joins what is left to one fixed directory itself -- so it has
            # no path argument to contain, and running _safe_path over its
            # filename would turn "label.png" into an absolute path that the
            # server would then reject as not being a bare filename.
            #
            # Keeping the check inside the server also means it still holds
            # when the server is driven by something other than this
            # registry, such as the MCP Inspector, where _safe_path is not
            # in the picture at all.
            path_arg = {"ocr_extract_text": "image_path", "ocr_extract_text_from_pdf": "pdf_path"}.get(name)
            if path_arg and path_arg in arguments:
                try:
                    arguments = {**arguments, path_arg: self._safe_path(arguments[path_arg])}
                except ValueError as e:
                    return f"Error: {e}"
            try:
                return asyncio.run(client.call_tool(name, arguments))
            except Exception as e:
                return f"Error: could not reach MCP server for '{name}' ({e})"

        return f"Error: unknown tool '{name}'"

    def _create(self, path: str, type: str) -> str:
        full_path = self._safe_path(path)

        if type == "folder":
            os.makedirs(full_path, exist_ok=True)
            return f"Created folder: {full_path}"

        if type == "file":
            if os.path.exists(full_path):
                return f"Error: '{path}' already exists. Use 'modify' to change it."
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w"):
                pass
            return f"Created file: {full_path}"

        return f"Error: invalid type '{type}', must be 'file' or 'folder'"

    def _navigate(self, path: str = "") -> str:
        full_path = self._safe_path(path) if path else self._workspace_root

        if not os.path.isdir(full_path):
            return f"Error: '{path}' is not a directory"

        entries = sorted(os.listdir(full_path))
        if not entries:
            return f"Empty directory: {path or '.'}"

        result = []
        for entry in entries:
            entry_path = os.path.join(full_path, entry)
            if os.path.isdir(entry_path):
                result.append(f"{entry}/ (folder)")
            else:
                result.append(f"{entry} (file)")

        return "\n".join(result)

    def _search(self, query: str, path: str = "") -> str:
        search_root = self._safe_path(path) if path else self._workspace_root

        if not os.path.isdir(search_root):
            return f"Error: '{path}' is not a directory"

        results = []
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for filename in files:
                if filename.endswith((".pyc", ".pyo", ".so", ".dll")):
                    continue

                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if query in line:
                                rel = os.path.relpath(filepath, self._workspace_root)
                                results.append(f"{rel}:{line_num}: {line.rstrip()}")
                except (IOError, UnicodeDecodeError):
                    continue

        if not results:
            return f"No matches found for '{query}'"

        if len(results) > 20:
            return "\n".join(results[:20]) + f"\n... and {len(results) - 20} more"

        return "\n".join(results)

    def _read(self, path: str) -> str:
        full_path = self._safe_path(path)

        if not os.path.isfile(full_path):
            return f"Error: The file '{path}' does not exist or is a directory."

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _read_many(self, path: str) -> str:
        full_path = self._safe_path(path)

        if not os.path.isdir(full_path):
            return f"Error: '{path}' is not a directory"

        filenames = sorted(
            f for f in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, f))
        )

        if not filenames:
            return f"'{path}' contains no files."

        sections = []
        for filename in filenames:
            filepath = os.path.join(full_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                sections.append(f"--- {filename} ---\n{content}")
            except (UnicodeDecodeError, OSError) as e:
                sections.append(f"--- {filename} ---\n[could not read: {e}]")

        return "\n\n".join(sections)

    def _modify(self, path: str, content: str) -> str:
        full_path = self._safe_path(path)

        if not os.path.isfile(full_path):
            return f"Error: The file '{path}' does not exist. Use the 'create' tool first."

        try:
            with open(full_path, "a", encoding="utf-8") as f:
                f.write(f"\n{content}")
            return f"Successfully appended content to '{path}'"
        except Exception as e:
            return f"Error modifying file: {str(e)}"

    def _remember_fact(self, subject: str, predicate: str, value: str) -> str:
        if not self._memory:
            return "Error: long-term memory is not available."
        return self._memory.remember(subject, predicate, value)

    def _recall_fact(self, query: str) -> str:
        if not self._memory:
            return "Error: long-term memory is not available."
        return self._memory.recall(query)

    def _recall_summaries(self, query: str) -> str:
        if not self._memory:
            return "Error: long-term memory is not available."
        return self._memory.recall_summaries(query)

    def _recall_all_summaries(self) -> str:
        if not self._memory:
            return "Error: long-term memory is not available."
        return self._memory.recall_all_summaries()

    def _check_receipt_processed(self, path: str) -> str:
        full_path = self._safe_path(path)

        if not os.path.isfile(full_path):
            return f"Error: '{path}' does not exist."

        if self._receipt_memory.is_processed(full_path):
            return f"'{path}' has already been processed and recorded in memory. Skip it -- do not add it to inventory.md again."

        return f"'{path}' has not been processed yet. Safe to OCR and add to inventory.md."

    def _mark_receipt_processed(self, path: str, item_count: int, total_value: float) -> str:
        full_path = self._safe_path(path)

        if not os.path.isfile(full_path):
            return f"Error: '{path}' does not exist."

        self._receipt_memory.record_receipt(full_path, item_count=item_count, total_value=total_value)
        return f"Recorded '{path}' as processed ({item_count} item(s), value {total_value:.2f})."

    def _forget_fact(self, id: str) -> str:
        if not self._memory:
            return "Error: long-term memory is not available."
        return self._memory.forget(id)


# The delegation tool's name. Defined here rather than inline because
# both the schema and the dispatch in call() must agree on it, and a
# policy can name it like any other tool.
DELEGATE_TOOL_NAME = "delegate_to_subagent"


class RegistryView:
    """A narrow, policy-scoped view onto a shared ToolRegistry.

    The ReAct loop touches a registry through exactly three methods --
    schemas(), is_builtin() and call() -- so an object offering just
    those can stand in for one. That is what lets a sub-agent have its
    own permissions without its own registry, which matters for two
    concrete reasons:

      - Building a ToolRegistry runs MCP tool discovery, spawning a
        subprocess per configured server. A registry per sub-agent would
        re-pay that on every delegation.
      - Each registry builds its own MemoryStore, which loads the receipt
        JSON into memory at startup and writes it back whole. Two of them
        over the same file silently lose each other's writes.

    This is an enforcement layer, not a pass-through. call() applies this
    view's policy and refuses before the shared registry is touched at
    all, because the view is the object a caller actually holds -- a
    caller that skips the ReAct loop must still be bound by the policy it
    was given. The shared registry then re-checks its own policy
    underneath, so the two layers stack rather than replacing each other.

    Deliberately not offered: any way to widen access. The view can only
    refuse things the underlying registry would have allowed, never the
    reverse, and there is no parameter that would override the
    registry's own policy.
    """

    def __init__(self, registry, policy: PermissionPolicy, *, delegation: Delegation = None):
        self._registry = registry
        self._policy = policy
        # Present only when this run is allowed to delegate. AgentRunner
        # leaves it None once the delegation depth is used up, so the tool
        # is simply not offered rather than offered and then refused --
        # the model cannot spend an iteration asking for something it was
        # never going to be allowed to do.
        self._delegation = delegation

    def _delegate_schema(self) -> dict:
        """The delegation tool as the parent's model sees it.

        The role parameter is an enum of the configured roles, so the
        model cannot invent one, and each role's description is listed so
        it can tell them apart. The descriptions are written for this
        audience -- they say what a sub-agent is for, not how it works.
        """
        catalog = self._delegation.catalog
        roles = catalog.roles()
        described = " ".join(
            f"'{role}': {catalog.get(role).description}" for role in roles
        )
        return {
            "type": "function",
            "function": {
                "name": DELEGATE_TOOL_NAME,
                "description": (
                    "Hand one self-contained task to a specialized sub-agent and "
                    "get back just its result. Useful when a task needs a lot of "
                    "reading or intermediate steps whose details you do not need "
                    "to keep -- the sub-agent works in its own context and returns "
                    "only its conclusion. Give it everything it needs in one "
                    "instruction; it cannot see this conversation. "
                    f"Available sub-agents -- {described}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": roles,
                            "description": "Which sub-agent to delegate to.",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "The complete instruction for the sub-agent, "
                                "self-contained: it has no access to this "
                                "conversation, so name files, items and any "
                                "context it needs explicitly."
                            ),
                        },
                    },
                    "required": ["role", "task"],
                },
            },
        }

    def _delegate(self, role, task) -> str:
        """Runs a sub-agent and returns its result as this tool's result.

        The sub-run gets its own fresh context, so the parent's
        conversation never enters it and only the returned string comes
        back -- which is the whole benefit of delegating a task whose
        intermediate steps the parent does not need.
        """
        delegation = self._delegation
        catalog = delegation.catalog
        parent = delegation.parent

        # Re-checked even though the tool is not offered past the limit,
        # for the same reason the registry re-checks the policy the loop
        # already checked: this is the layer a caller cannot skip.
        if parent.depth >= catalog.max_delegation_depth:
            return (
                f"Error: delegation is not available at this depth "
                f"(limit {catalog.max_delegation_depth}). Do the task yourself "
                f"or report that it cannot be delegated further."
            )

        if not isinstance(role, str) or role not in catalog:
            available = ", ".join(catalog.roles()) or "none configured"
            return (
                f"Error: no sub-agent named {role!r}. Available: {available}."
            )
        if not isinstance(task, str) or not task.strip():
            return (
                f"Error: the task for sub-agent {role!r} must be a non-empty "
                f"instruction. It cannot see this conversation, so say what it "
                f"needs to do and name anything it needs."
            )

        try:
            result = delegation.runner.run(catalog.get(role), task, parent=parent)
        except PolicyEscalation as exc:
            # Caught deliberately rather than left to the loop's generic
            # exception handling, which would word it as a tool crash.
            # A definition read from config cannot reach here -- main.py
            # refuses to start on one -- so this is a spec built in code,
            # and the parent's model should be told plainly that the
            # delegation was refused and not retry it.
            return (
                f"Permission denied: the {role!r} sub-agent is configured with "
                f"permissions its parent does not hold, so it was not run "
                f"({exc}). Do not retry it -- do the task yourself if you are "
                f"permitted to, or tell the user the sub-agent is misconfigured."
            )

        if not result.success:
            # Surfaced rather than returned as though it were an answer:
            # a sub-agent that ran out of iterations returns an ordinary
            # string, and the parent would otherwise treat it as a result.
            return (
                f"The {role!r} sub-agent did not finish ({result.reason}). "
                f"Its last message was: {str(result)}"
            )
        return str(result)

    def schemas(self) -> list:
        """The tools this view exposes to the model.

        Filtered only when the policy denies by default. The distinction
        is deliberate:

          - An allowlist policy (deny by default, a handful of allows) is
            a specialist's. Showing it every tool in the harness would
            have it call ones it cannot use and spend an iteration
            learning that each time.
          - A denylist policy (allow by default, a few refusals) is a
            broad agent's. Keeping the refused tools visible is what
            makes the refusal happen at all -- a tool that is never
            offered is never denied either, so the deny outcome would
            stop being exercised or observable.
        """
        schemas = self._registry.schemas()
        if self._policy.default is Decision.DENY:
            schemas = [
                schema
                for schema in schemas
                if self._policy.decide(schema["function"]["name"]) is not Decision.DENY
            ]

        # Offered only when this run may delegate at all, and only when
        # the policy does not refuse it -- it is a tool like any other.
        if (
            self._delegation
            and self._delegation.catalog.roles()
            and self._policy.decide(DELEGATE_TOOL_NAME) is not Decision.DENY
        ):
            schemas = schemas + [self._delegate_schema()]
        return schemas

    def is_builtin(self, name: str) -> bool:
        # Delegation is part of the harness, not something an MCP server
        # provides, so it must not be reported as an MCP tool call.
        if name == DELEGATE_TOOL_NAME:
            return True
        return self._registry.is_builtin(name)

    def call(self, name: str, arguments, *, confirmed: bool = None) -> str:
        """Runs a tool if this view's policy permits it.

        `confirmed` is passed through untouched, so a confirmation the
        loop already obtained is not asked for a second time by the
        registry underneath. Like the registry, this view never prompts.
        """
        if not isinstance(name, str):
            return f"Error: tool name must be a string, got {type(name).__name__}"

        outcome = self._policy.decide(name)
        if outcome is Decision.DENY:
            return self._refuse(name, outcome, "denied")
        if outcome is Decision.REQUIRE_USER_CONFIRMATION and confirmed is not True:
            return self._refuse(name, outcome, "unconfirmed")

        if name == DELEGATE_TOOL_NAME:
            if not self._delegation:
                return (
                    "Error: delegation is not available to this agent. Do the "
                    "task yourself or report that it cannot be done."
                )
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return f"Error: could not parse arguments for '{name}'"
            if not isinstance(arguments, dict):
                return f"Error: could not parse arguments for '{name}'"
            return self._delegate(arguments.get("role"), arguments.get("task"))

        return self._registry.call(name, arguments, confirmed=confirmed)

    def _refuse(self, name: str, outcome, reason: str) -> str:
        """Reports a refusal and returns the text for the tool result.

        Refusals only, for the same reason ToolRegistry reports only
        refusals: the loop already reported every decision it made, and
        it only reaches this method for calls it allowed -- so a refusal
        here means a caller bypassed the loop, which is the case worth
        seeing, and the ordinary path is never counted twice.
        """
        hooks.emit(
            "permission_decision",
            tool_name=name,
            decision=outcome.value,
            source=self._policy.source_for(name),
            allowed=False,
        )
        return refusal_text(name, reason)
