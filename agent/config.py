import json
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model: str
    temperature: float
    base_url: str
    endpoint: str
    max_iterations: int
    workspace_root: str
    ltm_db_path: str
    api_key: str  # populated from environment, never from the JSON file
    context_window: int = 262144  # max tokens for the model's context window
    mcp_servers: dict = field(default_factory=dict)
    # Held as the raw config section, not as a PermissionPolicy: Config
    # stays a dumb data holder and agent/permissions.py owns the meaning
    # and validation of this section -- the same split already used for
    # mcp_servers, which ToolRegistry interprets.
    permissions: dict = field(default_factory=dict)
    # Sub-agent definitions, held raw for the same reason as permissions
    # above: agent/agents.py owns what a role definition means and
    # validates it via AgentCatalog.from_config().
    agents: dict = field(default_factory=dict)
    # The structured JSON-lines observability log (handout 8.4's "log
    # part"). Held raw and defaulted on, same pattern as mcp_servers --
    # a config file predating this feature still gets a working log at
    # the default path rather than silently getting none.
    run_log: dict = field(default_factory=dict)


def load_config(path: str = "config.json") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    llm = raw.get("llm", {})

    api_key = os.environ.get("INNKUBE_TOKEN")
    if not api_key:
        raise RuntimeError(
            "INNKUBE_TOKEN environment variable is not set. "
            "Export it before running the agent, e.g.:\n"
            "  export INNKUBE_TOKEN=your_token_here   (Linux/macOS)\n"
            "  set INNKUBE_TOKEN=your_token_here      (Windows cmd)"
        )

    # MCP servers are configured here, not hardcoded in ToolRegistry --
    # each entry controls whether that server is used at all, and (where
    # applicable) which script implements it. Defaults keep config.json
    # files without this section working exactly as before (both servers
    # enabled, default script paths).
    default_mcp_servers = {
        "recall": {"enabled": True},
        "ocr": {"enabled": True, "server_script": None},
        "qr": {"enabled": True},
    }
    mcp_servers = {**default_mcp_servers, **raw.get("mcp_servers", {})}

    # Tool permissions (handout 8.4). A config file without this section
    # gets an empty rule set and an 'allow' default, i.e. exactly the
    # behaviour that existed before permissions were introduced. The
    # outcome strings are validated by PermissionPolicy.from_config(),
    # not here, so a typo is reported by the component that understands
    # what a valid outcome is.
    default_permissions = {"default": "allow", "tools": {}}
    permissions = {**default_permissions, **raw.get("permissions", {})}

    # Sub-agent definitions (handout 8.4). A config file without this
    # section gets an empty catalog, which means delegation is simply not
    # configured -- the behaviour of every config written before
    # sub-agents existed. Role definitions are validated by
    # AgentCatalog.from_config(), not here, so the component that knows
    # what a valid definition looks like is the one that reports a bad one.
    default_agents = {"roles": {}}
    agents = {**default_agents, **raw.get("agents", {})}

    # Structured observability log. Defaults to enabled: unlike
    # permissions/agents, there is no old behaviour to preserve here --
    # this feature did not exist before, so "config file without this
    # section" should mean "get the log anyway", not "get nothing".
    default_run_log = {"enabled": True, "path": "data/run_log.jsonl"}
    run_log = {**default_run_log, **raw.get("run_log", {})}

    return Config(
        model=llm.get("model", ""),
        temperature=float(llm.get("temperature", 0.7)),
        base_url=llm.get("base_url", ""),
        endpoint=llm.get("endpoint", ""),
        max_iterations=int(raw.get("max_iterations", 15)),
        workspace_root=raw.get("workspace_root", "./workspace"),
        ltm_db_path=raw.get("ltm_db_path", "data/chroma"),
        api_key=api_key,
        context_window=int(llm.get("context_window", 262144)),
        mcp_servers=mcp_servers,
        permissions=permissions,
        agents=agents,
        run_log=run_log,
    )