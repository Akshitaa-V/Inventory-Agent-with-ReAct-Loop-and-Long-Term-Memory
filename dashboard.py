"""
dashboard.py — Combined Streamlit dashboard for the Common Inventory Agent.

Tabs:
  1. Inventory — upload receipts, search inventory, check recalls, view receipts
  2. Metrics   — agent observability via Prometheus (auto-refreshes every 10s)

Run with:
    streamlit run dashboard.py

Requires INNKUBE_TOKEN in the environment.
Metrics tab requires Prometheus running on port 9090.
"""

import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from agent.config import load_config
from agent.context import Context
from agent.llm_client import LLMClient
from agent.loop import react_step
from agent.ltm import Memory
from agent.main import SYSTEM_PROMPT, _ensure_sample_receipts, check_state_paths
from agent.tool_registry import ToolRegistry

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

st.set_page_config(page_title="Inventory Agent Dashboard", layout="wide")

_SEPARATOR_RE_TABLE = re.compile(r"^[\s\-:|]+$")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def parse_markdown_table(text: str):
    """Parses ALL pipe-delimited rows in `text` into a pandas DataFrame,
    using the first header+separator pair found to define columns.

    Real inventory.md files grow across many separate LLM calls over a
    long session, so formatting isn't perfectly uniform: blank lines
    between blocks, an occasional row with a missing or extra column,
    duplicated header rows if the agent ever re-emitted one. Rather than
    rejecting the whole table over one malformed row (which used to mean
    only the first block ever rendered as a table and everything after
    fell back to raw text), this scans the ENTIRE file for pipe rows and
    normalizes each one to the header's column count -- padding missing
    cells, truncating extra ones -- so the dashboard shows one complete,
    structured table covering everything that's actually in the file.
    """
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return None

    def split_row(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    def normalize(row, width):
        row = row[:width]
        row += [""] * (width - len(row))
        return row

    header = split_row(lines[0])
    width = len(header)
    start = 2 if len(lines) > 1 and _SEPARATOR_RE_TABLE.match(lines[1].strip()) else 1

    rows = []
    for line in lines[start:]:
        cells = split_row(line)
        if cells == header or _SEPARATOR_RE_TABLE.match(line.strip()):
            continue
        rows.append(normalize(cells, width))

    if not rows:
        return None

    return pd.DataFrame(rows, columns=header)


# ---------------------------------------------------------------------------
# Agent helpers (used by Inventory tab)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_agent():
    """Loaded once per server process, reused across reruns -- same
    config/LLM client/memory/tool registry the CLI would build in main().

    check_state_paths runs first so an unwritable data directory (or a
    ChromaDB open that fails on one) surfaces as a readable st.error at
    the caller instead of a raw traceback wedged in the cache. The check
    raises RuntimeError on every rerun -- exceptions are not cached --
    so fixing the volume and rerunning recovers without a server restart.
    """
    config = load_config("config.json")
    check_state_paths(config)
    llm = LLMClient(config)
    memory = Memory(db_path=config.ltm_db_path)
    tools = ToolRegistry(workspace_root=config.workspace_root, memory=memory, mcp_servers=config.mcp_servers)
    _ensure_sample_receipts(config.workspace_root)
    return config, llm, memory, tools


def run_agent_task(llm, memory, tools, config, prompt: str) -> str:
    """Runs one standalone agent task through the real react_step loop --
    identical mechanism to typing a line at the CLI's '>' prompt. Mirrors
    the CLI's own resilience: an LLM/network failure returns a clean
    message instead of crashing the whole page."""
    facts_str = memory.recall_all()
    effective_prompt = SYSTEM_PROMPT
    if facts_str:
        effective_prompt += "\n\n" + facts_str
    context = Context(effective_prompt)
    context.add_user_message(prompt)
    try:
        return react_step(llm, context, tools, config.max_iterations)
    except Exception as exc:
        return f"Error: could not complete this request ({type(exc).__name__}: {exc}). Try again."


# ---------------------------------------------------------------------------
# Prometheus helpers (used by Metrics tab)
# ---------------------------------------------------------------------------


def prom_query(expr: str) -> list[dict]:
    """Execute an instant PromQL query and return the result series."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": expr},
            timeout=5,
        )
        data = resp.json()
        if data.get("status") == "success":
            return data["data"]["result"]
    except requests.RequestException:
        pass
    return []


def metric_value(result: list[dict], labels: dict | None = None) -> float:
    """Extract a single float value from query results, optionally matching labels."""
    for series in result:
        if labels is None or all(series["metric"].get(k) == v for k, v in labels.items()):
            return float(series["value"][1])
    return 0.0


def metric_grouped_values(result: list[dict], group_label: str) -> dict[str, float]:
    """Group results by a label and return {label_value: float_value}."""
    out: dict[str, float] = {}
    for s in result:
        key = s["metric"].get(group_label, "unknown")
        out[key] = float(s["value"][1])
    return out


# ===========================================================================
# Tab 1: Inventory
# ===========================================================================


def render_inventory_tab():
    try:
        config, llm, memory, tools = get_agent()
    except (RuntimeError, OSError) as exc:
        # Unwritable state path or a ChromaDB open that failed on one --
        # show the actionable message (check_state_paths / Memory both
        # re-raise with the path and fix attached) instead of a traceback.
        st.error(str(exc))
        st.stop()

    st.title("Inventory Agent Dashboard")
    st.info(tools.startup_summary())

    # --- Upload + process ---
    st.header("Upload a receipt")
    uploaded = st.file_uploader(
        "Drop a photographed or scanned receipt",
        type=["jpg", "jpeg", "png", "tiff", "bmp", "pdf"],
    )

    if uploaded is not None:
        receipts_dir = os.path.join(config.workspace_root, "receipts")
        os.makedirs(receipts_dir, exist_ok=True)
        dest_path = os.path.join(receipts_dir, uploaded.name)

        with open(dest_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.image(dest_path, caption=uploaded.name, width=300)

        if st.button(f"Process '{uploaded.name}'"):
            with st.spinner("Agent is processing the receipt..."):
                prompt = (
                    f"Process the receipt at 'receipts/{uploaded.name}' and add "
                    f"its items to the inventory, following your usual rules "
                    f"(check memory first, OCR if it's an image, mark it "
                    f"processed once done)."
                )
                answer = run_agent_task(llm, memory, tools, config, prompt)
            st.session_state["last_result"] = answer
            st.rerun()

    if "last_result" in st.session_state:
        st.success(st.session_state.pop("last_result"))

    st.divider()

    # --- Product recall check ---
    st.header("Check a safety recall")
    recall_query = st.text_input("Product name (e.g. 'blender', 'baby monitor')", key="recall_query")
    if st.button("Check recall") and recall_query:
        with st.spinner("Checking the CPSC recall database..."):
            result = run_agent_task(
                llm, memory, tools, config,
                f"Check if '{recall_query}' has any safety recalls.",
            )
        st.info(result)

    st.divider()

    # --- Current inventory + search ---
    st.header("Current inventory")

    inventory_path = os.path.join(config.workspace_root, "inventory.md")
    if os.path.exists(inventory_path):
        with open(inventory_path, "r", encoding="utf-8") as f:
            inventory_text = f.read()
    else:
        inventory_text = ""

    query = st.text_input("Search (e.g. 'TV', 'laptop', a vendor name)")

    inventory_df = parse_markdown_table(inventory_text) if inventory_text else None

    if query:
        if inventory_df is not None:
            mask = inventory_df.apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
            matches_df = inventory_df[mask]
            if len(matches_df) > 0:
                st.success(f"{len(matches_df)} matching item(s):")
                st.dataframe(matches_df, width="stretch", hide_index=True)
            else:
                st.warning("No matches found.")
        else:
            matches = [line for line in inventory_text.splitlines() if query.lower() in line.lower()]
            if matches:
                st.success(f"{len(matches)} matching line(s):")
                st.code("\n".join(matches))
            else:
                st.warning("No matches found.")
    else:
        if inventory_df is not None:
            st.dataframe(inventory_df, width="stretch", hide_index=True)
        elif inventory_text:
            st.markdown(inventory_text)
        else:
            st.info("No inventory yet -- upload a receipt above to get started.")

    st.divider()

    # --- Receipt gallery ---
    st.header("Receipt images on file")

    receipts_dir = os.path.join(config.workspace_root, "receipts")
    if os.path.isdir(receipts_dir):
        image_files = sorted(f for f in os.listdir(receipts_dir) if f.lower().endswith(IMAGE_EXTS))
        if image_files:
            cols = st.columns(3)
            for i, fname in enumerate(image_files):
                with cols[i % 3]:
                    st.image(os.path.join(receipts_dir, fname), caption=fname, width="stretch")
        else:
            st.info("No receipt images in workspace/receipts yet.")
    else:
        st.info("Receipts folder doesn't exist yet.")


# ===========================================================================
# Tab 2: Metrics
# ===========================================================================


def render_metrics_tab():
    st_autorefresh(interval=10_000, key="metrics-refresh")

    st.title("Agent Metrics Dashboard")
    st.caption(f"Polling Prometheus at {PROMETHEUS_URL} — refreshes every 10s")

    try:
        requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=3)
        st.success("Prometheus connected", icon="\u2705")
    except Exception:
        st.error("Cannot reach Prometheus — make sure it is running on port 9090.")
        st.stop()

    # Row 1: Agent Throughput + Active Runs
    st.subheader("Agent Overview")
    col1, col2 = st.columns(2)

    with col1:
        # Summed over "kind": agent_runs_total is split into root and sub
        # runs, and metric_value returns only the first matching series, so
        # querying it raw would show one kind and silently drop the other.
        runs = prom_query("sum by (status) (agent_runs_total)")
        success = metric_value(runs, {"status": "success"})
        # "error", not "failure" -- metrics.py labels an unsuccessful run
        # "error", so the old query never matched and this bar always read 0.
        failure = metric_value(runs, {"status": "error"})
        total = success + failure

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Success", "Failure"],
            y=[success, failure],
            marker_color=["#2ecc71", "#e74c3c"],
            text=[int(success), int(failure)],
            textposition="auto",
        ))
        fig.update_layout(
            title="Agent Runs",
            yaxis_title="Count",
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Total runs: {int(total)}")

    with col2:
        active = prom_query("agent_active_runs")
        active_val = metric_value(active)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=active_val,
            title={"text": "Active Runs"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#3498db"},
                "steps": [
                    {"range": [0, 3], "color": "#d5f5e3"},
                    {"range": [3, 7], "color": "#fdebd0"},
                    {"range": [7, 10], "color": "#fadbd8"},
                ],
            },
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Run Duration + LLM Latency
    st.subheader("Latency")
    col3, col4 = st.columns(2)

    with col3:
        # Summed over "kind" for the same reason as above -- without the
        # sum this loop overwrites each bucket once per kind instead of
        # adding them, so the histogram showed only one kind's runs.
        run_dur = prom_query("sum by (le) (agent_run_duration_seconds_bucket)")
        if run_dur:
            buckets = {}
            for s in run_dur:
                le = s["metric"].get("le", "+Inf")
                buckets[le] = float(s["value"][1])

            sorted_bks = sorted(
                [(float(k), v) for k, v in buckets.items() if k != "+Inf"],
                key=lambda x: x[0],
            )
            labels = [f"<={b[0]}s" for b in sorted_bks] + [">30s"]
            counts = [b[1] for b in sorted_bks] + [buckets.get("+Inf", 0)]

            fig = go.Figure(go.Bar(
                x=labels,
                y=counts,
                marker_color="#9b59b6",
            ))
            fig.update_layout(
                title="Agent Run Duration Distribution",
                xaxis_title="Duration",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No run duration data yet.")

    with col4:
        llm_dur = prom_query("llm_call_duration_seconds_bucket")
        if llm_dur:
            buckets = {}
            for s in llm_dur:
                le = s["metric"].get("le", "+Inf")
                buckets[le] = float(s["value"][1])

            sorted_bks = sorted(
                [(float(k), v) for k, v in buckets.items() if k != "+Inf"],
                key=lambda x: x[0],
            )
            labels = [f"<={b[0]}s" for b in sorted_bks] + [">20s"]
            counts = [b[1] for b in sorted_bks] + [buckets.get("+Inf", 0)]

            fig = go.Figure(go.Bar(
                x=labels,
                y=counts,
                marker_color="#e67e22",
            ))
            fig.update_layout(
                title="LLM Call Latency Distribution",
                xaxis_title="Latency",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LLM latency data yet.")

    # Row 3: Token Consumption
    st.subheader("Token Usage")
    col5, col6 = st.columns(2)

    with col5:
        tokens = prom_query("llm_tokens_total")
        token_data = metric_grouped_values(tokens, "direction")

        if token_data:
            fig = go.Figure(go.Bar(
                x=["Input Tokens", "Output Tokens"],
                y=[token_data.get("input", 0), token_data.get("output", 0)],
                marker_color=["#1abc9c", "#3498db"],
                text=[f"{int(token_data.get('input', 0)):,}", f"{int(token_data.get('output', 0)):,}"],
                textposition="auto",
            ))
            fig.update_layout(
                title="Token Consumption",
                yaxis_title="Tokens",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No token data yet.")

    with col6:
        llm_calls = prom_query("llm_calls_total")
        call_data = metric_grouped_values(llm_calls, "model")

        if call_data:
            fig = go.Figure(go.Pie(
                labels=list(call_data.keys()),
                values=list(call_data.values()),
                hole=0.4,
                marker_colors=px.colors.qualitative.Set2,
            ))
            fig.update_layout(
                title="LLM Calls by Model",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LLM call data yet.")

    # Row 4: Tool Usage
    st.subheader("Tool Usage")
    col7, col8 = st.columns(2)

    with col7:
        tool_calls = prom_query("tool_calls_total")
        tool_data = metric_grouped_values(tool_calls, "tool")

        if tool_data:
            sorted_tools = sorted(tool_data.items(), key=lambda x: x[1], reverse=True)
            fig = go.Figure(go.Bar(
                x=[t[0] for t in sorted_tools],
                y=[t[1] for t in sorted_tools],
                marker_color="#2980b9",
                text=[int(t[1]) for t in sorted_tools],
                textposition="auto",
            ))
            fig.update_layout(
                title="Tool Calls by Name",
                xaxis_title="Tool",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tool call data yet.")

    with col8:
        tool_by_type = metric_grouped_values(tool_calls, "type")

        if tool_by_type:
            fig = go.Figure(go.Pie(
                labels=list(tool_by_type.keys()),
                values=list(tool_by_type.values()),
                hole=0.4,
                marker_colors=["#e74c3c", "#2ecc71"],
            ))
            fig.update_layout(
                title="Tool Calls: Built-in vs MCP",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tool type data yet.")

    # Row 5: Success/Failure Rates + Permissions
    st.subheader("Outcomes")
    col9, col10 = st.columns(2)

    with col9:
        tool_status = metric_grouped_values(tool_calls, "status")

        if tool_status:
            fig = go.Figure(go.Pie(
                labels=list(tool_status.keys()),
                values=list(tool_status.values()),
                hole=0.5,
                marker_colors=["#2ecc71", "#e74c3c"],
            ))
            fig.update_layout(
                title="Tool Call Success vs Failure",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tool status data yet.")

    with col10:
        perms = prom_query("permission_decisions_total")
        perm_data = metric_grouped_values(perms, "decision")

        if perm_data:
            fig = go.Figure(go.Bar(
                x=list(perm_data.keys()),
                y=list(perm_data.values()),
                marker_color=["#2ecc71", "#e74c3c"],
                text=[int(v) for v in perm_data.values()],
                textposition="auto",
            ))
            fig.update_layout(
                title="Permission Decisions",
                yaxis_title="Count",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No permission data yet.")

    # Row 6: Sub-agent Activity
    st.subheader("Sub-agent Activity")
    col11, _ = st.columns([1, 1])

    with col11:
        # Summed over "role": subagent_runs_total carries both role and
        # status, so grouping the raw series by status alone would have
        # one role's count overwrite another's rather than adding them.
        # (Group by role instead for a per-sub-agent breakdown.)
        sub_runs = prom_query("sum by (status) (subagent_runs_total)")
        sub_data = metric_grouped_values(sub_runs, "status")

        if sub_data:
            fig = go.Figure(go.Bar(
                x=list(sub_data.keys()),
                y=list(sub_data.values()),
                marker_color=["#f39c12", "#8e44ad"],
                text=[int(v) for v in sub_data.values()],
                textposition="auto",
            ))
            fig.update_layout(
                title="Sub-agent Runs by Status",
                yaxis_title="Count",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sub-agent data yet.")


# ===========================================================================
# Main layout
# ===========================================================================

tab_inventory, tab_metrics = st.tabs(["Inventory", "Metrics"])

with tab_inventory:
    render_inventory_tab()

with tab_metrics:
    render_metrics_tab()
