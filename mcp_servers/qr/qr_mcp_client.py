"""
qr_mcp_client.py — thin async client for qr_server.py.

Same per-call connect/close pattern as recall_mcp_client.py and
ocr_mcp_client.py: a fresh stdio connection for each list_tools() /
call_tool(), so ToolRegistry can drive it with plain asyncio.run(...) with
no session lifecycle to manage.

It differs from the other two in one respect. The QR server writes files,
and the directory it writes them to has to be the agent's configured
workspace -- which the server cannot know, since it is spawned as a bare
subprocess. That configuration is passed as an environment variable, and
passing it requires building the child environment explicitly; see
_build_env for why env=None will not do.
"""

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

SERVER_SCRIPT = str(Path(__file__).parent / "qr_server.py")

# Subdirectory of the workspace that generated codes are written into. Kept
# out of the workspace root so QR output is visibly separate from the
# inventory files the agent maintains.
OUTPUT_SUBDIR = "qr_codes"


class QRMCPClient:
    """Self-hosted QR generation MCP client. Spawns qr_server.py fresh for
    each call.

    Args:
        workspace_root: The agent's workspace. Generated codes are written
            to <workspace_root>/qr_codes. If omitted, the server falls back
            to its own default, which is the repository's workspace -- that
            fallback exists for running the server standalone, and is not
            what the agent should rely on.
    """

    def __init__(
        self,
        workspace_root: str | None = None,
        server_script: str = SERVER_SCRIPT,
        python_executable: str = sys.executable,
    ):
        self._server_script = server_script
        self._python_executable = python_executable
        self._output_dir = (
            os.path.join(os.path.abspath(workspace_root), OUTPUT_SUBDIR)
            if workspace_root
            else None
        )

    def _build_env(self) -> dict:
        """Builds the child process environment.

        Passing env=None -- what the recall and OCR clients do -- does not
        mean "inherit the parent environment". MCP's stdio transport
        substitutes get_default_environment(), a small allowlist (PATH,
        HOME, SHELL, USER and a couple more) that deliberately excludes
        everything else, so a custom variable set in the parent never
        arrives. QR_OUTPUT_DIR therefore has to be added explicitly.

        That allowlist is used as the base rather than a copy of os.environ,
        which keeps the property that made env=None safe: the agent's
        INNKUBE_TOKEN, and anything else in the parent environment, is not
        handed to a subprocess that has no use for it.
        """
        env = get_default_environment()
        if self._output_dir:
            env["QR_OUTPUT_DIR"] = self._output_dir
        return env

    async def list_tools(self) -> list:
        async with AsyncExitStack() as stack:
            session = await self._open_session(stack)
            result = await session.list_tools()
            return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        async with AsyncExitStack() as stack:
            session = await self._open_session(stack)
            result = await session.call_tool(name, arguments)
            text_parts = [
                block.text for block in result.content
                if getattr(block, "type", None) == "text"
            ]
            return "\n".join(text_parts) if text_parts else "(empty result)"

    async def _open_session(self, stack: AsyncExitStack) -> ClientSession:
        server_params = StdioServerParameters(
            command=self._python_executable,
            args=[self._server_script],
            env=self._build_env(),
        )
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------
async def _smoke_test():
    client = QRMCPClient(workspace_root="./workspace")
    tools = await client.list_tools()
    print(f"Discovered {len(tools)} tool(s):")
    for t in tools:
        print(f"  - {t.name}: {t.description.splitlines()[0]}")

    result = await client.call_tool(
        "generate_qr_code",
        {"filename": "smoke-test.png", "payload_type": "url", "url": "https://example.com"},
    )
    print("\n--- Sample result ---")
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
