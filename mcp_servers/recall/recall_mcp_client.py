"""
recall_mcp_client.py — thin async client for recall_server.py.

Matches the same per-call connect/close pattern as ocr_mcp_client.py: opens a fresh
stdio connection for each list_tools()/call_tool() call, so it plugs into ToolRegistry the
same way (asyncio.run(client.list_tools()) / asyncio.run(client.call_tool(...))),
no persistent-session lifecycle for the caller to manage.
"""

import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = str(Path(__file__).parent / "recall_server.py")


class RecallMCPClient:
    """Self-hosted CPSC recall MCP client. Spawns recall_server.py fresh
    for each call -- no OAuth, no login, identical results for anyone
    running the code."""

    def __init__(self, server_script: str = SERVER_SCRIPT, python_executable: str = sys.executable):
        self._server_script = server_script
        self._python_executable = python_executable

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
            env=None,
        )
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------
async def _smoke_test():
    client = RecallMCPClient()
    tools = await client.list_tools()
    print(f"Discovered {len(tools)} tool(s):")
    for t in tools:
        print(f"  - {t.name}: {t.description.splitlines()[0]}")

    result = await client.call_tool("check_product_recall", {"product_name": "blender", "max_results": 2})
    print("\n--- Sample result ---")
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
