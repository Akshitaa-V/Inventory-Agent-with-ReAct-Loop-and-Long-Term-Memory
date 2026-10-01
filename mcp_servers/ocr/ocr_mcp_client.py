"""
ocr_mcp_client.py — thin async client for ocr_server.py.

Matches the same interface shape as recall_mcp_client.py: list_tools() and
call_tool() are each self-contained (open a stdio connection, do the work,
close it) rather than holding a long-lived session. That's what lets
ToolRegistry call them with plain `asyncio.run(...)` with no extra
connection-lifecycle management.
"""

import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = str(Path(__file__).parent / "ocr_server.py")


class OCRMCPClient:
    """Self-hosted OCR MCP client. Spawns ocr_server.py fresh for each call."""

    def __init__(self, server_script: str = SERVER_SCRIPT, python_executable: str = sys.executable):
        self._server_script = server_script
        self._python_executable = python_executable

    async def list_tools(self) -> list:
        """Returns the MCP Tool objects the OCR server exposes (name,
        description, inputSchema) -- the shape ToolRegistry reads off any
        configured MCP client's list_tools()."""
        async with AsyncExitStack() as stack:
            session = await self._open_session(stack)
            result = await session.list_tools()
            return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Calls one of the OCR server's tools and returns its text result."""
        async with AsyncExitStack() as stack:
            session = await self._open_session(stack)
            result = await session.call_tool(name, arguments)
            text_parts = [
                block.text for block in result.content
                if getattr(block, "type", None) == "text"
            ]
            return "\n".join(text_parts) if text_parts else "(empty OCR result)"

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
    client = OCRMCPClient()
    tools = await client.list_tools()
    print(f"Discovered {len(tools)} tools:")
    for t in tools:
        print(f"  - {t.name}: {t.description.splitlines()[0]}")

    langs = await client.call_tool("ocr_list_languages", {})
    print(f"\nAvailable languages: {langs}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
