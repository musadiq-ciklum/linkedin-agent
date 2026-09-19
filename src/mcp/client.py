# src/mcp/client.py
#
# Sync wrapper around the FastMCP in-memory client.
# Lets non-async code (Streamlit, RAGPipeline) call MCP tools without a subprocess.
#
import asyncio

from fastmcp import Client

from src.mcp.server import mcp


def call_tool(tool_name: str, arguments: dict) -> str:
    """Call an MCP tool by name and return its text result synchronously."""
    async def _call():
        async with Client(mcp) as client:
            result = await client.call_tool(tool_name, arguments)
            return result.content[0].text

    return asyncio.run(_call())
