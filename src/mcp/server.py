# src/mcp/server.py
#
# MCP Server exposing all configured knowledge sources as callable tools.
# Run standalone for Claude Desktop / Claude API:
#   fastmcp run src/mcp/server.py
#
import json

from fastmcp import FastMCP

from src.config import (
    CONFLUENCE_URL,
    CONFLUENCE_USER,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY,
)

mcp = FastMCP("LinkedIn RAG")

# ── Confluence tools (registered only when credentials are configured) ─────────

if CONFLUENCE_URL and CONFLUENCE_USER and CONFLUENCE_API_TOKEN:
    from src.mcp.confluence import ConfluenceSource

    def _confluence() -> ConfluenceSource:
        return ConfluenceSource(
            url=CONFLUENCE_URL,
            user=CONFLUENCE_USER,
            api_token=CONFLUENCE_API_TOKEN,
            space_key=CONFLUENCE_SPACE_KEY,
        )

    @mcp.tool()
    def search_confluence(query: str) -> str:
        """Search Confluence space for content relevant to the query. Returns a JSON list of results."""
        results = _confluence().search(query)
        return json.dumps([
            {"page_id": r.doc_id, "title": r.metadata["title"], "text": r.text}
            for r in results
        ])

    @mcp.tool()
    def get_confluence_page(page_id: str) -> str:
        """Fetch the full content of a Confluence page by its ID."""
        return _confluence().fetch_page(page_id).text


if __name__ == "__main__":
    mcp.run()
