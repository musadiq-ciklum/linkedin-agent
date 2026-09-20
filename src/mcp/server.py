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
    GIT_REPO_URL,
    GIT_LOCAL_PATH,
    GIT_TOKEN,
    GIT_EXTENSIONS,
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


# ── Git tools (registered only when GIT_REPO_URL is configured) ───────────────

if GIT_REPO_URL:
    from src.mcp.git_source import GitSource

    _git_source: GitSource | None = None

    def _get_git_source() -> GitSource:
        global _git_source
        if _git_source is None:
            extensions = [e.strip() for e in GIT_EXTENSIONS.split(",") if e.strip()]
            _git_source = GitSource(
                repo_url=GIT_REPO_URL,
                local_path=GIT_LOCAL_PATH,
                token=GIT_TOKEN or None,
                file_extensions=extensions or None,
            )
            _git_source.initialize()
        return _git_source

    @mcp.tool()
    def search_git(query: str) -> str:
        """Search Git repository files for content relevant to the query. Returns a JSON list of results."""
        results = _get_git_source().search(query)
        return json.dumps([
            {
                "doc_id": r.doc_id,
                "file_path": r.metadata.get("file_path", ""),
                "repo": r.metadata.get("repo", ""),
                "text": r.text,
            }
            for r in results
        ])


if __name__ == "__main__":
    mcp.run()
