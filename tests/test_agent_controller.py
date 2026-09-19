# tests/test_agent_controller.py
from unittest.mock import patch

from src.agent.controller import AgentController


def _controller():
    return AgentController()


# ── Confluence routing ────────────────────────────────────────────────────────

def test_returns_confluence_when_configured_and_query_matches():
    with patch("src.agent.controller.CONFLUENCE_URL", "https://example.atlassian.net"), \
         patch("src.agent.controller.CONFLUENCE_USER", "user@example.com"), \
         patch("src.agent.controller.CONFLUENCE_API_TOKEN", "token123"):
        assert _controller().decide("show me the docs") == "confluence"


def test_returns_retrieve_when_confluence_not_configured():
    with patch("src.agent.controller.CONFLUENCE_URL", ""), \
         patch("src.agent.controller.CONFLUENCE_USER", ""), \
         patch("src.agent.controller.CONFLUENCE_API_TOKEN", ""):
        assert _controller().decide("show me the docs") == "retrieve"


def test_returns_retrieve_when_confluence_url_missing():
    with patch("src.agent.controller.CONFLUENCE_URL", ""), \
         patch("src.agent.controller.CONFLUENCE_USER", "user@example.com"), \
         patch("src.agent.controller.CONFLUENCE_API_TOKEN", "token123"):
        assert _controller().decide("show me the wiki") == "retrieve"


# ── Generate routing ──────────────────────────────────────────────────────────

def test_returns_generate_for_linkedin_query():
    assert _controller().decide("write a linkedin post") == "generate"


def test_returns_generate_for_announce_query():
    assert _controller().decide("announce the project") == "generate"


# ── Retrieve routing ──────────────────────────────────────────────────────────

def test_returns_retrieve_for_generic_query():
    assert _controller().decide("what is retrieval augmented generation?") == "retrieve"
