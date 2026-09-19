# tests/test_confluence_source.py
from unittest.mock import patch, MagicMock

from src.mcp.confluence import ConfluenceSource, _html_to_text
from src.mcp.base import SourceDocument


def _make_source():
    return ConfluenceSource(
        url="https://example.atlassian.net",
        user="user@example.com",
        api_token="token",
        space_key="TEST",
    )


def _mock_response(results: list) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = {"results": results}
    mock.raise_for_status.return_value = None
    return mock


def _page_result(page_id="123", title="My Page", html="<p>Some content here.</p>"):
    return {
        "id": page_id,
        "title": title,
        "body": {"storage": {"value": html}},
    }


# ── _html_to_text ─────────────────────────────────────────────────────────────

def test_html_to_text_strips_html_tags():
    html = "<h1>Title</h1><p>Hello world.</p>"
    result = _html_to_text(html)
    assert "Title" in result
    assert "Hello world." in result
    assert "<h1>" not in result
    assert "<p>" not in result


def test_html_to_text_handles_empty_string():
    assert _html_to_text("") == ""


# ── ConfluenceSource.search ───────────────────────────────────────────────────

def test_search_returns_source_documents():
    source = _make_source()
    with patch("requests.get", return_value=_mock_response([_page_result()])):
        results = source.search("authentication")
    assert len(results) == 1
    assert isinstance(results[0], SourceDocument)


def test_search_uses_page_id_as_doc_id():
    source = _make_source()
    with patch("requests.get", return_value=_mock_response([_page_result(page_id="abc123")])):
        results = source.search("test")
    assert results[0].doc_id == "abc123"


def test_search_sets_correct_metadata():
    source = _make_source()
    with patch("requests.get", return_value=_mock_response([_page_result(page_id="99", title="Setup Guide")])):
        results = source.search("setup")
    meta = results[0].metadata
    assert meta["source"] == "confluence"
    assert meta["page_id"] == "99"
    assert meta["title"] == "Setup Guide"
    assert meta["space_key"] == "TEST"


def test_search_returns_empty_list_when_no_results():
    source = _make_source()
    with patch("requests.get", return_value=_mock_response([])):
        results = source.search("nothing")
    assert results == []
