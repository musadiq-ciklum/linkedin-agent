# tests/test_git_source.py
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.mcp.git_source import GitSource
from src.mcp.base import SourceDocument


def _make_source(tmp_path: Path) -> GitSource:
    return GitSource(
        repo_url="https://github.com/example/repo.git",
        local_path=str(tmp_path / "repo"),
    )


# ── _inject_token ─────────────────────────────────────────────────────────────

def test_inject_token_inserts_credentials_into_https_url():
    result = GitSource._inject_token("https://github.com/org/repo.git", "mytoken")
    assert result == "https://mytoken@github.com/org/repo.git"


def test_inject_token_leaves_url_unchanged_when_no_token():
    url = "https://github.com/org/repo.git"
    assert GitSource._inject_token(url, None) == url


def test_inject_token_leaves_ssh_url_unchanged():
    url = "git@github.com:org/repo.git"
    assert GitSource._inject_token(url, "token") == url


# ── _clone_or_pull ────────────────────────────────────────────────────────────

def test_clone_is_called_when_no_git_dir_exists(tmp_path):
    source = _make_source(tmp_path)
    with patch("src.mcp.git_source.Repo.clone_from") as mock_clone:
        mock_clone.return_value = MagicMock()
        source._clone_or_pull()
    mock_clone.assert_called_once()


def test_pull_is_called_when_git_dir_exists(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    mock_repo = MagicMock()
    source = GitSource(
        repo_url="https://github.com/example/repo.git",
        local_path=str(local),
    )
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        source._clone_or_pull()
    mock_repo.remotes.origin.pull.assert_called_once()


# ── search ────────────────────────────────────────────────────────────────────

def test_search_returns_source_documents_with_correct_metadata(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "README.md").write_text("This project uses Python and FastAPI for building a RAG pipeline.")

    source = GitSource(
        repo_url="https://github.com/example/repo.git",
        local_path=str(local),
    )
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        results = source.search("Python FastAPI")

    assert len(results) >= 1
    assert isinstance(results[0], SourceDocument)
    assert results[0].metadata["source"] == "git"
    assert results[0].metadata["repo"] == "repo"
    assert results[0].metadata["file_path"] == "README.md"


def test_search_returns_empty_list_when_no_files_match_query(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "README.md").write_text("This file is about databases and SQL.")

    source = GitSource(
        repo_url="https://github.com/example/repo.git",
        local_path=str(local),
    )
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        results = source.search("quantum physics")

    assert results == []


def test_search_limits_results_to_limit_param(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    for i in range(10):
        (local / f"file{i}.md").write_text(f"Python FastAPI project number {i} with RAG pipeline.")

    source = GitSource(
        repo_url="https://github.com/example/repo.git",
        local_path=str(local),
    )
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        results = source.search("Python FastAPI", limit=3)

    assert len(results) == 3


def test_search_ignores_non_matching_extensions(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "config.json").write_text('{"python": "function", "project": "setup"}')
    (local / "README.md").write_text("Python documentation for the project.")

    source = GitSource(
        repo_url="https://github.com/example/repo.git",
        local_path=str(local),
    )
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        results = source.search("python function")

    for r in results:
        assert not r.metadata["file_path"].endswith(".json")


# ── _extract_filename ─────────────────────────────────────────────────────────

def test_extract_filename_detects_file_path_in_query():
    assert GitSource._extract_filename("what does src/mcp/confluence.py contain") == "confluence.py"


def test_extract_filename_detects_bare_filename():
    assert GitSource._extract_filename("explain confluence.py") == "confluence.py"


def test_extract_filename_returns_none_for_plain_query():
    assert GitSource._extract_filename("how does the RAG pipeline work") is None


# ── _fetch_file ───────────────────────────────────────────────────────────────

def test_fetch_file_returns_source_document_for_existing_file(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "README.md").write_text("# My Project\nThis is the readme.")

    source = GitSource(repo_url="https://github.com/example/repo.git", local_path=str(local))
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        source._clone_or_pull()
        result = source._fetch_file("README.md")

    assert result is not None
    assert isinstance(result, SourceDocument)
    assert "My Project" in result.text
    assert result.metadata["file_path"] == "README.md"


def test_fetch_file_returns_none_for_missing_file(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()

    source = GitSource(repo_url="https://github.com/example/repo.git", local_path=str(local))
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        source._clone_or_pull()
        result = source._fetch_file("nonexistent.py")

    assert result is None


def test_fetch_file_truncates_large_files(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "big.md").write_text("x" * 5000)

    source = GitSource(repo_url="https://github.com/example/repo.git", local_path=str(local))
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        source._clone_or_pull()
        result = source._fetch_file("big.md")

    assert result is not None
    assert "truncated" in result.text


# ── default extensions ─────────────────────────────────────────────────────────

def test_default_extensions_include_common_doc_and_code_formats():
    from src.config import GIT_EXTENSIONS
    extensions = {e.strip() for e in GIT_EXTENSIONS.split(",") if e.strip()}
    assert ".md" in extensions
    assert ".txt" in extensions
    assert ".rst" in extensions
    assert ".py" in extensions
