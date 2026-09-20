# tests/test_git_source.py
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.mcp.git_source import GitSource, DEFAULT_EXTENSIONS
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

    assert len(results) <= 3


def test_search_ignores_non_matching_extensions(tmp_path):
    local = tmp_path / "repo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "script.py").write_text("def python_function(): pass")
    (local / "README.md").write_text("Python documentation for the project.")

    source = GitSource(
        repo_url="https://github.com/example/repo.git",
        local_path=str(local),
    )
    mock_repo = MagicMock()
    with patch("src.mcp.git_source.Repo", return_value=mock_repo):
        results = source.search("python function")

    for r in results:
        assert not r.metadata["file_path"].endswith(".py")


# ── default extensions ─────────────────────────────────────────────────────────

def test_default_extensions_include_common_doc_formats():
    assert ".md" in DEFAULT_EXTENSIONS
    assert ".txt" in DEFAULT_EXTENSIONS
    assert ".rst" in DEFAULT_EXTENSIONS


def test_default_extensions_exclude_code_files():
    assert ".py" not in DEFAULT_EXTENSIONS
    assert ".js" not in DEFAULT_EXTENSIONS
