# src/mcp/git_source.py
import re
from pathlib import Path
from typing import Optional

from git import Repo, GitCommandError

from src.mcp.base import MCPSource, SourceDocument
from src.data_prep.chunker import chunk_text
from src.config import GIT_LOCAL_PATH

DEFAULT_EXTENSIONS = {".md", ".txt", ".rst"}
_CHUNK_SIZE = 600


class GitSource(MCPSource):
    """
    MCP source that clones or pulls a Git repository and searches file
    contents for relevant chunks on each query call. Always pulls latest
    before searching so content is never stale.
    """

    def __init__(
        self,
        repo_url: str,
        local_path: str = GIT_LOCAL_PATH,
        token: Optional[str] = None,
        file_extensions: Optional[list[str]] = None,
    ):
        self._repo_url = self._inject_token(repo_url, token)
        self._local_path = Path(local_path)
        self._extensions = set(file_extensions or DEFAULT_EXTENSIONS)
        self._repo: Optional[Repo] = None
        self._repo_name = Path(repo_url.rstrip("/").split("/")[-1]).stem

    @staticmethod
    def _inject_token(repo_url: str, token: Optional[str]) -> str:
        if token and repo_url.startswith("https://"):
            return repo_url.replace("https://", f"https://{token}@", 1)
        return repo_url

    def _clone_or_pull(self) -> Repo:
        self._local_path.mkdir(parents=True, exist_ok=True)
        if (self._local_path / ".git").exists():
            repo = Repo(str(self._local_path))
            repo.remotes.origin.pull()
        else:
            repo = Repo.clone_from(self._repo_url, str(self._local_path))
        self._repo = repo
        return repo

    def initialize(self) -> None:
        """Clone or pull the repository. Call once at startup."""
        self._clone_or_pull()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def _score_chunk(self, query: str, chunk: str) -> float:
        qtok = self._tokenize(query)
        dtok = self._tokenize(chunk)
        return float(len(qtok & dtok)) / (len(qtok) or 1.0)

    def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        self._clone_or_pull()

        scored: list[tuple[float, str, str, int]] = []

        for filepath in self._local_path.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix not in self._extensions:
                continue

            rel_path = str(filepath.relative_to(self._local_path))
            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue

            if not text:
                continue

            chunks = chunk_text(text, chunk_size=_CHUNK_SIZE)
            for i, chunk in enumerate(chunks):
                score = self._score_chunk(query, chunk)
                if score > 0:
                    scored.append((score, chunk, rel_path, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk, rel_path, chunk_idx in scored[:limit]:
            doc_id = f"git::{self._repo_name}::{rel_path}::{chunk_idx}"
            results.append(SourceDocument(
                text=chunk,
                doc_id=doc_id,
                metadata={
                    "source": "git",
                    "repo": self._repo_name,
                    "file_path": rel_path,
                },
            ))

        return results
