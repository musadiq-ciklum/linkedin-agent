# src/mcp/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SourceDocument:
    text: str
    doc_id: str
    metadata: dict = field(default_factory=dict)


class MCPSource(ABC):
    """Abstract base for MCP data sources (Confluence, Git, etc.)."""

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        """Search the source for content relevant to the query."""
        ...
