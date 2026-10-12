# src/llm/router.py
from src.config import ROUTER_LONG_QUERY_WORD_THRESHOLD

_MCP_DECISIONS = {"confluence", "git"}


class LLMRouter:
    def select(self, query: str, agent_decision: str) -> str:
        if agent_decision in _MCP_DECISIONS:
            return "gemini"
        if len(query.split()) > ROUTER_LONG_QUERY_WORD_THRESHOLD:
            return "gemini"
        return "groq"
