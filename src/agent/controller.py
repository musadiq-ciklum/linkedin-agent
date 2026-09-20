# src/agent/controller.py
from src.config import CONFLUENCE_URL, CONFLUENCE_USER, CONFLUENCE_API_TOKEN, GIT_REPO_URL


class AgentController:
    """
    Decides whether the agent should retrieve external context
    or generate a direct response.
    """

    def decide(self, query: str) -> str:
        q = query.lower()

        confluence_triggers = [
            "confluence", "documentation", "docs", "wiki", "wiki page",
        ]

        generative_triggers = [
            "write", "post", "linkedin", "announce",
            "describe yourself", "how you were built",
            "this project", "ai academy", "ciklum"
        ]

        git_triggers = [
            "git", "repo", "repository", "source code", "readme", "commit", "branch",
        ]

        confluence_configured = bool(CONFLUENCE_URL and CONFLUENCE_USER and CONFLUENCE_API_TOKEN)
        if confluence_configured and any(t in q for t in confluence_triggers):
            return "confluence"

        git_configured = bool(GIT_REPO_URL)
        if git_configured and any(t in q for t in git_triggers):
            return "git"

        if any(t in q for t in generative_triggers):
            return "generate"

        return "retrieve"
