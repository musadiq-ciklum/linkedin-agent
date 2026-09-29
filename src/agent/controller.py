# src/agent/controller.py
from src.config import CONFLUENCE_URL, CONFLUENCE_USER, CONFLUENCE_API_TOKEN, GIT_REPO_URL

ROUTES = [
    {
        "name": "git",
        "triggers": ["git", "repo", "repository", "source code", "readme", "commit", "branch", "in the repo", "src/"],
        "configured": lambda: bool(GIT_REPO_URL),
    },
    {
        "name": "confluence",
        "triggers": ["confluence", "documentation", "docs", "wiki", "wiki page"],
        "configured": lambda: bool(CONFLUENCE_URL and CONFLUENCE_USER and CONFLUENCE_API_TOKEN),
    },
    {
        "name": "generate",
        "triggers": ["write", "post", "linkedin", "announce", "describe yourself", "how you were built", "this project", "ai academy", "ciklum"],
        "configured": lambda: True,
    },
]


class AgentController:
    """
    Decides whether the agent should retrieve external context
    or generate a direct response.
    """

    def decide(self, query: str) -> str:
        q = query.lower()
        for route in ROUTES:
            if route["configured"]() and any(t in q for t in route["triggers"]):
                return route["name"]
        return "retrieve"
