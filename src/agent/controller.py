# src/agent/controller.py

class AgentController:
    """
    Decides whether the agent should retrieve external context
    or generate a direct response.
    """

    def decide(self, query: str) -> str:
        q = query.lower()

        generative_triggers = [
            "write", "post", "linkedin", "announce",
            "describe yourself", "how you were built",
            "this project", "ai academy", "ciklum"
        ]

        if any(t in q for t in generative_triggers):
            return "generate"

        return "retrieve"
