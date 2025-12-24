# src/llm/fake.py
from src.llm.base import LLMClient, LLMResponse

class FakeLLMClient(LLMClient):
    """
    Deterministic fake LLM used for tests.
    """

    def __init__(self, answer: str = "FAKE ANSWER"):
        self.answer = answer

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text=self.answer,
            model="fake-llm",
            usage=None,
        )
