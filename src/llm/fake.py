# src/llm/fake.py
from typing import Generator
from src.llm.base import LLMClient, LLMResponse

class FakeLLMClient(LLMClient):
    """Deterministic fake LLM used for tests."""

    def __init__(self, answer: str = "FAKE ANSWER"):
        self.answer = answer

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text=self.answer,
            model="fake-llm",
            usage=None,
        )

    def stream(self, prompt: str) -> Generator[str, None, None]:
        yield from self.answer.split()
