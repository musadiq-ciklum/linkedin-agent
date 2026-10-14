# src/llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        ...

    @abstractmethod
    def stream(self, prompt: str) -> Generator[str, None, None]:
        ...
