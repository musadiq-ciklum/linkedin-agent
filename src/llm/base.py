# src/llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Generator


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Optional[Dict] = None


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        ...

    @abstractmethod
    def stream(self, prompt: str) -> Generator[str, None, None]:
        ...
