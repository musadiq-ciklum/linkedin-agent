# src/llm/gemini.py
from typing import Generator
from src.llm.base import LLMClient, LLMResponse
from src.config import load_gemini


class GeminiLLMClient(LLMClient):

    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name
        self.model = load_gemini(model_name)

    def generate(self, prompt: str) -> LLMResponse:
        response = self.model.generate_content(prompt)

        # Gemini may return no parts → response.text raises ValueError
        try:
            text = response.text
        except Exception:
            text = "I could not find this information in the knowledge base."

        return LLMResponse(
            text=text,
            model=self.model_name,
            usage=None,
        )

    def stream(self, prompt: str) -> Generator[str, None, None]:
        response = self.model.generate_content(prompt, stream=True)
        for chunk in response:
            try:
                if chunk.text:
                    yield chunk.text
            except Exception:
                pass
