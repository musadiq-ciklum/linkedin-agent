# src/llm/gemini.py
from src.llm.base import LLMClient, LLMResponse
from src.config import load_gemini


class GeminiLLMClient(LLMClient):
    """
    Gemini LLM client using google.generativeai.
    """

    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name
        self.model = load_gemini(model_name)

    def generate(self, prompt: str) -> LLMResponse:
        response = self.model.generate_content(prompt)

        # Gemini responses usually expose `.text`
        text = getattr(response, "text", str(response))

        return LLMResponse(
            text=text,
            model=self.model_name,
            usage=None,  # Gemini usage metadata can be added later
        )
