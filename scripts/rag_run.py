# scripts/rag_run.py
import sys
from pathlib import Path

# -----------------------------
# Ensure src/ is importable
# -----------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# -----------------------------
# Imports
# -----------------------------
from src.llm.gemini import GeminiLLMClient
from src.prompts.prompt_builder import PromptBuilder
from src.rag.pipeline import RAGPipeline

from src.search.semantic_search import semantic_search
from src.search.vector_search import get_store
from src.search.reranker import LocalSimpleRanker


class Reranker(LocalSimpleRanker):
    def rerank(self, query: str, docs):
        if not docs:
            return []

        texts = [d["text"] for d in docs]
        scored = self.score(query, texts)  # [(text, score)]

        # merge back into docs
        score_map = {text: score for text, score in scored}

        for d in docs:
            d["rerank_score"] = float(score_map.get(d["text"], 0.0))

        return docs

# -----------------------------
# Retriever wrapper
# -----------------------------
class Retriever:
    def __init__(self, store=None, top_k=5, min_score=0.2):
        self.store = store or get_store()
        self.top_k = top_k
        self.min_score = min_score

    def search(self, query: str, top_k: int = None):
        top_k = top_k or self.top_k
        results = semantic_search(query, store=self.store, top_k=top_k)

        docs = []
        for i, (text, score) in enumerate(results):
            if score < self.min_score:
                continue

            docs.append({
                "id": f"doc_{i}",
                "text": text,
                "score": float(score),
            })

        return docs


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/rag_run.py \"your question here\"")
        sys.exit(1)

    query = sys.argv[1]

    PROMPTS_DIR = ROOT / "src" / "prompts"

    retriever = Retriever(min_score=0.3)
    reranker = Reranker()

    prompt_builder = PromptBuilder(
        mode="agent",
        prompts_dir=PROMPTS_DIR,
    )

    llm = GeminiLLMClient()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm,
        prompt_builder=prompt_builder,
    )

    answer = pipeline.run(query)

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
