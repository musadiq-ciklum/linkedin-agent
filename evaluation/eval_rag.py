# evaluation/eval_rag.py
import sys
import os
import json
import time
import csv
from pathlib import Path

# -----------------------------
# Ensure project root is importable
# -----------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# -----------------------------
# Imports (metrics + pipeline)
# -----------------------------
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    keyword_overlap_score,
)
from evaluation.noop_reranker import NoOpReranker

from src.rag.pipeline import RAGPipeline
from src.llm.fake import FakeLLMClient
from src.prompts.prompt_builder import PromptBuilder

from src.search.semantic_search import semantic_search
from src.search.vector_search import get_store
from src.search.reranker import LocalSimpleRanker


# -----------------------------
# Reranker (same as rag_run.py)
# -----------------------------
class Reranker(LocalSimpleRanker):
    def rerank(self, query: str, docs):
        if not docs:
            return []

        texts = [d["text"] for d in docs]
        scored = self.score(query, texts)  # [(text, score)]

        score_map = {text: score for text, score in scored}

        for d in docs:
            d["rerank_score"] = float(score_map.get(d["text"], 0.0))

        return docs


# -----------------------------
# Retriever (same as rag_run.py)
# -----------------------------
class Retriever:
    def __init__(self, store=None, top_k=5, min_score=0.5):
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
# Dataset loader
# -----------------------------
def load_dataset(path: str):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Evaluation core
# -----------------------------
def evaluate_pipeline(pipeline, retriever, reranker, dataset, top_k=5):
    results = []

    for sample in dataset:
        query = sample["query"]
        gold_docs = sample.get("relevant_docs", [])
        expected_answer = sample.get("expected_answer", "")

        # -------- Retrieval timing --------
        t0 = time.perf_counter()
        retrieved_docs = retriever.search(query, top_k=top_k)
        t1 = time.perf_counter()

        # -------- Rerank timing --------
        reranked_docs = reranker.rerank(query, retrieved_docs)
        t2 = time.perf_counter()

        # -------- LLM timing --------
        answer = pipeline.run(query)
        t3 = time.perf_counter()

        retrieved_texts = [d["text"] for d in retrieved_docs]

        results.append({
            "query": query,
            "precision@k": precision_at_k(retrieved_texts, gold_docs, k=top_k),
            "recall@k": recall_at_k(retrieved_texts, gold_docs, k=top_k),
            "rag_quality": keyword_overlap_score(answer, expected_answer),
            "latency_retrieval_ms": (t1 - t0) * 1000,
            "latency_rerank_ms": (t2 - t1) * 1000,
            "latency_llm_ms": (t3 - t2) * 1000,
            "latency_total_ms": (t3 - t0) * 1000,
        })

    return results


# -----------------------------
# CSV export
# -----------------------------
def export_to_csv(results, path):
    if not results:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    DATASET_PATH = ROOT / "evaluation" / "data" / "sample_eval.json"
    PROMPTS_DIR = ROOT / "src" / "prompts"

    dataset = load_dataset(DATASET_PATH)

    # Shared components
    retriever = Retriever(min_score=0.3)
    prompt_builder = PromptBuilder(
        mode="agent",
        prompts_dir=PROMPTS_DIR,
    )

    # -----------------------------
    # Pipeline WITH reranking
    # -----------------------------
    reranker = Reranker()

    pipeline_with_rerank = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_client=FakeLLMClient(),
        prompt_builder=prompt_builder,
        top_k=5,
    )

    results_with = evaluate_pipeline(
        pipeline_with_rerank,
        retriever,
        reranker,
        dataset,
        top_k=5,
    )

    export_to_csv(
        results_with,
        ROOT / "evaluation" / "reports" / "report_with_rerank.csv",
    )

    # -----------------------------
    # Pipeline WITHOUT reranking
    # -----------------------------
    no_reranker = NoOpReranker()

    pipeline_without_rerank = RAGPipeline(
        retriever=retriever,
        reranker=no_reranker,
        llm_client=FakeLLMClient(),
        prompt_builder=prompt_builder,
        top_k=5,
    )

    results_without = evaluate_pipeline(
        pipeline_without_rerank,
        retriever,
        no_reranker,
        dataset,
        top_k=5,
    )

    export_to_csv(
        results_without,
        ROOT / "evaluation" / "reports" / "report_without_rerank.csv",
    )

    print("\n✅ Evaluation complete")
    print("CSV reports generated:")
    print(" - evaluation/reports/report_with_rerank.csv")
    print(" - evaluation/reports/report_without_rerank.csv")
