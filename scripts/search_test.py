# scripts/search_test.py
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
from src.search.semantic_search import semantic_search
from src.vectorstore.db_store import ChromaStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--persist_dir", type=str, default="data/chroma_test", help="Chroma DB folder")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    # ---- Load store ----
    store = ChromaStore(persist_dir=args.persist_dir, embedder_override="minilm")

    print(f"Collection: {store.collection.name}")
    print(f"Docs count: {store.collection.count()}\n")

    # ---- Run semantic search (vector search + rerank) ----
    results = semantic_search(
        query=args.query,
        store=store,
        top_k=args.top_k,
    )

    if not results:
        print("No results found.")
        return

    # ---- Print results ----
    print("Search Results:\n")
    for idx, (doc, score) in enumerate(results, start=1):
        print(f"{idx}. Score: {score:.3f}")
        print(doc.strip())
        print("-" * 40)


if __name__ == "__main__":
    main()
