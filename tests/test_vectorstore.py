# tests/test_vectorstore.py
from src.vectorstore.db_store import ChromaStore

def test_chroma_store_index_and_search(tmp_path):
    # Use a temporary directory for isolation
    persist_dir = tmp_path / "chroma"
    store = ChromaStore(persist_dir=str(persist_dir), embedder_override="minilm")

    sample_texts = [
        "This is a test",
        "Another test sentence"
    ]

    # Add documents
    store.add(
        ids=[f"doc{i}" for i in range(len(sample_texts))],
        documents=sample_texts
    )

    # Simple vector search
    results = store.search(sample_texts[0], n_results=1)

    # Verify results structure
    assert isinstance(results, dict)
    assert "documents" in results
    assert len(results["documents"][0]) >= 1
