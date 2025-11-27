# tests/test_vectorstore.py
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore


def test_embedding_model():
    model = EmbeddingModel()
    vec = model.embed_text("Hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0  # MiniLM embeddings ≈ 384 dims


def test_chroma_store_index_and_search(tmp_path):
    # Temporary isolated DB
    persist_dir = tmp_path / "chroma"
    store = ChromaStore(persist_directory=str(persist_dir))

    sample_texts = ["this is a test", "another test sentence"]
    model = EmbeddingModel()
    embeddings = model.embed_texts(sample_texts)

    metadata = [
        {"source": "test", "chunk_index": 0},
        {"source": "test", "chunk_index": 1}
    ]

    store.add_documents(sample_texts, embeddings, metadata)

    result = store.search(embeddings[0], n_results=1)

    assert "documents" in result
    assert len(result["documents"][0]) >= 1
