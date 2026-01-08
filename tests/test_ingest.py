# tests/test_ingest.py
from src.utils import ingest_text

class DummyEmbedder:
    def embed(self, texts):
        return [[0.1] * 5 for _ in texts]

class DummyStore:
    def __init__(self):
        self.called = False

    def add(self, ids, documents, embeddings=None, metadatas=None):
        self.called = True
        assert len(ids) == len(documents)
        assert embeddings is not None
        assert metadatas is not None

def test_ingest_text_calls_store():
    embedder = DummyEmbedder()
    store = DummyStore()

    chunks = ingest_text(
        text="hello world " * 50,
        filename="test.txt",
        embedder=embedder,
        vector_store=store,
    )

    assert chunks > 0
    assert store.called
