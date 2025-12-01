import tempfile
from src.vectorstore.db_store import ChromaStore
from src.embedder.fake import FakeEmbedder


def test_chromastore_uses_fake_embedder(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaStore(persist_dir=tmpdir, embedder_override="fake")
    assert isinstance(store.embedder, FakeEmbedder)
