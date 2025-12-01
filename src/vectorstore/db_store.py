import hashlib
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from src.embedder.factory import create_embedder


class ChromaStore:
    """
    A wrapper around ChromaDB that supports embedder switching.
    Fully testable, no side effects, deterministic collection names.
    """

    def __init__(self, persist_dir: str, embedder_name: str = "hf", embedder_override: str | None = None):
        self.persist_dir = persist_dir

        # Use override if provided
        self.embedder_name = embedder_override or embedder_name

        # Load embedder (must support list[str] input)
        self.embedder = create_embedder(self.embedder_name)

        # Init chroma client
        self.client = PersistentClient(path=persist_dir)

        # Each embedder has its own collection namespace
        self.collection = self._get_or_create_collection()

    def _collection_name(self) -> str:
        """
        Deterministically names the collection based on the embedder.
        Example: linkedin_hf_abc123
        """
        key = f"linkedin_{self.embedder_name}"
        hashed = hashlib.sha1(key.encode()).hexdigest()[:8]
        return f"{key}_{hashed}"

    def _get_or_create_collection(self):
        name = self._collection_name()
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # -------------------------
    # Public API
    # -------------------------

    def add(self, ids, documents, embeddings=None, metadatas=None):
        """
        Add documents to the Chroma collection.
        If embeddings are provided, use them directly; otherwise, generate via embedder.
        """
        if embeddings is None:
            embeddings = self.embedder.embed(documents)
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


    def search(self, query_text: str, n_results: int = 5):
        query_vec = self.embedder.embed([query_text])[0]
        return self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
        )

    def reset(self):
        """Drops the collection for testing."""
        name = self._collection_name()
        self.client.delete_collection(name)
