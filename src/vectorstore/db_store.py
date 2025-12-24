# src/vectorstore/db_store.py

import hashlib
from chromadb import PersistentClient
from src.embedder.factory import create_embedder


class ChromaEmbeddingWrapper:
    """
    Adapter for Chroma 0.4.16+ embedding function API.
    Ensures the embedder receives full strings, not characters.
    """

    def __init__(self, embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]

        # embedder.embed() returns List[List[float]]
        return self._embedder.embed(input)

    def embed(self, input: list[str]):
        return self.__call__(input)

    def name(self):
        return f"custom_{self._embedder.__class__.__name__}"


class ChromaStore:
    """
    A clean, correct wrapper around ChromaDB for storing documents
    with a selected embedder.
    """

    def __init__(self, persist_dir: str, embedder_name="minilm", embedder_override=None):
        self.persist_dir = persist_dir
        self.embedder_name = embedder_override or embedder_name
        self.embedder = create_embedder(self.embedder_name)

        self.client = PersistentClient(path=persist_dir)
        self.collection = self._get_or_create_collection()

        print("Using collection:", self.collection.name)
        print("Docs count:", self.collection.count())

    def _collection_name(self):
        key = f"linkedin_{self.embedder_name}"
        hashed = hashlib.sha1(key.encode()).hexdigest()[:8]
        return f"{key}_{hashed}"

    def _get_or_create_collection(self):
        name = self._collection_name()
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=ChromaEmbeddingWrapper(self.embedder),
        )

    # -------------------------------------------------------------

    def add(self, ids, documents, embeddings=None, metadatas=None):
        if embeddings is None:
            embeddings = self.embedder.embed(documents)  # Correct full-doc embedding

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
        """Drops and recreates the collection."""
        name = self._collection_name()
        self.client.delete_collection(name)
        self.collection = self._get_or_create_collection()
