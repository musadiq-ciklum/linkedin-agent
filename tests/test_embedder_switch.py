import os
import pytest
from src.embedder.factory import create_embedder
from src.embedder.fake import FakeEmbedder


def test_load_embedder_fake(monkeypatch):
    monkeypatch.setenv("EMBEDDER_MODE", "fake")
    embedder = load_embedder()
    assert isinstance(embedder, FakeEmbedder)


def test_load_embedder_override_fake(monkeypatch):
    monkeypatch.delenv("EMBEDDER_MODE", raising=False)
    embedder = load_embedder("fake")
    assert isinstance(embedder, FakeEmbedder)


def test_load_embedder_onnx_fallback(monkeypatch):
    """Force ONNX failure → ensures fallback to FakeEmbedder."""
    monkeypatch.setenv("EMBEDDER_MODE", "onnx")

    # Force ONNX embedding to fail by mocking import or constructor
    monkeypatch.setenv("EMBEDDER_MODEL", "non-existent-model")

    embedder = load_embedder()
    assert isinstance(embedder, FakeEmbedder)
