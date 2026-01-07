# tests/test_embedding_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_embedding_api():
    res = client.post(
        "/embedding",
        json={"text": "Hello world"},
    )

    assert res.status_code == 200

    body = res.json()
    assert isinstance(body["embedding"], list)
    assert body["dimensions"] == len(body["embedding"])
    assert body["dimensions"] > 0
    assert "model" in body


def test_embedding_empty_text():
    res = client.post("/embedding", json={"text": ""})
    assert res.status_code == 422
