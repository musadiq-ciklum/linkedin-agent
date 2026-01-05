# tests/test_ask.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_ask_endpoint_basic():
    response = client.post(
        "/ask",
        json={"query": "What is precision@k?"}
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "contexts" in data
    assert isinstance(data["contexts"], list)
