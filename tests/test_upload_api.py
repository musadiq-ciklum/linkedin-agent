# tests/test_upload_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_upload_txt():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"RAG systems use vector databases")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] > 0
