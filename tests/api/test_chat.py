"""
Tests for RAG chat API endpoint (POST /chat) with Gemini LLM integration.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.config import VECTOR_DIM
from app.main import app


@pytest.fixture
def client_with_text_data():
    with TestClient(app) as c:
        # Insert test vectors with text metadata
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((30, VECTOR_DIM)).astype(np.float32)
        items = [
            {
                "id": f"text_{i}",
                "vector": vectors[i].tolist(),
                "metadata": {
                    "text": f"Sample statement {i} about web development and fast api framework.",
                    "category": "Web Dev"
                }
            }
            for i in range(30)
        ]
        c.post("/vectors/bulk", json={"vectors": items})
        c.post("/rebuild/brute")
        c.post("/rebuild/hnsw")
        yield c


def test_chat_endpoint_brute(client_with_text_data):
    client = client_with_text_data
    response = client.post("/chat", json={
        "query": "express js framework for node backend",
        "index": "brute",
        "k": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "express js framework for node backend"
    assert data["index_used"] == "brute"
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert len(data["sources"]) == 3
    assert "retrieval_latency_ms" in data
    assert "llm_latency_ms" in data
    assert "model_used" in data


def test_chat_endpoint_hnsw(client_with_text_data):
    client = client_with_text_data
    response = client.post("/chat", json={
        "query": "what is machine learning",
        "index": "hnsw",
        "k": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert data["index_used"] == "hnsw"
    assert len(data["sources"]) == 2


def test_chat_endpoint_custom_prompt(client_with_text_data):
    client = client_with_text_data
    response = client.post("/chat", json={
        "query": "explain fastapi",
        "system_prompt": "You are a concise expert technical assistant.",
        "temperature": 0.3,
        "k": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 1


def test_chat_endpoint_invalid_index(client_with_text_data):
    client = client_with_text_data
    response = client.post("/chat", json={
        "query": "test query",
        "index": "nonexistent_index",
        "k": 3
    })
    assert response.status_code == 400
