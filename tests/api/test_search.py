"""
Tests for search API endpoint across all index types.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.config import VECTOR_DIM
from app.main import app


@pytest.fixture
def client_with_data():
    with TestClient(app) as c:
        # Insert test vectors
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, VECTOR_DIM)).astype(np.float32)
        items = [
            {"id": f"doc_{i}", "vector": vectors[i].tolist()}
            for i in range(50)
        ]
        c.post("/vectors/bulk", json={"vectors": items})
        # Rebuild all indexes
        c.post("/rebuild/brute")
        c.post("/rebuild/ivf")
        c.post("/rebuild/hnsw")
        yield c, vectors[0].tolist()


def test_search_brute(client_with_data):
    client, query_vec = client_with_data
    response = client.post("/search", json={
        "query": query_vec,
        "k": 5,
        "index": "brute"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "brute"
    assert len(data["results"]) == 5
    assert data["results"][0]["id"] == "doc_0"
    assert pytest.approx(data["results"][0]["score"], abs=1e-5) == 1.0


def test_search_ivf(client_with_data):
    client, query_vec = client_with_data
    response = client.post("/search", json={
        "query": query_vec,
        "k": 5,
        "index": "ivf",
        "nprobe": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "ivf"
    assert len(data["results"]) <= 5


def test_search_hnsw(client_with_data):
    client, query_vec = client_with_data
    response = client.post("/search", json={
        "query": query_vec,
        "k": 5,
        "index": "hnsw",
        "ef_search": 20
    })
    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "hnsw"
    assert len(data["results"]) <= 5


def test_search_invalid_index_fails(client_with_data):
    client, query_vec = client_with_data
    response = client.post("/search", json={
        "query": query_vec,
        "k": 5,
        "index": "nonexistent"
    })
    assert response.status_code == 400


def test_search_dimension_mismatch_fails(client_with_data):
    client, _ = client_with_data
    response = client.post("/search", json={
        "query": [0.1] * 10,
        "k": 5,
        "index": "brute"
    })
    assert response.status_code == 400
