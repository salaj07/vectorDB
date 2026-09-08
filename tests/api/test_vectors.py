"""
Tests for vector CRUD API endpoints.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.config import VECTOR_DIM
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_insert_vector(client):
    vec = [0.1] * VECTOR_DIM
    response = client.post("/vectors", json={"id": "v1", "vector": vec, "metadata": {"name": "test"}})
    assert response.status_code == 201
    assert response.json() == {"id": "v1", "status": "inserted"}


def test_insert_duplicate_vector_fails(client):
    vec = [0.1] * VECTOR_DIM
    client.post("/vectors", json={"id": "v1", "vector": vec})
    response = client.post("/vectors", json={"id": "v1", "vector": vec})
    assert response.status_code == 409


def test_insert_dimension_mismatch_fails(client):
    vec = [0.1] * 5  # Wrong dimension (expecting 128)
    response = client.post("/vectors", json={"id": "v2", "vector": vec})
    assert response.status_code == 400


def test_get_vector(client):
    vec = [0.5] * VECTOR_DIM
    client.post("/vectors", json={"id": "v_get", "vector": vec, "metadata": {"tag": "findme"}})
    
    response = client.get("/vectors/v_get")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "v_get"
    assert len(data["vector"]) == VECTOR_DIM
    assert data["metadata"] == {"tag": "findme"}


def test_get_nonexistent_vector_fails(client):
    response = client.get("/vectors/nonexistent")
    assert response.status_code == 404


def test_delete_vector(client):
    vec = [0.2] * VECTOR_DIM
    client.post("/vectors", json={"id": "v_del", "vector": vec})

    response = client.delete("/vectors/v_del")
    assert response.status_code == 200
    assert response.json() == {"id": "v_del", "status": "deleted"}

    # Attempting to get deleted vector fails
    get_res = client.get("/vectors/v_del")
    assert get_res.status_code == 404


def test_bulk_insert(client):
    items = [
        {"id": f"bulk_{i}", "vector": [0.01 * i] * VECTOR_DIM, "metadata": {"idx": i}}
        for i in range(10)
    ]
    response = client.post("/vectors/bulk", json={"vectors": items})
    assert response.status_code == 201
    assert response.json() == {"inserted": 10, "status": "inserted"}
