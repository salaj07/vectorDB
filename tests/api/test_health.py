"""
Tests for API health, root, stats, and rebuild endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_endpoint(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "vectors" in data
    assert "total_vectors" in data
    assert "dimension" in data
    assert "indexes" in data


def test_rebuild_endpoint(client):
    response = client.post("/rebuild/brute")
    assert response.status_code == 200
    assert response.json()["status"] == "rebuilt"
