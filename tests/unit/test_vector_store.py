"""
Tests for app.core.vector_store — in-memory vector storage.
"""

import numpy as np
import pytest
import tempfile
import os

from app.core.vector_store import VectorStore
from app.core.exceptions import (
    DimensionMismatchError,
    DuplicateVectorError,
    VectorNotFoundError,
)


@pytest.fixture
def store():
    """Fresh VectorStore with dimension 4."""
    return VectorStore(dimension=4)


@pytest.fixture
def populated_store():
    """VectorStore with 5 vectors already inserted."""
    s = VectorStore(dimension=4)
    for i in range(5):
        vec = np.random.default_rng(i).random(4).astype(np.float32)
        s.insert(f"vec_{i}", vec, {"idx": i})
    return s


class TestInsert:
    def test_insert_single(self, store):
        store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        assert store.count() == 1

    def test_insert_normalizes(self, store):
        store.insert("v1", [3.0, 4.0, 0.0, 0.0])
        vec = store.get("v1")["vector"]
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_insert_with_metadata(self, store):
        store.insert("v1", [1.0, 0.0, 0.0, 0.0], {"text": "hello"})
        data = store.get("v1")
        assert data["metadata"]["text"] == "hello"

    def test_duplicate_id_raises(self, store):
        store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(DuplicateVectorError):
            store.insert("v1", [0.0, 1.0, 0.0, 0.0])

    def test_wrong_dimension_raises(self, store):
        with pytest.raises(DimensionMismatchError):
            store.insert("v1", [1.0, 0.0])  # dim 2 != 4


class TestBulkInsert:
    def test_bulk_insert(self, store):
        vecs = np.random.randn(10, 4).astype(np.float32)
        ids = [f"vec_{i}" for i in range(10)]
        store.bulk_insert(ids, vecs)
        assert store.count() == 10

    def test_bulk_normalizes(self, store):
        vecs = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        store.bulk_insert(["v1"], vecs)
        norm = np.linalg.norm(store.vectors[0])
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_bulk_duplicate_raises(self, store):
        store.insert("v1", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(DuplicateVectorError):
            store.bulk_insert(["v1"], np.ones((1, 4), dtype=np.float32))

    def test_bulk_wrong_dim_raises(self, store):
        with pytest.raises(DimensionMismatchError):
            store.bulk_insert(["v1"], np.ones((1, 3), dtype=np.float32))


class TestRetrieve:
    def test_get_existing(self, populated_store):
        data = populated_store.get("vec_0")
        assert data["id"] == "vec_0"
        assert data["active"] is True
        assert data["vector"].shape == (4,)

    def test_get_nonexistent_raises(self, store):
        with pytest.raises(VectorNotFoundError):
            store.get("nonexistent")


class TestDelete:
    def test_delete_marks_inactive(self, populated_store):
        populated_store.delete("vec_2")
        data = populated_store.get("vec_2")
        assert data["active"] is False

    def test_delete_reduces_active_count(self, populated_store):
        assert populated_store.count() == 5
        populated_store.delete("vec_0")
        assert populated_store.count() == 4

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(VectorNotFoundError):
            store.delete("nonexistent")

    def test_active_vectors_exclude_deleted(self, populated_store):
        populated_store.delete("vec_1")
        active_ids = populated_store.get_active_ids()
        assert "vec_1" not in active_ids
        assert len(active_ids) == 4


class TestAccessors:
    def test_count_empty(self, store):
        assert store.count() == 0

    def test_total_count(self, populated_store):
        populated_store.delete("vec_0")
        assert populated_store.total_count() == 5
        assert populated_store.count() == 4

    def test_is_empty(self, store):
        assert store.is_empty()

    def test_active_vectors_shape(self, populated_store):
        vecs = populated_store.get_active_vectors()
        assert vecs.shape == (5, 4)

    def test_active_ids_length(self, populated_store):
        ids = populated_store.get_active_ids()
        assert len(ids) == 5


class TestPersistence:
    def test_save_and_load(self, populated_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            populated_store.save(tmpdir)

            loaded = VectorStore(dimension=4)
            loaded.load(tmpdir)

            assert loaded.count() == populated_store.count()
            assert loaded.total_count() == populated_store.total_count()
            np.testing.assert_allclose(
                loaded.vectors, populated_store.vectors, atol=1e-6
            )
            assert list(loaded.ids) == list(populated_store.ids)

    def test_save_load_preserves_deletion(self, populated_store):
        populated_store.delete("vec_2")
        with tempfile.TemporaryDirectory() as tmpdir:
            populated_store.save(tmpdir)
            loaded = VectorStore(dimension=4)
            loaded.load(tmpdir)
            assert loaded.count() == 4
            assert loaded.get("vec_2")["active"] is False
