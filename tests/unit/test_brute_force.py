"""
Tests for app.indexes.brute_force — exact nearest-neighbor search.
"""

import numpy as np
import pytest

from app.core.vector_store import VectorStore
from app.core.distance import normalize
from app.core.types import SearchResult, IndexState
from app.core.exceptions import IndexNotBuiltError, InvalidSearchParameterError
from app.indexes.brute_force import BruteForceIndex


@pytest.fixture
def brute_setup():
    """Create a store with 20 vectors of dim 8, and a built brute-force index."""
    rng = np.random.default_rng(42)
    store = VectorStore(dimension=8)
    vecs = rng.standard_normal((20, 8)).astype(np.float32)
    ids = [f"vec_{i}" for i in range(20)]
    store.bulk_insert(ids, vecs)

    index = BruteForceIndex()
    index.build(store.vectors, store.ids, store.active_mask)
    return store, index


class TestBruteForceIndex:

    def test_state_after_build(self, brute_setup):
        _, index = brute_setup
        assert index.state == IndexState.READY

    def test_state_before_build(self):
        index = BruteForceIndex()
        assert index.state == IndexState.EMPTY

    def test_search_not_built_raises(self):
        index = BruteForceIndex()
        query = np.random.randn(8).astype(np.float32)
        with pytest.raises(IndexNotBuiltError):
            index.search(query, k=5)

    def test_exact_nearest_neighbor(self, brute_setup):
        store, index = brute_setup
        # Query with one of the stored vectors — should find itself as top-1
        query = store.vectors[5]  # already normalized
        results = index.search(query, k=1)
        assert len(results) == 1
        assert results[0].id == "vec_5"
        assert np.isclose(results[0].score, 1.0, atol=1e-4)

    def test_top_k_correctness(self, brute_setup):
        _, index = brute_setup
        query = np.random.default_rng(99).standard_normal(8).astype(np.float32)
        results = index.search(query, k=5)
        assert len(results) == 5
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_have_valid_ids(self, brute_setup):
        store, index = brute_setup
        query = np.random.default_rng(99).standard_normal(8).astype(np.float32)
        results = index.search(query, k=10)
        valid_ids = set(store.ids)
        for r in results:
            assert r.id in valid_ids

    def test_deleted_vectors_ignored(self, brute_setup):
        store, index = brute_setup
        store.delete("vec_5")
        # Re-build is not needed for brute force — it reads active_mask directly
        query = store.vectors[5]
        results = index.search(query, k=5)
        result_ids = [r.id for r in results]
        assert "vec_5" not in result_ids

    def test_result_ordering(self, brute_setup):
        _, index = brute_setup
        query = np.random.default_rng(7).standard_normal(8).astype(np.float32)
        results = index.search(query, k=10)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_invalid_k_raises(self, brute_setup):
        _, index = brute_setup
        query = np.random.randn(8).astype(np.float32)
        with pytest.raises(InvalidSearchParameterError):
            index.search(query, k=0)

    def test_k_larger_than_n(self, brute_setup):
        _, index = brute_setup
        query = np.random.randn(8).astype(np.float32)
        results = index.search(query, k=100)
        assert len(results) == 20  # clamped to available vectors

    def test_search_returns_search_result_type(self, brute_setup):
        _, index = brute_setup
        query = np.random.randn(8).astype(np.float32)
        results = index.search(query, k=3)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_stats(self, brute_setup):
        _, index = brute_setup
        s = index.stats()
        assert s["name"] == "brute"
        assert s["state"] == "ready"
        assert s["total_vectors"] == 20
        assert s["active_vectors"] == 20
