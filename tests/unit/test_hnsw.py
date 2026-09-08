"""
Tests for app.indexes.hnsw — HNSW approximate index.
"""

import numpy as np
import pytest

from app.core.vector_store import VectorStore
from app.core.distance import normalize
from app.core.types import SearchResult, IndexState
from app.core.exceptions import IndexNotBuiltError, InvalidSearchParameterError
from app.indexes.brute_force import BruteForceIndex
from app.indexes.hnsw import HNSWIndex


@pytest.fixture
def hnsw_setup():
    """
    Create a store with 200 clustered vectors of dim 16,
    and build both brute-force and HNSW indexes.
    """
    rng = np.random.default_rng(42)
    dim = 16
    n = 200
    store = VectorStore(dimension=dim)

    # Create 4 clusters of 50 vectors each
    centers = normalize(rng.standard_normal((4, dim)).astype(np.float32))
    all_vecs = []
    for c in centers:
        noise = rng.standard_normal((50, dim)).astype(np.float32) * 0.1
        all_vecs.append(c + noise)
    vecs = np.vstack(all_vecs)

    ids = [f"vec_{i}" for i in range(n)]
    store.bulk_insert(ids, vecs)

    brute = BruteForceIndex()
    brute.build(store.vectors, store.ids, store.active_mask)

    hnsw = HNSWIndex(m=16, ef_construction=100, ef_search=50)
    hnsw.build(store.vectors, store.ids, store.active_mask)

    return store, brute, hnsw


@pytest.fixture
def small_hnsw():
    """Very small graph for basic connectivity tests."""
    store = VectorStore(dimension=4)
    vecs = normalize(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0.9, 0.1, 0, 0],
        [0.1, 0.9, 0, 0],
    ], dtype=np.float32))
    ids = [f"v{i}" for i in range(5)]
    store.bulk_insert(ids, vecs)

    hnsw = HNSWIndex(m=4, ef_construction=10, ef_search=10)
    hnsw.build(store.vectors, store.ids, store.active_mask)
    return store, hnsw


class TestHNSWIndex:

    def test_state_after_build(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        assert hnsw.state == IndexState.READY

    def test_state_before_build(self):
        hnsw = HNSWIndex()
        assert hnsw.state == IndexState.EMPTY

    def test_search_not_built_raises(self):
        hnsw = HNSWIndex()
        with pytest.raises(IndexNotBuiltError):
            hnsw.search(np.random.randn(16).astype(np.float32))

    def test_all_nodes_inserted(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        assert len(hnsw.neighbors) == 200

    def test_entry_point_exists(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        assert hnsw.entry_point >= 0
        assert hnsw.max_level >= 0

    def test_search_returns_valid_ids(self, hnsw_setup):
        store, _, hnsw = hnsw_setup
        query = np.random.default_rng(99).standard_normal(16).astype(np.float32)
        results = hnsw.search(query, k=10)
        valid_ids = set(store.ids)
        for r in results:
            assert r.id in valid_ids

    def test_search_returns_correct_count(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        query = np.random.default_rng(99).standard_normal(16).astype(np.float32)
        results = hnsw.search(query, k=5)
        assert len(results) == 5

    def test_results_ordered_descending(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        query = np.random.default_rng(99).standard_normal(16).astype(np.float32)
        results = hnsw.search(query, k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recall_above_threshold(self, hnsw_setup):
        """HNSW recall should be reasonably high on clustered data."""
        _, brute, hnsw = hnsw_setup
        rng = np.random.default_rng(123)

        recalls = []
        for _ in range(20):
            query = rng.standard_normal(16).astype(np.float32)
            exact = brute.search(query, k=10)
            approx = hnsw.search(query, k=10, ef_search=100)

            exact_ids = {r.id for r in exact}
            approx_ids = {r.id for r in approx}
            recall = len(exact_ids & approx_ids) / len(exact_ids)
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        # HNSW should achieve at least 50% recall on this small dataset
        assert mean_recall >= 0.5, f"Mean recall {mean_recall:.2f} < 0.5"

    def test_deleted_nodes_ignored(self, hnsw_setup):
        store, _, hnsw = hnsw_setup
        store.delete("vec_5")
        query = store.vectors[5]
        results = hnsw.search(query, k=10)
        result_ids = [r.id for r in results]
        assert "vec_5" not in result_ids

    def test_invalid_k_raises(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        with pytest.raises(InvalidSearchParameterError):
            hnsw.search(np.random.randn(16).astype(np.float32), k=0)

    def test_invalid_ef_search_raises(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        with pytest.raises(InvalidSearchParameterError):
            hnsw.search(np.random.randn(16).astype(np.float32), k=5, ef_search=0)

    def test_single_node_search(self):
        store = VectorStore(dimension=4)
        store.insert("only", [1, 0, 0, 0])
        hnsw = HNSWIndex(m=4, ef_construction=10, ef_search=10)
        hnsw.build(store.vectors, store.ids, store.active_mask)
        results = hnsw.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
        assert len(results) == 1
        assert results[0].id == "only"

    def test_small_graph_connectivity(self, small_hnsw):
        store, hnsw = small_hnsw
        # Query with first vector — should find itself as top-1
        query = store.vectors[0]
        results = hnsw.search(query, k=1)
        assert results[0].id == "v0"

    def test_small_graph_neighbor_exists(self, small_hnsw):
        _, hnsw = small_hnsw
        # Every node should have at least one neighbor at layer 0
        for idx in range(5):
            layer0_neighbors = hnsw.neighbors[idx].get(0, [])
            assert len(layer0_neighbors) > 0

    def test_search_returns_search_result_type(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        query = np.random.randn(16).astype(np.float32)
        results = hnsw.search(query, k=3)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_stats(self, hnsw_setup):
        _, _, hnsw = hnsw_setup
        s = hnsw.stats()
        assert s["name"] == "hnsw"
        assert s["state"] == "ready"
        assert s["total_nodes"] == 200
        assert s["max_level"] >= 0
        assert s["total_edges"] > 0
