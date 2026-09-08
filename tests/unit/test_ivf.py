"""
Tests for app.indexes.ivf — IVF-Flat approximate index.
"""

import numpy as np
import pytest

from app.core.vector_store import VectorStore
from app.core.distance import normalize
from app.core.types import SearchResult, IndexState
from app.core.exceptions import IndexNotBuiltError, InvalidSearchParameterError
from app.indexes.brute_force import BruteForceIndex
from app.indexes.ivf import IVFIndex


@pytest.fixture
def ivf_setup():
    """
    Create a store with 200 clustered vectors of dim 16,
    and build both brute-force and IVF indexes.
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

    ivf = IVFIndex(nlist=4, nprobe=2)
    ivf.build(store.vectors, store.ids, store.active_mask)

    return store, brute, ivf


class TestIVFIndex:

    def test_state_after_build(self, ivf_setup):
        _, _, ivf = ivf_setup
        assert ivf.state == IndexState.READY

    def test_state_before_build(self):
        ivf = IVFIndex()
        assert ivf.state == IndexState.EMPTY

    def test_search_not_built_raises(self):
        ivf = IVFIndex()
        with pytest.raises(IndexNotBuiltError):
            ivf.search(np.random.randn(16).astype(np.float32), k=5)

    def test_centroid_count(self, ivf_setup):
        _, _, ivf = ivf_setup
        assert ivf.centroids.shape[0] == 4

    def test_inverted_lists_cover_all_vectors(self, ivf_setup):
        store, _, ivf = ivf_setup
        total = sum(len(v) for v in ivf.inverted_lists.values())
        assert total == store.total_count()

    def test_each_vector_in_exactly_one_list(self, ivf_setup):
        store, _, ivf = ivf_setup
        all_indices = []
        for indices in ivf.inverted_lists.values():
            all_indices.extend(indices)
        # No duplicates
        assert len(all_indices) == len(set(all_indices))
        # All vectors accounted for
        assert len(all_indices) == store.total_count()

    def test_search_returns_valid_results(self, ivf_setup):
        store, _, ivf = ivf_setup
        query = np.random.default_rng(99).standard_normal(16).astype(np.float32)
        results = ivf.search(query, k=5)
        assert len(results) == 5
        valid_ids = set(store.ids)
        for r in results:
            assert r.id in valid_ids

    def test_results_ordered_descending(self, ivf_setup):
        _, _, ivf = ivf_setup
        query = np.random.default_rng(99).standard_normal(16).astype(np.float32)
        results = ivf.search(query, k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_nprobe_equals_nlist_matches_brute_force(self, ivf_setup):
        """When nprobe == nlist, IVF should search ALL clusters → exact search."""
        store, brute, ivf = ivf_setup
        query = np.random.default_rng(99).standard_normal(16).astype(np.float32)

        exact_results = brute.search(query, k=10)
        ivf_results = ivf.search(query, k=10, nprobe=4)  # nprobe == nlist

        exact_ids = {r.id for r in exact_results}
        ivf_ids = {r.id for r in ivf_results}
        assert exact_ids == ivf_ids

    def test_nprobe_one(self, ivf_setup):
        _, _, ivf = ivf_setup
        query = np.random.default_rng(99).standard_normal(16).astype(np.float32)
        results = ivf.search(query, k=5, nprobe=1)
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_deleted_vectors_ignored(self, ivf_setup):
        store, _, ivf = ivf_setup
        store.delete("vec_5")
        query = store.vectors[5]
        results = ivf.search(query, k=10, nprobe=4)
        result_ids = [r.id for r in results]
        assert "vec_5" not in result_ids

    def test_invalid_nprobe_raises(self, ivf_setup):
        _, _, ivf = ivf_setup
        query = np.random.randn(16).astype(np.float32)
        with pytest.raises(InvalidSearchParameterError):
            ivf.search(query, k=5, nprobe=0)

    def test_invalid_k_raises(self, ivf_setup):
        _, _, ivf = ivf_setup
        query = np.random.randn(16).astype(np.float32)
        with pytest.raises(InvalidSearchParameterError):
            ivf.search(query, k=0)

    def test_stats(self, ivf_setup):
        _, _, ivf = ivf_setup
        s = ivf.stats()
        assert s["name"] == "ivf"
        assert s["state"] == "ready"
        assert s["actual_clusters"] == 4
        assert s["indexed_vectors"] == 200
