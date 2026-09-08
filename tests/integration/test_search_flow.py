"""
Tests for the full search flow: insert → build → search → delete → search.
"""

import numpy as np
import pytest

from app.core.vector_store import VectorStore
from app.core.distance import normalize
from app.indexes.brute_force import BruteForceIndex
from app.indexes.ivf import IVFIndex
from app.indexes.hnsw import HNSWIndex


class TestSearchFlow:
    """End-to-end insert → build → search → delete → search tests."""

    def test_insert_build_search_delete_search(self):
        """
        Insert A, B, C → build → search → delete B → search → B never appears.
        """
        dim = 8
        store = VectorStore(dimension=dim)

        rng = np.random.default_rng(42)
        vec_a = rng.standard_normal(dim).astype(np.float32)
        vec_b = rng.standard_normal(dim).astype(np.float32)
        vec_c = rng.standard_normal(dim).astype(np.float32)

        store.insert("A", vec_a, {"label": "a"})
        store.insert("B", vec_b, {"label": "b"})
        store.insert("C", vec_c, {"label": "c"})

        brute = BruteForceIndex()
        brute.build(store.vectors, store.ids, store.active_mask)

        # Search — all three should be findable
        results = brute.search(vec_b, k=3)
        result_ids = {r.id for r in results}
        assert "A" in result_ids
        assert "B" in result_ids
        assert "C" in result_ids

        # Delete B
        store.delete("B")

        # Search again — B should not appear
        results = brute.search(vec_b, k=3)
        result_ids = {r.id for r in results}
        assert "B" not in result_ids
        assert len(results) == 2

    def test_flow_with_ivf(self):
        """Full flow with IVF index."""
        dim = 8
        store = VectorStore(dimension=dim)

        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((20, dim)).astype(np.float32)
        ids = [f"vec_{i}" for i in range(20)]
        store.bulk_insert(ids, vecs)

        ivf = IVFIndex(nlist=4, nprobe=4)
        ivf.build(store.vectors, store.ids, store.active_mask)

        # Search works
        results = ivf.search(vecs[5], k=5)
        assert len(results) == 5
        assert results[0].id == "vec_5"

        # Delete and verify
        store.delete("vec_5")
        results = ivf.search(vecs[5], k=5)
        assert "vec_5" not in [r.id for r in results]

    def test_flow_with_hnsw(self):
        """Full flow with HNSW index."""
        dim = 8
        store = VectorStore(dimension=dim)

        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((20, dim)).astype(np.float32)
        ids = [f"vec_{i}" for i in range(20)]
        store.bulk_insert(ids, vecs)

        hnsw = HNSWIndex(m=8, ef_construction=50, ef_search=20)
        hnsw.build(store.vectors, store.ids, store.active_mask)

        # Search works
        results = hnsw.search(vecs[5], k=5)
        assert len(results) >= 1

        # Delete and verify
        store.delete("vec_5")
        results = hnsw.search(vecs[5], k=5)
        assert "vec_5" not in [r.id for r in results]

    def test_multiple_deletes_then_search(self):
        """Delete multiple vectors and verify none appear."""
        dim = 8
        store = VectorStore(dimension=dim)

        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((50, dim)).astype(np.float32)
        ids = [f"v{i}" for i in range(50)]
        store.bulk_insert(ids, vecs)

        brute = BruteForceIndex()
        brute.build(store.vectors, store.ids, store.active_mask)

        # Delete first 10 vectors
        deleted = set()
        for i in range(10):
            store.delete(f"v{i}")
            deleted.add(f"v{i}")

        # No deleted vector should appear in any search
        for i in range(10, 50):
            results = brute.search(vecs[i], k=10)
            result_ids = {r.id for r in results}
            assert result_ids.isdisjoint(deleted)
