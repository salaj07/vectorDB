"""
Tests for cross-index correctness.

Uses the same small dataset for Brute Force, IVF, and HNSW.
Validates that all indexes return consistent, valid results.
"""

import numpy as np
import pytest

from app.core.distance import normalize


class TestCrossIndexCorrectness:
    """Compare results across all three indexes."""

    def test_all_result_ids_exist(self, vector_store, brute_index, ivf_index, hnsw_index, small_queries):
        valid_ids = set(vector_store.ids)
        for query in small_queries:
            for index in [brute_index, ivf_index, hnsw_index]:
                results = index.search(query, k=5)
                for r in results:
                    assert r.id in valid_ids, f"{index.name}: invalid ID {r.id}"

    def test_result_count_matches_k(self, brute_index, ivf_index, hnsw_index, small_queries):
        k = 5
        for query in small_queries:
            for index in [brute_index, ivf_index, hnsw_index]:
                results = index.search(query, k=k)
                assert len(results) == k, f"{index.name}: got {len(results)}, expected {k}"

    def test_scores_ordered_descending(self, brute_index, ivf_index, hnsw_index, small_queries):
        for query in small_queries:
            for index in [brute_index, ivf_index, hnsw_index]:
                results = index.search(query, k=10)
                scores = [r.score for r in results]
                assert scores == sorted(scores, reverse=True), \
                    f"{index.name}: scores not descending"

    def test_deleted_ids_never_appear(self, vector_store, brute_index, ivf_index, hnsw_index, small_queries):
        # Delete a few vectors
        vector_store.delete("vec_5")
        vector_store.delete("vec_10")
        vector_store.delete("vec_15")

        deleted_ids = {"vec_5", "vec_10", "vec_15"}

        for query in small_queries:
            for index in [brute_index, ivf_index, hnsw_index]:
                results = index.search(query, k=10)
                result_ids = {r.id for r in results}
                assert result_ids.isdisjoint(deleted_ids), \
                    f"{index.name}: deleted IDs found in results"

    def test_ivf_recall_vs_brute(self, brute_index, ivf_index, small_queries):
        """IVF with default nprobe should have measurable recall."""
        recalls = []
        for query in small_queries:
            exact = brute_index.search(query, k=10)
            approx = ivf_index.search(query, k=10, nprobe=10)  # search all clusters

            exact_ids = {r.id for r in exact}
            approx_ids = {r.id for r in approx}
            recall = len(exact_ids & approx_ids) / len(exact_ids)
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        # With nprobe=nlist, should be perfect or near-perfect
        assert mean_recall >= 0.95, f"IVF recall with full probe: {mean_recall:.2f}"

    def test_hnsw_recall_vs_brute(self, brute_index, hnsw_index, small_queries):
        """HNSW should achieve reasonable recall."""
        recalls = []
        for query in small_queries:
            exact = brute_index.search(query, k=10)
            approx = hnsw_index.search(query, k=10, ef_search=50)

            exact_ids = {r.id for r in exact}
            approx_ids = {r.id for r in approx}
            recall = len(exact_ids & approx_ids) / len(exact_ids)
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        assert mean_recall >= 0.3, f"HNSW recall: {mean_recall:.2f}"
