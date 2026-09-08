"""
Tests for app.core.topk — optimized top-k selection.
"""

import numpy as np
import pytest

from app.core.topk import top_k
from app.core.types import SearchResult
from app.core.exceptions import InvalidSearchParameterError


class TestTopK:
    """Tests for the top_k() function."""

    def _make_ids(self, n: int) -> np.ndarray:
        return np.array([f"vec_{i}" for i in range(n)])

    def test_correct_top_k_ids(self):
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3], dtype=np.float32)
        ids = self._make_ids(5)
        results = top_k(scores, ids, k=3)
        result_ids = [r.id for r in results]
        assert result_ids == ["vec_1", "vec_3", "vec_2"]

    def test_correct_descending_order(self):
        scores = np.array([0.3, 0.7, 0.5, 0.9, 0.1], dtype=np.float32)
        ids = self._make_ids(5)
        results = top_k(scores, ids, k=3)
        result_scores = [r.score for r in results]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_k_equals_one(self):
        scores = np.array([0.2, 0.8, 0.5], dtype=np.float32)
        ids = self._make_ids(3)
        results = top_k(scores, ids, k=1)
        assert len(results) == 1
        assert results[0].id == "vec_1"
        assert np.isclose(results[0].score, 0.8, atol=1e-6)

    def test_k_equals_n(self):
        scores = np.array([0.3, 0.1, 0.9, 0.5], dtype=np.float32)
        ids = self._make_ids(4)
        results = top_k(scores, ids, k=4)
        assert len(results) == 4
        result_scores = [r.score for r in results]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_k_greater_than_n_clamps(self):
        scores = np.array([0.5, 0.3], dtype=np.float32)
        ids = self._make_ids(2)
        results = top_k(scores, ids, k=10)
        assert len(results) == 2

    def test_duplicate_scores(self):
        scores = np.array([0.5, 0.5, 0.5, 0.3], dtype=np.float32)
        ids = self._make_ids(4)
        results = top_k(scores, ids, k=3)
        assert len(results) == 3
        # All top-3 should have score 0.5
        for r in results:
            assert np.isclose(r.score, 0.5, atol=1e-6)

    def test_invalid_k_raises(self):
        scores = np.array([0.5], dtype=np.float32)
        ids = self._make_ids(1)
        with pytest.raises(InvalidSearchParameterError):
            top_k(scores, ids, k=0)
        with pytest.raises(InvalidSearchParameterError):
            top_k(scores, ids, k=-1)

    def test_empty_scores_returns_empty(self):
        scores = np.array([], dtype=np.float32)
        ids = np.array([])
        results = top_k(scores, ids, k=5)
        assert results == []

    def test_mask_filters_inactive(self):
        scores = np.array([0.9, 0.1, 0.8, 0.7], dtype=np.float32)
        ids = self._make_ids(4)
        mask = np.array([True, True, False, True])  # vec_2 is inactive
        results = top_k(scores, ids, k=3, mask=mask)
        result_ids = [r.id for r in results]
        assert "vec_2" not in result_ids
        assert len(results) == 3

    def test_mask_all_inactive_returns_empty(self):
        scores = np.array([0.9, 0.8], dtype=np.float32)
        ids = self._make_ids(2)
        mask = np.array([False, False])
        results = top_k(scores, ids, k=2, mask=mask)
        assert results == []

    def test_returns_search_result_instances(self):
        scores = np.array([0.5], dtype=np.float32)
        ids = self._make_ids(1)
        results = top_k(scores, ids, k=1)
        assert isinstance(results[0], SearchResult)
        assert isinstance(results[0].id, str)
        assert isinstance(results[0].score, float)
