"""
Tests for app.core.distance — normalization and cosine similarity.
"""

import numpy as np
import pytest

from app.core.distance import normalize, cosine_similarity, cosine_similarity_matrix


class TestNormalize:
    """Tests for the normalize() function."""

    def test_single_vector_has_unit_length(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        normed = normalize(v)
        assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-6)

    def test_batch_vectors_have_unit_length(self):
        vecs = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]], dtype=np.float32)
        normed = normalize(vecs)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector_stays_zero(self):
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        normed = normalize(v)
        np.testing.assert_array_equal(normed, v)

    def test_zero_vector_in_batch(self):
        vecs = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        normed = normalize(vecs)
        # First vector should be normalized
        assert np.isclose(np.linalg.norm(normed[0]), 1.0, atol=1e-6)
        # Zero vector should remain zero
        np.testing.assert_array_equal(normed[1], [0.0, 0.0])

    def test_already_normalized_is_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        normed = normalize(v)
        np.testing.assert_allclose(normed, v, atol=1e-6)

    def test_output_dtype_is_float32(self):
        v = np.array([1, 2, 3])  # integer input
        normed = normalize(v)
        assert normed.dtype == np.float32

    def test_shape_preserved_1d(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert normalize(v).shape == (3,)

    def test_shape_preserved_2d(self):
        vecs = np.random.randn(10, 5).astype(np.float32)
        assert normalize(vecs).shape == (10, 5)


class TestCosineSimilarity:
    """Tests for cosine_similarity() — assumes pre-normalized inputs."""

    def test_identical_vectors_similarity_one(self):
        v = normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        vecs = v.reshape(1, -1)
        scores = cosine_similarity(v, vecs)
        assert np.isclose(scores[0], 1.0, atol=1e-5)

    def test_opposite_vectors_similarity_negative_one(self):
        v1 = normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = normalize(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        scores = cosine_similarity(v1, v2.reshape(1, -1))
        assert np.isclose(scores[0], -1.0, atol=1e-5)

    def test_orthogonal_vectors_similarity_zero(self):
        v1 = normalize(np.array([1.0, 0.0], dtype=np.float32))
        v2 = normalize(np.array([0.0, 1.0], dtype=np.float32))
        scores = cosine_similarity(v1, v2.reshape(1, -1))
        assert np.isclose(scores[0], 0.0, atol=1e-5)

    def test_batch_output_shape(self):
        query = normalize(np.random.randn(16).astype(np.float32))
        vecs = normalize(np.random.randn(100, 16).astype(np.float32))
        scores = cosine_similarity(query, vecs)
        assert scores.shape == (100,)

    def test_scores_in_valid_range(self):
        query = normalize(np.random.randn(32).astype(np.float32))
        vecs = normalize(np.random.randn(50, 32).astype(np.float32))
        scores = cosine_similarity(query, vecs)
        assert np.all(scores >= -1.0 - 1e-5)
        assert np.all(scores <= 1.0 + 1e-5)

    def test_float32_inputs(self):
        query = normalize(np.array([0.5, 0.5], dtype=np.float32))
        vecs = normalize(np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32))
        scores = cosine_similarity(query, vecs)
        assert scores.dtype == np.float32


class TestCosineSimilarityMatrix:
    """Tests for cosine_similarity_matrix() — pairwise computation."""

    def test_output_shape(self):
        queries = normalize(np.random.randn(10, 16).astype(np.float32))
        vecs = normalize(np.random.randn(50, 16).astype(np.float32))
        result = cosine_similarity_matrix(queries, vecs)
        assert result.shape == (10, 50)

    def test_self_similarity_diagonal(self):
        vecs = normalize(np.random.randn(5, 8).astype(np.float32))
        result = cosine_similarity_matrix(vecs, vecs)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-5)
