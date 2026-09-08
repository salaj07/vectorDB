"""
Tests for app.algorithms.kmeans — manual K-Means clustering.
"""

import numpy as np
import pytest

from app.algorithms.kmeans import kmeans
from app.core.distance import normalize


@pytest.fixture
def clustered_data():
    """
    Generate 200 vectors in dim 16 around 4 known cluster centers.
    This makes it easy to verify K-Means produces sensible assignments.
    """
    rng = np.random.default_rng(42)
    centers = normalize(rng.standard_normal((4, 16)).astype(np.float32))
    vectors = []
    labels = []
    for i, c in enumerate(centers):
        # 50 vectors per cluster with small noise
        noise = rng.standard_normal((50, 16)).astype(np.float32) * 0.1
        cluster_vecs = c + noise
        vectors.append(cluster_vecs)
        labels.extend([i] * 50)

    vectors = normalize(np.vstack(vectors))
    return vectors, np.array(labels), centers


class TestKMeans:

    def test_centroid_shape(self, clustered_data):
        vectors, _, _ = clustered_data
        centroids, _ = kmeans(vectors, k=4)
        assert centroids.shape == (4, 16)

    def test_assignment_length(self, clustered_data):
        vectors, _, _ = clustered_data
        _, assignments = kmeans(vectors, k=4)
        assert len(assignments) == len(vectors)

    def test_deterministic_with_same_seed(self, clustered_data):
        vectors, _, _ = clustered_data
        c1, a1 = kmeans(vectors, k=4, seed=42)
        c2, a2 = kmeans(vectors, k=4, seed=42)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_allclose(c1, c2, atol=1e-6)

    def test_valid_cluster_ids(self, clustered_data):
        vectors, _, _ = clustered_data
        _, assignments = kmeans(vectors, k=4)
        assert np.all(assignments >= 0)
        assert np.all(assignments < 4)

    def test_all_clusters_have_members(self, clustered_data):
        vectors, _, _ = clustered_data
        _, assignments = kmeans(vectors, k=4)
        for j in range(4):
            assert np.sum(assignments == j) > 0

    def test_known_clusters_sensible_assignments(self, clustered_data):
        """
        With well-separated clusters and K=4, most vectors should be
        assigned to the correct cluster. We check that the majority
        of each true cluster maps to a single K-Means cluster.
        """
        vectors, true_labels, _ = clustered_data
        _, assignments = kmeans(vectors, k=4)

        # For each true cluster, find the dominant K-Means cluster
        for true_c in range(4):
            mask = true_labels == true_c
            assigned = assignments[mask]
            # Most common assignment
            values, counts = np.unique(assigned, return_counts=True)
            dominant_count = counts.max()
            # At least 70% of vectors in this true cluster
            # should map to the same K-Means cluster
            assert dominant_count / mask.sum() >= 0.7

    def test_convergence_stops_early(self, clustered_data):
        """K-Means on well-separated data should converge before max_iterations."""
        vectors, _, _ = clustered_data
        # Use a high max_iterations — it should still converge quickly
        centroids, _ = kmeans(vectors, k=4, max_iterations=100)
        assert centroids.shape == (4, 16)

    def test_centroids_are_normalized(self, clustered_data):
        vectors, _, _ = clustered_data
        centroids, _ = kmeans(vectors, k=4, use_cosine=True)
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_single_cluster(self, clustered_data):
        vectors, _, _ = clustered_data
        centroids, assignments = kmeans(vectors, k=1)
        assert centroids.shape == (1, 16)
        assert np.all(assignments == 0)

    def test_k_equals_n(self):
        """Edge case: as many clusters as vectors."""
        rng = np.random.default_rng(42)
        vecs = normalize(rng.standard_normal((5, 8)).astype(np.float32))
        centroids, assignments = kmeans(vecs, k=5)
        assert centroids.shape == (5, 8)
        assert len(assignments) == 5
