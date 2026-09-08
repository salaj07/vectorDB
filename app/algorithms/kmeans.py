"""
VectorForge — Manual K-Means Clustering

Used by IVF-Flat to partition the vector space into clusters.
Implemented from scratch — no sklearn.

Algorithm:
    1. Initialize K centroids (random selection from data)
    2. Assign each vector to nearest centroid
    3. Recompute centroid means
    4. Repeat until convergence or max_iterations
"""

from __future__ import annotations

import numpy as np

from app.core.distance import normalize, cosine_similarity_matrix


def kmeans(
    vectors: np.ndarray,
    k: int,
    max_iterations: int = 20,
    seed: int = 42,
    use_cosine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run K-Means clustering on a set of vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Shape ``(N, D)`` — **already normalized** vectors.
    k : int
        Number of clusters.
    max_iterations : int
        Maximum iterations before stopping.
    seed : int
        Random seed for reproducibility.
    use_cosine : bool
        If True, use cosine distance (1 - similarity) for assignment.
        If False, use Euclidean distance.

    Returns
    -------
    centroids : np.ndarray
        Shape ``(K, D)`` — final cluster centroids (normalized if use_cosine).
    assignments : np.ndarray
        Shape ``(N,)`` int — cluster ID for each vector.
    """
    rng = np.random.default_rng(seed)
    n, d = vectors.shape

    if k >= n:
        # Edge case: more clusters than vectors
        centroids = vectors.copy()
        assignments = np.arange(n, dtype=np.int32)
        return centroids, assignments

    # Step 1: Initialize centroids by random selection (no replacement)
    init_indices = rng.choice(n, size=k, replace=False)
    centroids = vectors[init_indices].copy()

    assignments = np.zeros(n, dtype=np.int32)

    for iteration in range(max_iterations):
        # Step 2: Assign each vector to the nearest centroid
        if use_cosine:
            # similarity: (K, N) — centroids as "queries", vectors as "database"
            # We want (N,) assignments, so compute (N, K) similarities
            sim_matrix = cosine_similarity_matrix(vectors, centroids)  # (N, K)
            new_assignments = np.argmax(sim_matrix, axis=1).astype(np.int32)
        else:
            # Euclidean: ||v - c||^2 = ||v||^2 + ||c||^2 - 2 v·c
            # For normalized vectors, ||v||^2 = ||c||^2 = 1
            # So distance^2 = 2 - 2 v·c  →  minimize distance = maximize v·c
            dots = vectors @ centroids.T  # (N, K)
            new_assignments = np.argmax(dots, axis=1).astype(np.int32)

        # Step 3: Check convergence
        if np.array_equal(new_assignments, assignments) and iteration > 0:
            break

        assignments = new_assignments

        # Step 4: Recompute centroids (vectorized scatter-add)
        sum_vecs = np.zeros((k, d), dtype=np.float32)
        counts = np.bincount(assignments, minlength=k)
        np.add.at(sum_vecs, assignments, vectors)
        mask = counts > 0
        centroids[mask] = sum_vecs[mask] / counts[mask, np.newaxis]

        # Re-normalize centroids if using cosine distance
        if use_cosine:
            centroids = normalize(centroids)

    return centroids, assignments
