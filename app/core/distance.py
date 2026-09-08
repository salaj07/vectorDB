"""
VectorForge — Distance & Normalization Utilities

This module is the SINGLE source of truth for:
  - L2 normalization
  - Cosine similarity computation

Every index (Brute Force, IVF, HNSW) reuses these functions.
Do NOT duplicate this logic elsewhere.
"""

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors so each has unit length.

    Parameters
    ----------
    vectors : np.ndarray
        Shape ``(D,)`` for a single vector or ``(N, D)`` for a batch.

    Returns
    -------
    np.ndarray
        Normalized vectors with the same shape.  Zero-length vectors
        are returned as zeros (not NaN).
    """
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0.0:
            return vectors
        return vectors / norm

    # Batch: (N, D)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero — keep zero vectors as zeros
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and a matrix of vectors.

    Both ``query`` and ``vectors`` are assumed to be **already normalized**.
    When inputs are unit-length, cosine similarity reduces to a dot product.

    Parameters
    ----------
    query : np.ndarray
        Shape ``(D,)`` — a single normalized query vector.
    vectors : np.ndarray
        Shape ``(N, D)`` — normalized stored vectors.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` — similarity scores in ``[-1, 1]``.
    """
    query = np.asarray(query, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)

    # Matrix-vector multiply: (N, D) @ (D,) → (N,)
    return vectors @ query


def cosine_similarity_matrix(
    queries: np.ndarray, vectors: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between Q queries and N vectors.

    Parameters
    ----------
    queries : np.ndarray
        Shape ``(Q, D)`` — normalized query vectors.
    vectors : np.ndarray
        Shape ``(N, D)`` — normalized stored vectors.

    Returns
    -------
    np.ndarray
        Shape ``(Q, N)`` — similarity matrix.
    """
    queries = np.asarray(queries, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)

    # (Q, D) @ (D, N) → (Q, N)
    return queries @ vectors.T
