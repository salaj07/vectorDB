"""
VectorForge — Optimized Top-K Selection

Uses np.argpartition for O(N) partial sort instead of O(N log N) full sort.
Only the final k candidates are sorted.

This utility is shared by Brute Force, IVF, and HNSW indexes.
"""

from __future__ import annotations

import numpy as np

from app.core.types import SearchResult
from app.core.exceptions import InvalidSearchParameterError


def top_k(
    scores: np.ndarray,
    ids: np.ndarray,
    k: int,
    mask: np.ndarray | None = None,
) -> list[SearchResult]:
    """
    Select the top-k highest-scoring results.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N,)`` — similarity scores.
    ids : np.ndarray
        Shape ``(N,)`` — corresponding vector IDs (strings).
    k : int
        Number of results to return.  Must be >= 1.
    mask : np.ndarray | None
        Shape ``(N,)`` boolean mask.  ``True`` = active (included).
        If None, all entries are included.

    Returns
    -------
    list[SearchResult]
        Top-k results sorted by score descending.

    Raises
    ------
    InvalidSearchParameterError
        If k < 1 or inputs are empty after masking.
    """
    if k < 1:
        raise InvalidSearchParameterError("k", k, "k must be >= 1")

    scores = np.asarray(scores, dtype=np.float32)
    ids = np.asarray(ids)

    # Apply active mask
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        active_idx = np.where(mask)[0]
        if active_idx.size == 0:
            return []
        scores = scores[active_idx]
        ids = ids[active_idx]

    n = len(scores)
    if n == 0:
        return []

    # Clamp k to available count
    k = min(k, n)

    if k >= n:
        # All elements requested — just sort
        sorted_idx = np.argsort(scores)[::-1]
    else:
        # argpartition: O(N) to find top-k candidates
        # We negate scores because argpartition finds smallest,
        # and we want largest.
        part_idx = np.argpartition(scores, -k)[-k:]
        # Sort only those k candidates
        sorted_idx = part_idx[np.argsort(scores[part_idx])[::-1]]

    return [
        SearchResult(id=str(ids[i]), score=float(scores[i]))
        for i in sorted_idx
    ]
