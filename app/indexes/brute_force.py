"""
VectorForge — Brute Force Index

Exact search: compares the query against every active vector.
Serves as the ground-truth reference implementation and benchmark baseline.

Complexity: O(N × D) per query.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.distance import normalize, cosine_similarity
from app.core.topk import top_k
from app.core.types import IndexState, SearchResult
from app.core.exceptions import (
    DimensionMismatchError,
    IndexNotBuiltError,
    InvalidSearchParameterError,
)
from app.indexes.base import BaseIndex


class BruteForceIndex(BaseIndex):
    """
    Exact nearest-neighbor search via exhaustive comparison.

    This index does not build any data structure — it simply stores
    references to the vectors, IDs, and active mask from the VectorStore.
    """

    def __init__(self):
        super().__init__(name="brute")
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._active_mask: np.ndarray | None = None

    def build(self, vectors: np.ndarray, ids: np.ndarray, active_mask: np.ndarray) -> None:
        """
        'Build' the brute force index — simply stores references.

        Parameters
        ----------
        vectors     : (N, D) normalized float32
        ids         : (N,) string IDs
        active_mask : (N,) bool
        """
        self._vectors = vectors
        self._ids = ids
        self._active_mask = active_mask
        self._state = IndexState.READY

    def search(self, query: np.ndarray, k: int = 10, **kwargs) -> list[SearchResult]:
        """
        Exact top-k search.

        Parameters
        ----------
        query : (D,) vector — will be normalized
        k     : number of results

        Returns
        -------
        list[SearchResult] sorted by cosine similarity descending.
        """
        if self._state not in (IndexState.READY, IndexState.DIRTY):
            raise IndexNotBuiltError(self.name)

        if k < 1:
            raise InvalidSearchParameterError("k", k, "k must be >= 1")

        query = np.asarray(query, dtype=np.float32)
        if self._vectors is not None and self._vectors.ndim == 2 and self._vectors.shape[1] > 0:
            if query.shape[0] != self._vectors.shape[1]:
                raise DimensionMismatchError(self._vectors.shape[1], query.shape[0])
        query = normalize(query)

        # Compute cosine similarity against ALL vectors (including deleted)
        scores = cosine_similarity(query, self._vectors)

        # top_k handles the active_mask filtering
        return top_k(scores, self._ids, k=k, mask=self._active_mask)

    def delete(self, vector_id: str) -> None:
        """
        Brute force doesn't maintain its own structure —
        deletion is handled by the shared active_mask.
        """
        # The active_mask is shared with VectorStore,
        # so deletion in the store automatically applies here.
        pass

    def stats(self) -> dict[str, Any]:
        if self._vectors is None:
            return {"name": self.name, "state": self._state.value, "vectors": 0}
        active = int(np.sum(self._active_mask)) if self._active_mask is not None else 0
        return {
            "name": self.name,
            "state": self._state.value,
            "total_vectors": len(self._vectors),
            "active_vectors": active,
            "dimension": self._vectors.shape[1] if len(self._vectors) > 0 else 0,
        }
