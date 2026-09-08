"""
VectorForge — IVF-Flat (Inverted File Index)

Approximate nearest-neighbor search using cluster-based pruning.

Algorithm:
    Build:
        1. Run K-Means on all vectors → K centroids
        2. Assign each vector to its nearest centroid
        3. Build inverted lists: cluster_id → [vector indices]

    Search:
        1. Normalize query
        2. Compute similarity to all centroids
        3. Select top-nprobe clusters
        4. Gather candidate vectors from those clusters
        5. Exact cosine similarity on candidates
        6. Return top-k

Trade-off:
    Higher nprobe → higher recall, higher latency
    Lower  nprobe → lower  latency, lower recall
    nprobe = nlist → equivalent to brute force
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
from app.algorithms.kmeans import kmeans
from app.config import IVF_NLIST, IVF_NPROBE, IVF_KMEANS_MAX_ITER


class IVFIndex(BaseIndex):
    """
    IVF-Flat approximate nearest-neighbor index.

    Approximate candidate selection + exact vector comparison = IVF-Flat.
    """

    def __init__(
        self,
        nlist: int = IVF_NLIST,
        nprobe: int = IVF_NPROBE,
        max_iter: int = IVF_KMEANS_MAX_ITER,
    ):
        super().__init__(name="ivf")
        self.nlist: int = nlist
        self.default_nprobe: int = nprobe
        self.max_iter: int = max_iter

        # Built state
        self.centroids: np.ndarray | None = None
        self.inverted_lists: dict[int, list[int]] = {}

        # References to store data
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._active_mask: np.ndarray | None = None

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        active_mask: np.ndarray,
    ) -> None:
        """
        Build the IVF index by clustering vectors with K-Means.

        Parameters
        ----------
        vectors     : (N, D) normalized float32
        ids         : (N,) string IDs
        active_mask : (N,) bool
        """
        self._state = IndexState.BUILDING
        self._vectors = vectors
        self._ids = ids
        self._active_mask = active_mask

        n = len(vectors)
        actual_nlist = min(self.nlist, n)

        # Run K-Means
        self.centroids, assignments = kmeans(
            vectors, k=actual_nlist, max_iterations=self.max_iter
        )

        # Build inverted lists
        self.inverted_lists = {}
        for idx in range(n):
            cluster_id = int(assignments[idx])
            if cluster_id not in self.inverted_lists:
                self.inverted_lists[cluster_id] = []
            self.inverted_lists[cluster_id].append(idx)

        self._state = IndexState.READY

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        """
        Approximate top-k search using IVF.

        Parameters
        ----------
        query  : (D,) vector
        k      : number of results
        nprobe : number of clusters to search (overrides default)
        """
        if self._state not in (IndexState.READY, IndexState.DIRTY):
            raise IndexNotBuiltError(self.name)

        if nprobe is None:
            nprobe = self.default_nprobe

        if nprobe < 1:
            raise InvalidSearchParameterError("nprobe", nprobe, "must be >= 1")
        if k < 1:
            raise InvalidSearchParameterError("k", k, "must be >= 1")

        query = np.asarray(query, dtype=np.float32)
        if self._vectors is not None and self._vectors.ndim == 2 and self._vectors.shape[1] > 0:
            if query.shape[0] != self._vectors.shape[1]:
                raise DimensionMismatchError(self._vectors.shape[1], query.shape[0])
        query = normalize(query)

        # Clamp nprobe to actual cluster count
        actual_nlist = len(self.inverted_lists)
        nprobe = min(nprobe, actual_nlist)

        # Step 1: Find nearest clusters
        centroid_scores = cosine_similarity(query, self.centroids)
        if nprobe >= len(centroid_scores):
            top_clusters = np.arange(len(centroid_scores))
        else:
            top_clusters = np.argpartition(centroid_scores, -nprobe)[-nprobe:]

        # Step 2: Gather candidate vector indices from selected clusters
        candidate_indices = []
        for cluster_id in top_clusters:
            cluster_id = int(cluster_id)
            if cluster_id in self.inverted_lists:
                candidate_indices.extend(self.inverted_lists[cluster_id])

        if not candidate_indices:
            return []

        candidate_indices = np.array(candidate_indices, dtype=np.int64)

        # Step 3: Exact cosine similarity on candidates
        candidate_vectors = self._vectors[candidate_indices]
        candidate_ids = self._ids[candidate_indices]
        candidate_mask = self._active_mask[candidate_indices]

        scores = cosine_similarity(query, candidate_vectors)

        return top_k(scores, candidate_ids, k=k, mask=candidate_mask)

    def delete(self, vector_id: str) -> None:
        """
        Logical deletion — handled by the shared active_mask.
        The inverted lists still contain the index, but search
        filters it out via the mask.
        """
        pass

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": self.name,
            "state": self._state.value,
            "nlist": self.nlist,
            "default_nprobe": self.default_nprobe,
        }
        if self.centroids is not None:
            info["actual_clusters"] = len(self.inverted_lists)
            total = sum(len(v) for v in self.inverted_lists.values())
            info["indexed_vectors"] = total
            info["avg_cluster_size"] = (
                total / len(self.inverted_lists) if self.inverted_lists else 0
            )
        return info
