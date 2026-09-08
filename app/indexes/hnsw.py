"""
VectorForge — HNSW (Hierarchical Navigable Small World) Index

A graph-based approximate nearest-neighbor index with multiple layers.

Higher layers: fewer nodes, long-range connections, fast navigation.
Layer 0:       all nodes,  detailed neighborhood.

Parameters:
    M               : max neighbors per node per layer
    ef_construction : build-time search width (quality of graph)
    ef_search       : query-time search width (quality of results)

Build:
    For each vector:
        1. Choose random level
        2. Start from entry point at top layer
        3. Greedy search down to node's level
        4. At each level, find ef_construction nearest neighbors
        5. Connect bidirectionally, prune to M
        6. Update entry point if new node's level is higher

Search:
    1. Start from entry point at top layer
    2. Greedy search at upper layers (1-nearest)
    3. At layer 0, expand to ef_search candidates
    4. Return top-k from candidates
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from app.core.distance import normalize
from app.core.types import IndexState, SearchResult
from app.core.exceptions import (
    DimensionMismatchError,
    IndexNotBuiltError,
    InvalidSearchParameterError,
)
from app.indexes.base import BaseIndex
from app.config import HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH


class HNSWIndex(BaseIndex):
    """
    HNSW approximate nearest-neighbor index.

    Represents vectors as nodes in a multi-layer graph.
    Similar vectors are connected as neighbors.
    """

    def __init__(
        self,
        m: int = HNSW_M,
        ef_construction: int = HNSW_EF_CONSTRUCTION,
        ef_search: int = HNSW_EF_SEARCH,
    ):
        super().__init__(name="hnsw")
        self.M: int = m
        self.M_max0: int = m * 2        # max neighbors at layer 0
        self.ef_construction: int = ef_construction
        self.ef_search: int = ef_search
        self.mL: float = 1.0 / math.log(m) if m > 1 else 1.0

        # Graph structure
        # neighbors[node_idx][layer] = list of neighbor indices
        self.neighbors: dict[int, dict[int, list[int]]] = {}
        self.max_level: int = -1
        self.entry_point: int = -1

        # References to store data
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._active_mask: np.ndarray | None = None
        self._node_levels: dict[int, int] = {}

        # RNG for level assignment
        self._rng = np.random.default_rng(42)

    def _random_level(self) -> int:
        """Assign a random level to a new node."""
        r = self._rng.uniform(0.0, 1.0)
        level = int(-math.log(max(r, 1e-9)) * self.mL)
        return level

    def _similarity(self, idx_a: int, idx_b: int) -> float:
        """Cosine similarity between two stored vectors."""
        return float(np.dot(self._vectors[idx_a], self._vectors[idx_b]))

    def _similarity_to_query(self, query: np.ndarray, idx: int) -> float:
        """Cosine similarity between a query vector and a stored vector."""
        return float(np.dot(query, self._vectors[idx]))

    def _search_layer(
        self, query: np.ndarray, entry: int, ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """
        Search a single layer starting from entry point using HNSW Algorithm 2.

        Uses two heaps:
          - candidates_heap: min-heap of (-sim, idx) so heappop yields the
            unvisited candidate with the highest similarity.
          - results_heap: min-heap of (sim, idx) tracking the best ef results found,
            where results_heap[0] is the worst (lowest similarity) result.

        Returns a list of (similarity, node_idx) tuples for the top ef candidates.
        """
        visited = {entry}
        sim_entry = self._similarity_to_query(query, entry)

        candidates_heap = [(-sim_entry, entry)]
        results_heap = [(sim_entry, entry)]

        while candidates_heap:
            neg_sim_c, c_idx = heapq.heappop(candidates_heap)
            current_best_sim = -neg_sim_c

            worst_result_sim = results_heap[0][0]
            if len(results_heap) >= ef and current_best_sim < worst_result_sim:
                break

            node_neighbors = self.neighbors.get(c_idx, {}).get(layer, [])
            for n_idx in node_neighbors:
                if n_idx in visited:
                    continue
                visited.add(n_idx)

                n_sim = self._similarity_to_query(query, n_idx)

                # Push neighbor to candidate heap
                heapq.heappush(candidates_heap, (-n_sim, n_idx))

                # Push to results heap if better than worst result or capacity not reached
                if len(results_heap) < ef or n_sim > results_heap[0][0]:
                    heapq.heappush(results_heap, (n_sim, n_idx))
                    if len(results_heap) > ef:
                        heapq.heappop(results_heap)

        # Convert to (similarity, idx) list
        return [(sim, idx) for sim, idx in results_heap]

    def _select_neighbors(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Select the top-m neighbors from candidates by similarity."""
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in candidates[:m]]

    def _add_node(self, idx: int, level: int, query_vec: np.ndarray) -> None:
        """Insert a single node into the graph."""
        self._node_levels[idx] = level
        self.neighbors[idx] = {l: [] for l in range(level + 1)}

        if self.entry_point == -1:
            self.entry_point = idx
            self.max_level = level
            return

        ep = self.entry_point

        # Traverse from top to node's level + 1 (greedy 1-nearest)
        for lc in range(self.max_level, level, -1):
            if lc not in self.neighbors.get(ep, {}):
                break
            candidates = self._search_layer(query_vec, ep, ef=1, layer=lc)
            if candidates:
                ep = max(candidates, key=lambda x: x[0])[1]

        # From node's level down to 0, do ef_construction search
        for lc in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(
                query_vec, ep, ef=self.ef_construction, layer=lc
            )

            m = self.M if lc > 0 else self.M_max0
            neighbors = self._select_neighbors(candidates, m)

            # Connect new node to its neighbors
            self.neighbors[idx][lc] = neighbors

            # Connect neighbors back to new node (bidirectional)
            for n_idx in neighbors:
                if lc not in self.neighbors[n_idx]:
                    self.neighbors[n_idx][lc] = []

                self.neighbors[n_idx][lc].append(idx)

                # Prune if over capacity
                if len(self.neighbors[n_idx][lc]) > m:
                    # Keep the best m neighbors
                    scored = [
                        (self._similarity(n_idx, nb), nb)
                        for nb in self.neighbors[n_idx][lc]
                    ]
                    self.neighbors[n_idx][lc] = self._select_neighbors(scored, m)

            if candidates:
                ep = max(candidates, key=lambda x: x[0])[1]

        # Update entry point if new node has higher level
        if level > self.max_level:
            self.entry_point = idx
            self.max_level = level

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        active_mask: np.ndarray,
    ) -> None:
        """
        Build the HNSW graph by inserting all vectors.

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

        # Reset graph
        self.neighbors = {}
        self._node_levels = {}
        self.max_level = -1
        self.entry_point = -1
        self._rng = np.random.default_rng(42)

        n = len(vectors)
        for i in range(n):
            level = self._random_level()
            self._add_node(i, level, vectors[i])

        self._state = IndexState.READY

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        """
        Approximate top-k search via HNSW graph traversal.

        Parameters
        ----------
        query     : (D,) vector
        k         : number of results
        ef_search : search width (overrides default)
        """
        if self._state not in (IndexState.READY, IndexState.DIRTY):
            raise IndexNotBuiltError(self.name)
        if k < 1:
            raise InvalidSearchParameterError("k", k, "must be >= 1")

        if ef_search is None:
            ef_search = self.ef_search
        if ef_search < 1:
            raise InvalidSearchParameterError(
                "ef_search", ef_search, "must be >= 1"
            )

        query = np.asarray(query, dtype=np.float32)
        if self._vectors is not None and self._vectors.ndim == 2 and self._vectors.shape[1] > 0:
            if query.shape[0] != self._vectors.shape[1]:
                raise DimensionMismatchError(self._vectors.shape[1], query.shape[0])
        query = normalize(query)

        # Ensure ef >= k
        ef = max(ef_search, k)

        ep = self.entry_point
        if ep == -1:
            return []

        # Traverse upper layers with greedy 1-nearest
        for lc in range(self.max_level, 0, -1):
            candidates = self._search_layer(query, ep, ef=1, layer=lc)
            if candidates:
                ep = max(candidates, key=lambda x: x[0])[1]

        # Layer 0: full ef search
        candidates = self._search_layer(query, ep, ef=ef, layer=0)

        # Filter by active mask and sort
        candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, idx in candidates:
            if self._active_mask[idx]:
                results.append(
                    SearchResult(id=str(self._ids[idx]), score=sim)
                )
                if len(results) >= k:
                    break

        return results

    def delete(self, vector_id: str) -> None:
        """Logical deletion — handled by shared active_mask."""
        pass

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": self.name,
            "state": self._state.value,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
        if self._vectors is not None:
            info["total_nodes"] = len(self.neighbors)
            info["max_level"] = self.max_level
            info["entry_point"] = self.entry_point
            total_edges = sum(
                len(nbrs)
                for node in self.neighbors.values()
                for nbrs in node.values()
            )
            info["total_edges"] = total_edges
        return info
