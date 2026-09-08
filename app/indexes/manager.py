"""
VectorForge — Index Manager

Central coordinator for VectorStore and all indexes.
The API layer uses this manager — it never manipulates indexes directly.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.core.exceptions import (
    IndexNotBuiltError,
    InvalidSearchParameterError,
)
from app.core.types import IndexState, SearchResult
from app.core.vector_store import VectorStore
from app.indexes.brute_force import BruteForceIndex
from app.indexes.hnsw import HNSWIndex
from app.indexes.ivf import IVFIndex
from app.config import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    IVF_NLIST,
    IVF_NPROBE,
    VECTOR_DIM,
)


class IndexManager:
    """
    Manages the VectorStore and all three indexes.

    Responsibilities:
        - Load/build indexes
        - Coordinate insertion
        - Coordinate deletion
        - Route search to the correct index
        - Expose stats
        - Mark stale indexes
    """

    def __init__(self, dimension: int = VECTOR_DIM):
        self.store = VectorStore(dimension=dimension)
        self.brute = BruteForceIndex()
        self.ivf = IVFIndex(nlist=IVF_NLIST, nprobe=IVF_NPROBE)
        self.hnsw = HNSWIndex(m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH)

        self._indexes = {
            "brute": self.brute,
            "ivf": self.ivf,
            "hnsw": self.hnsw,
        }

    def insert(
        self,
        vector_id: str,
        vector: list | np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Insert a vector and mark approximate indexes as dirty."""
        self.store.insert(vector_id, vector, metadata)
        self.ivf.mark_dirty()
        self.hnsw.mark_dirty()

    def insert_vector(
        self,
        vector_id: str,
        vector: list | np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Alias for insert."""
        self.insert(vector_id, vector, metadata)

    def bulk_insert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict | None] | None = None,
    ) -> int:
        """Bulk insert vectors and mark approximate indexes dirty."""
        count = self.store.bulk_insert(ids, vectors, metadatas)
        self.ivf.mark_dirty()
        self.hnsw.mark_dirty()
        return count

    def get_vector(self, vector_id: str) -> dict:
        """Retrieve vector dictionary by ID."""
        return self.store.get(vector_id)

    def delete(self, vector_id: str) -> None:
        """Mark a vector as deleted across store and all indexes."""
        self.store.delete(vector_id)

    def delete_vector(self, vector_id: str) -> None:
        """Alias for delete."""
        self.delete(vector_id)

    def search(
        self,
        query: list | np.ndarray,
        k: int = 10,
        index: str = "brute",
        **kwargs,
    ) -> list[SearchResult]:
        """
        Search using the specified index.

        Parameters
        ----------
        query : vector
        k     : number of results
        index : "brute", "ivf", or "hnsw"
        **kwargs : index-specific params (nprobe, ef_search)
        """
        if index not in self._indexes:
            raise InvalidSearchParameterError(
                "index", index, f"must be one of {list(self._indexes.keys())}"
            )

        idx = self._indexes[index]
        return idx.search(query, k=k, **kwargs)

    def build_all(self) -> dict[str, float]:
        """Build all indexes. Returns build times in seconds."""
        times = {}
        for name, idx in self._indexes.items():
            start = time.perf_counter()
            idx.build(self.store.vectors, self.store.ids, self.store.active_mask)
            times[name] = time.perf_counter() - start
        return times

    def build_index(self, name: str) -> None:
        """Build a specific index."""
        if name not in self._indexes:
            raise InvalidSearchParameterError(
                "index", name, f"must be one of {list(self._indexes.keys())}"
            )
        idx = self._indexes[name]
        idx.build(self.store.vectors, self.store.ids, self.store.active_mask)

    def stats(self) -> dict[str, Any]:
        """Return system-wide statistics."""
        return {
            "vectors": self.store.count(),
            "total_vectors": self.store.total_count(),
            "dimension": self.store.dimension,
            "indexes": {
                name: idx.state.value
                for name, idx in self._indexes.items()
            },
        }

    def load_data(self, directory: str) -> None:
        """Load vectors from disk and build all indexes."""
        self.store.load(directory)
        self.build_all()
