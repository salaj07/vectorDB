"""
VectorForge — Base Index Interface

All indexes (Brute Force, IVF, HNSW) implement this interface.
This prevents the API layer from containing index-specific logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.types import IndexState, SearchResult


class BaseIndex(ABC):
    """
    Abstract base class for all VectorForge indexes.

    Subclasses must implement build(), search(), delete(), and stats().
    """

    def __init__(self, name: str):
        self.name: str = name
        self._state: IndexState = IndexState.EMPTY

    @property
    def state(self) -> IndexState:
        return self._state

    @abstractmethod
    def build(self, vectors, ids, active_mask) -> None:
        """
        Build (or rebuild) the index from the current store data.

        Parameters
        ----------
        vectors     : np.ndarray (N, D) — normalized vectors
        ids         : np.ndarray (N,)   — string IDs
        active_mask : np.ndarray (N,)   — boolean active mask
        """
        ...

    @abstractmethod
    def search(self, query, k: int = 10, **kwargs) -> list[SearchResult]:
        """
        Search the index for the top-k most similar vectors.

        Parameters
        ----------
        query : np.ndarray (D,) — query vector (will be normalized internally)
        k     : int             — number of results
        **kwargs                — index-specific params (nprobe, ef_search, etc.)

        Returns
        -------
        list[SearchResult] sorted by score descending.
        """
        ...

    @abstractmethod
    def delete(self, vector_id: str) -> None:
        """Mark a vector as deleted in this index."""
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return index-specific statistics."""
        ...

    def mark_dirty(self) -> None:
        """Mark the index as needing a rebuild."""
        if self._state == IndexState.READY:
            self._state = IndexState.DIRTY
