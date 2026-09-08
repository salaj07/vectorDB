"""
VectorForge — In-Memory Vector Store

Central storage layer backed by NumPy arrays.
All indexes reference this store rather than maintaining their own copies.

Layout:
    vectors     : np.ndarray  (N, D) float32 — normalized embeddings
    ids         : np.ndarray  (N,)           — string identifiers
    metadata    : dict[str, dict]            — optional per-vector metadata
    active_mask : np.ndarray  (N,) bool      — True = active, False = deleted
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.core.distance import normalize
from app.core.exceptions import (
    DimensionMismatchError,
    DuplicateVectorError,
    VectorNotFoundError,
)


class VectorStore:
    """
    In-memory vector storage with logical deletion and amortized O(1) insertion.

    Backing memory is pre-allocated with capacity-doubling reallocation.
    Vectors are normalized on insertion so that cosine similarity
    reduces to a simple dot product at query time.
    """

    def __init__(self, dimension: int, initial_capacity: int = 1024):
        self.dimension: int = dimension
        self._capacity: int = max(initial_capacity, 16)
        self._size: int = 0

        self._buf: np.ndarray = np.empty((self._capacity, dimension), dtype=np.float32)
        self._ids_buf: np.ndarray = np.empty(self._capacity, dtype=object)
        self._mask_buf: np.ndarray = np.ones(self._capacity, dtype=bool)
        self.metadata: dict[str, dict] = {}

        # Fast ID → row-index lookup
        self._id_to_idx: dict[str, int] = {}

    def _resize(self, min_capacity: int) -> None:
        """Double the buffer capacity or grow to at least min_capacity."""
        new_capacity = max(self._capacity * 2, min_capacity, 16)
        new_buf = np.empty((new_capacity, self.dimension), dtype=np.float32)
        new_ids = np.empty(new_capacity, dtype=object)
        new_mask = np.ones(new_capacity, dtype=bool)

        if self._size > 0:
            new_buf[: self._size] = self._buf[: self._size]
            new_ids[: self._size] = self._ids_buf[: self._size]
            new_mask[: self._size] = self._mask_buf[: self._size]

        self._buf = new_buf
        self._ids_buf = new_ids
        self._mask_buf = new_mask
        self._capacity = new_capacity

    @property
    def vectors(self) -> np.ndarray:
        """Zero-copy view of live vector data."""
        return self._buf[: self._size]

    @vectors.setter
    def vectors(self, val: np.ndarray) -> None:
        val = np.asarray(val, dtype=np.float32)
        self._size = len(val)
        self._capacity = max(1024, self._size)
        self._buf = np.empty((self._capacity, self.dimension), dtype=np.float32)
        if self._size > 0:
            self._buf[: self._size] = val

    @property
    def ids(self) -> np.ndarray:
        """Zero-copy view of vector IDs."""
        return self._ids_buf[: self._size]

    @ids.setter
    def ids(self, val: np.ndarray) -> None:
        val = np.asarray(val, dtype=object)
        if len(val) > self._capacity:
            self._resize(len(val))
        self._ids_buf[: len(val)] = val
        self._size = max(self._size, len(val))

    @property
    def active_mask(self) -> np.ndarray:
        """Zero-copy view of the active (non-deleted) boolean mask."""
        return self._mask_buf[: self._size]

    @active_mask.setter
    def active_mask(self, val: np.ndarray) -> None:
        val = np.asarray(val, dtype=bool)
        if len(val) > self._capacity:
            self._resize(len(val))
        self._mask_buf[: len(val)] = val
        self._size = max(self._size, len(val))

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert(
        self,
        vector_id: str,
        vector: np.ndarray | list,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Insert a vector. Normalizes it automatically with amortized O(1) cost."""
        vector_id = str(vector_id)

        if vector_id in self._id_to_idx:
            raise DuplicateVectorError(vector_id)

        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.dimension:
            raise DimensionMismatchError(self.dimension, vec.shape[-1] if vec.size else 0)

        # Normalize once at ingestion
        vec = normalize(vec)

        if self._size + 1 > self._capacity:
            self._resize(self._size + 1)

        idx = self._size
        self._buf[idx] = vec
        self._ids_buf[idx] = vector_id
        self._mask_buf[idx] = True
        self._id_to_idx[vector_id] = idx
        self._size += 1

        if meta:
            self.metadata[vector_id] = meta

    # ------------------------------------------------------------------
    # Bulk load (for dataset generation / startup)
    # ------------------------------------------------------------------
    def bulk_insert(
        self,
        ids: np.ndarray | list,
        vectors: np.ndarray,
        metadata_list: list[dict] | None = None,
    ) -> int:
        """
        Insert many vectors at once — vectorized batch ingestion.

        Parameters
        ----------
        ids      : array-like of string IDs, length N
        vectors  : np.ndarray shape (N, D), will be normalized
        metadata_list : optional list of dicts, length N
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise DimensionMismatchError(
                self.dimension,
                vectors.shape[1] if vectors.ndim == 2 else -1,
            )

        ids_arr = np.asarray(ids, dtype=object)
        if len(ids_arr) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        # Check for duplicates before mutating state
        seen: set[str] = set()
        for vid in ids_arr:
            vid_str = str(vid)
            if vid_str in self._id_to_idx or vid_str in seen:
                raise DuplicateVectorError(vid_str)
            seen.add(vid_str)

        num_new = len(ids_arr)
        if num_new == 0:
            return 0

        # Normalize batch
        normed = normalize(vectors)

        if self._size + num_new > self._capacity:
            self._resize(self._size + num_new)

        start = self._size
        end = start + num_new

        self._buf[start:end] = normed
        self._ids_buf[start:end] = ids_arr
        self._mask_buf[start:end] = True

        for i, vid in enumerate(ids_arr):
            vid_str = str(vid)
            self._id_to_idx[vid_str] = start + i
            if metadata_list and i < len(metadata_list) and metadata_list[i]:
                self.metadata[vid_str] = metadata_list[i]

        self._size = end
        return num_new

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def get(self, vector_id: str) -> dict:
        """Return vector data by ID. Raises VectorNotFoundError if missing."""
        vector_id = str(vector_id)
        if vector_id not in self._id_to_idx:
            raise VectorNotFoundError(vector_id)

        idx = self._id_to_idx[vector_id]
        return {
            "id": vector_id,
            "vector": self._buf[idx].copy(),
            "active": bool(self._mask_buf[idx]),
            "metadata": self.metadata.get(vector_id, None),
        }

    # ------------------------------------------------------------------
    # Delete (logical)
    # ------------------------------------------------------------------
    def delete(self, vector_id: str) -> None:
        """Mark a vector as deleted (logical deletion)."""
        vector_id = str(vector_id)
        if vector_id not in self._id_to_idx:
            raise VectorNotFoundError(vector_id)

        idx = self._id_to_idx[vector_id]
        self._mask_buf[idx] = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_active_vectors(self) -> np.ndarray:
        """Return (M, D) array of active vectors only."""
        return self.vectors[self.active_mask]

    def get_active_ids(self) -> np.ndarray:
        """Return IDs of active vectors only."""
        return self.ids[self.active_mask]

    def get_all_vectors(self) -> np.ndarray:
        """Return all vectors (including deleted)."""
        return self.vectors

    def get_all_ids(self) -> np.ndarray:
        """Return all IDs (including deleted)."""
        return self.ids

    def count(self) -> int:
        """Return the number of active (non-deleted) vectors."""
        return int(np.sum(self._mask_buf[: self._size]))

    def total_count(self) -> int:
        """Return total number of vectors including deleted."""
        return self._size

    def is_empty(self) -> bool:
        return self.total_count() == 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        """Save vectors, IDs, and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        np.save(os.path.join(directory, "ids.npy"), self.ids)
        np.save(os.path.join(directory, "active_mask.npy"), self.active_mask)
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, directory: str) -> None:
        """Load vectors, IDs, and metadata from disk."""
        loaded_vecs = np.load(
            os.path.join(directory, "vectors.npy"), allow_pickle=False
        ).astype(np.float32)
        self.dimension = loaded_vecs.shape[1] if len(loaded_vecs) > 0 else self.dimension
        self._size = len(loaded_vecs)
        self._capacity = max(1024, self._size)

        self._buf = np.empty((self._capacity, self.dimension), dtype=np.float32)
        if self._size > 0:
            self._buf[: self._size] = loaded_vecs

        self._ids_buf = np.empty(self._capacity, dtype=object)
        if self._size > 0:
            loaded_ids = np.load(os.path.join(directory, "ids.npy"), allow_pickle=True)
            self._ids_buf[: self._size] = loaded_ids

        self._mask_buf = np.ones(self._capacity, dtype=bool)
        active_path = os.path.join(directory, "active_mask.npy")
        if os.path.exists(active_path) and self._size > 0:
            self._mask_buf[: self._size] = np.load(
                active_path, allow_pickle=False
            ).astype(bool)

        meta_path = os.path.join(directory, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Rebuild index
        self._id_to_idx = {str(vid): i for i, vid in enumerate(self.ids)}
