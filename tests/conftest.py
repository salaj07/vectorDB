"""
VectorForge — Shared Test Fixtures

Reusable fixtures for unit, integration, and API tests.
Uses small datasets (N=100, D=16) for fast execution.
"""

import os
import numpy as np
import pytest

os.environ["VECTORFORGE_SKIP_AUTOLOAD"] = "1"

from app.core.vector_store import VectorStore
from app.core.distance import normalize
from app.indexes.brute_force import BruteForceIndex
from app.indexes.ivf import IVFIndex
from app.indexes.hnsw import HNSWIndex


# ---------------------------------------------------------------------------
# Small dataset constants
# ---------------------------------------------------------------------------
SMALL_N = 100
SMALL_D = 16
SMALL_Q = 10
SMALL_SEED = 42


@pytest.fixture
def small_vectors():
    """100 normalized vectors of dimension 16."""
    rng = np.random.default_rng(SMALL_SEED)
    vecs = rng.standard_normal((SMALL_N, SMALL_D)).astype(np.float32)
    return normalize(vecs)


@pytest.fixture
def small_ids():
    """100 string IDs."""
    return [f"vec_{i}" for i in range(SMALL_N)]


@pytest.fixture
def small_queries():
    """10 query vectors of dimension 16."""
    rng = np.random.default_rng(SMALL_SEED + 1)
    queries = rng.standard_normal((SMALL_Q, SMALL_D)).astype(np.float32)
    return normalize(queries)


@pytest.fixture
def vector_store(small_vectors, small_ids):
    """VectorStore populated with 100 vectors."""
    store = VectorStore(dimension=SMALL_D)
    store.bulk_insert(small_ids, small_vectors)
    return store


@pytest.fixture
def brute_index(vector_store):
    """BruteForceIndex built on the small dataset."""
    index = BruteForceIndex()
    index.build(vector_store.vectors, vector_store.ids, vector_store.active_mask)
    return index


@pytest.fixture
def ivf_index(vector_store):
    """IVFIndex built on the small dataset."""
    index = IVFIndex(nlist=10, nprobe=3)
    index.build(vector_store.vectors, vector_store.ids, vector_store.active_mask)
    return index


@pytest.fixture
def hnsw_index(vector_store):
    """HNSWIndex built on the small dataset."""
    index = HNSWIndex(m=8, ef_construction=30, ef_search=20)
    index.build(vector_store.vectors, vector_store.ids, vector_store.active_mask)
    return index
