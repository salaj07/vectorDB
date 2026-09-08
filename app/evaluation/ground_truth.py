"""
VectorForge — Ground Truth Generation

Generate exact top-k results using Brute Force for a set of queries.
These serve as the correctness baseline for evaluating ANN indexes.
"""

from __future__ import annotations

import numpy as np

from app.core.vector_store import VectorStore
from app.core.distance import normalize, cosine_similarity
from app.core.topk import top_k


def generate_ground_truth(
    store: VectorStore,
    queries: np.ndarray,
    k: int = 10,
) -> list[list[str]]:
    """
    Generate exact top-k neighbor IDs for each query using brute force.

    Parameters
    ----------
    store   : VectorStore with vectors loaded
    queries : (Q, D) normalized query vectors
    k       : number of neighbors per query

    Returns
    -------
    list[list[str]] — for each query, a list of top-k neighbor IDs.
    """
    queries = np.asarray(queries, dtype=np.float32)
    queries = normalize(queries)

    vectors = store.get_all_vectors()
    ids = store.get_all_ids()
    mask = store.active_mask

    ground_truth = []
    for query in queries:
        scores = cosine_similarity(query, vectors)
        results = top_k(scores, ids, k=k, mask=mask)
        gt_ids = [r.id for r in results]
        ground_truth.append(gt_ids)

    return ground_truth
