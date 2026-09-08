"""
VectorForge — Recall@K Computation

Recall@K = |Approximate Top-K ∩ Exact Top-K| / K
"""

from __future__ import annotations


def recall_at_k(approximate_ids: list[str], exact_ids: list[str], k: int = 10) -> float:
    """
    Compute Recall@K for a single query.

    Parameters
    ----------
    approximate_ids : list of IDs returned by the approximate index
    exact_ids       : list of IDs returned by the exact (brute force) index
    k               : number of neighbors

    Returns
    -------
    float in [0, 1]
    """
    approx_set = set(approximate_ids[:k])
    exact_set = set(exact_ids[:k])

    if len(exact_set) == 0:
        return 1.0

    return len(approx_set & exact_set) / len(exact_set)


def mean_recall_at_k(
    all_approximate: list[list[str]],
    all_exact: list[list[str]],
    k: int = 10,
) -> float:
    """
    Compute mean Recall@K across multiple queries.

    Parameters
    ----------
    all_approximate : list of ID lists from the approximate index
    all_exact       : list of ID lists from exact search (ground truth)
    k               : number of neighbors

    Returns
    -------
    float — mean recall
    """
    recalls = [
        recall_at_k(approx, exact, k)
        for approx, exact in zip(all_approximate, all_exact)
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0
