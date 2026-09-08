"""
VectorForge — Benchmarking Utilities

Measure build time, query latency, recall, speedup, and memory.
"""

from __future__ import annotations

import time
import sys
from typing import Any, Callable

import numpy as np

from app.core.types import SearchResult
from app.evaluation.recall import recall_at_k


def measure_latency(
    search_fn: Callable[..., list[SearchResult]],
    queries: np.ndarray,
    k: int = 10,
    warmup: int = 10,
    **search_kwargs,
) -> dict[str, float]:
    """
    Measure search latency statistics.

    Parameters
    ----------
    search_fn    : callable that takes (query, k, **kwargs) and returns results
    queries      : (Q, D) query vectors
    k            : number of results per query
    warmup       : number of warmup queries to skip
    **search_kwargs : extra params like nprobe or ef_search

    Returns
    -------
    dict with avg_ms, p50_ms, p95_ms, min_ms, max_ms
    """
    # Warmup
    for i in range(min(warmup, len(queries))):
        search_fn(queries[i], k=k, **search_kwargs)

    latencies = []
    for query in queries:
        start = time.perf_counter()
        search_fn(query, k=k, **search_kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "total_queries": len(queries),
    }


def benchmark_index(
    index: Any,
    queries: np.ndarray,
    ground_truth: dict[int, list[str]],
    k: int = 10,
    name: str = "",
    memory_mb: float | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Benchmark an index on latency, recall@k, and memory.
    """
    stats = measure_latency(index.search, queries, k=k, **kwargs)

    # Compute average recall across queries
    recalls = []
    for idx, q_vec in enumerate(queries):
        res = index.search(q_vec, k=k, **kwargs)
        approx_ids = [r.id for r in res]
        if isinstance(ground_truth, list):
            gt_ids = ground_truth[idx] if idx < len(ground_truth) else []
        elif isinstance(ground_truth, dict):
            gt_ids = ground_truth.get(idx, ground_truth.get(str(idx), []))
        else:
            gt_ids = []
        recalls.append(recall_at_k(approx_ids, gt_ids, k=k))

    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    return {
        "name": name or getattr(index, "name", "Index"),
        "mean_ms": stats["avg_ms"],
        "p50_ms": stats["p50_ms"],
        "p95_ms": stats["p95_ms"],
        "p99_ms": stats["p99_ms"],
        "recall": avg_recall,
        "memory_mb": memory_mb,
    }


def format_benchmark_results(
    results: list[dict[str, Any]],
    brute_latency_ms: float | None = None,
) -> str:
    """Format benchmark results into a clean markdown / plain text table."""
    lines = []
    has_memory = any(r.get("memory_mb") is not None for r in results)

    if has_memory:
        header = f"{'Index Name':<30} {'Recall@10':<10} {'Mean (ms)':<10} {'p50 (ms)':<10} {'p95 (ms)':<10} {'p99 (ms)':<10} {'Speedup':<8} {'Memory':<10}"
    else:
        header = f"{'Index Name':<30} {'Recall@10':<10} {'Mean (ms)':<10} {'p50 (ms)':<10} {'p95 (ms)':<10} {'p99 (ms)':<10} {'Speedup':<8}"
    line_len = len(header)

    lines.append("=" * line_len)
    lines.append(header)
    lines.append("-" * line_len)

    base_lat = brute_latency_ms or (results[0]["mean_ms"] if results else 1.0)

    for r in results:
        speedup = base_lat / r["mean_ms"] if r["mean_ms"] > 0 else 0.0
        p99_str = f"{r.get('p99_ms', 0.0):>9.4f}"
        base_str = (
            f"{r['name']:<30} {r['recall']*100:>8.2f}%  {r['mean_ms']:>9.4f}  "
            f"{r['p50_ms']:>9.4f}  {r['p95_ms']:>9.4f}  {p99_str}  {speedup:>7.2f}x"
        )
        if has_memory:
            mem_val = r.get("memory_mb")
            mem_str = f"{mem_val:>8.2f}MB" if mem_val is not None else "       N/A"
            lines.append(f"{base_str}  {mem_str}")
        else:
            lines.append(base_str)

    lines.append("=" * line_len)
    return "\n".join(lines)


def measure_build_time(build_fn: Callable, *args, **kwargs) -> tuple[float, Any]:
    """Measure index build time."""
    start = time.perf_counter()
    result = build_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def estimate_memory(vectors: np.ndarray, extra_bytes: int = 0) -> dict[str, float]:
    """Estimate memory usage."""
    vec_bytes = vectors.nbytes
    return {
        "vectors_mb": vec_bytes / (1024 * 1024),
        "extra_mb": extra_bytes / (1024 * 1024),
        "total_mb": (vec_bytes + extra_bytes) / (1024 * 1024),
    }
