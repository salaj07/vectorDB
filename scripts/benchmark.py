"""
VectorForge Script — Comprehensive Benchmark Runner

Measures latency, Recall@K, and QPS for Brute Force, IVF, and HNSW indexes.
"""

import os
import json
import sys
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import DATA_DIR, TOP_K
from app.evaluation.benchmark import benchmark_index, format_benchmark_results
from app.evaluation.ground_truth import generate_ground_truth
from app.indexes.manager import IndexManager


def main():
    vec_path = os.path.join(DATA_DIR, "vectors.npy")
    queries_path = os.path.join(DATA_DIR, "queries.npy")

    if not os.path.exists(vec_path) or not os.path.exists(queries_path):
        print("Dataset not found. Generating dataset first...")
        from scripts.generate_dataset import generate_clustered_dataset
        generate_clustered_dataset()

    manager = IndexManager()
    manager.load_data(DATA_DIR)

    queries = np.load(queries_path)
    print(f"\nRunning VectorForge Benchmark on {len(queries)} queries (k={TOP_K})...")

    # Generate ground truth
    print("Computing exact ground truth baseline...")
    gt_dict = generate_ground_truth(manager.store, queries, k=TOP_K)

    # Memory calculations
    vec_mb = manager.store.vectors.nbytes / (1024 * 1024)
    ivf_extra = (
        manager.ivf.centroids.nbytes if manager.ivf.centroids is not None else 0
    ) + len(manager.store.ids) * 4
    ivf_mb = (manager.store.vectors.nbytes + ivf_extra) / (1024 * 1024)
    hnsw_edges = sum(
        len(nbrs)
        for node in manager.hnsw.neighbors.values()
        for nbrs in node.values()
    )
    hnsw_extra = hnsw_edges * 8 + len(manager.hnsw.neighbors) * 64
    hnsw_mb = (manager.store.vectors.nbytes + hnsw_extra) / (1024 * 1024)

    results = []

    # 1. Brute Force
    res_brute = benchmark_index(
        index=manager.brute,
        queries=queries,
        ground_truth=gt_dict,
        k=TOP_K,
        name="Brute Force (Exact)",
        memory_mb=vec_mb,
    )
    results.append(res_brute)

    # 2. IVF configurations
    for nprobe in [1, 2, 5, 10]:
        res_ivf = benchmark_index(
            index=manager.ivf,
            queries=queries,
            ground_truth=gt_dict,
            k=TOP_K,
            name=f"IVF-Flat (nprobe={nprobe})",
            memory_mb=ivf_mb,
            nprobe=nprobe,
        )
        results.append(res_ivf)

    # 3. HNSW configurations
    for ef_search in [10, 30, 50, 100]:
        res_hnsw = benchmark_index(
            index=manager.hnsw,
            queries=queries,
            ground_truth=gt_dict,
            k=TOP_K,
            name=f"HNSW (ef_search={ef_search})",
            memory_mb=hnsw_mb,
            ef_search=ef_search,
        )
        results.append(res_hnsw)

    # Format and print benchmark summary table
    table = format_benchmark_results(results, brute_latency_ms=res_brute["mean_ms"])
    print("\n" + table)


if __name__ == "__main__":
    main()
