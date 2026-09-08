"""
VectorForge Script — Generate Synthetic Dataset

Generates synthetic clustered vector datasets for benchmarking and testing.

Outputs:
    data/vectors.npy
    data/ids.npy
    data/queries.npy
    data/metadata.json
"""

import argparse
import os
import json
import sys
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import (
    DATA_DIR,
    DEFAULT_N,
    DEFAULT_D,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_NUM_QUERIES,
)
from app.core.distance import normalize


def generate_clustered_dataset(
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    num_clusters: int = DEFAULT_NUM_CLUSTERS,
    num_queries: int = DEFAULT_NUM_QUERIES,
    seed: int = 42,
):
    """Generate synthetic vectors with cluster structure."""
    rng = np.random.default_rng(seed)

    print(f"Generating dataset: N={n}, D={d}, Clusters={num_clusters}, Queries={num_queries}...")

    # Generate cluster centers
    centers = rng.standard_normal((num_clusters, d)).astype(np.float32)
    centers = normalize(centers)

    # Assign samples per cluster
    samples_per_cluster = n // num_clusters
    remainder = n % num_clusters

    vectors = []
    metadata = {}
    ids = []

    count = 0
    for c_idx in range(num_clusters):
        num_samples = samples_per_cluster + (1 if c_idx < remainder else 0)
        # Cluster vectors = center + small noise
        noise = rng.normal(scale=0.1, size=(num_samples, d)).astype(np.float32)
        cluster_vecs = centers[c_idx] + noise
        cluster_vecs = normalize(cluster_vecs)

        for vec in cluster_vecs:
            vid = f"vec_{count}"
            ids.append(vid)
            vectors.append(vec)
            metadata[vid] = {"cluster": c_idx, "index": count}
            count += 1

    vectors = np.array(vectors, dtype=np.float32)
    ids_arr = np.array(ids, dtype=object)

    # Generate query vectors (mix of near-cluster and random queries)
    query_centers = centers[rng.choice(num_clusters, size=num_queries)]
    query_noise = rng.normal(scale=0.15, size=(num_queries, d)).astype(np.float32)
    queries = normalize(query_centers + query_noise)

    os.makedirs(DATA_DIR, exist_ok=True)

    np.save(os.path.join(DATA_DIR, "vectors.npy"), vectors)
    np.save(os.path.join(DATA_DIR, "ids.npy"), ids_arr)
    np.save(os.path.join(DATA_DIR, "queries.npy"), queries)

    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to '{DATA_DIR}':")
    print(f"  vectors.npy  : {vectors.shape}")
    print(f"  ids.npy      : {ids_arr.shape}")
    print(f"  queries.npy  : {queries.shape}")
    print(f"  metadata.json: {len(metadata)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VectorForge Dataset Generator")
    parser.add_argument("-n", "--num-vectors", type=int, default=DEFAULT_N, help="Number of vectors")
    parser.add_argument("-d", "--dimension", type=int, default=DEFAULT_D, help="Vector dimension")
    parser.add_argument("-c", "--clusters", type=int, default=DEFAULT_NUM_CLUSTERS, help="Number of clusters")
    parser.add_argument("-q", "--queries", type=int, default=DEFAULT_NUM_QUERIES, help="Number of query vectors")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    generate_clustered_dataset(
        n=args.num_vectors,
        d=args.dimension,
        num_clusters=args.clusters,
        num_queries=args.queries,
        seed=args.seed,
    )
