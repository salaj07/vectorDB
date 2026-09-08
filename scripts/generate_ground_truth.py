"""
VectorForge Script — Generate Ground Truth Results

Runs exact Brute Force search on generated queries to establish ground truth.

Outputs:
    data/ground_truth.json
"""

import os
import json
import sys
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import DATA_DIR, TOP_K
from app.evaluation.ground_truth import generate_ground_truth
from app.indexes.manager import IndexManager


def main():
    vec_path = os.path.join(DATA_DIR, "vectors.npy")
    queries_path = os.path.join(DATA_DIR, "queries.npy")

    if not os.path.exists(vec_path) or not os.path.exists(queries_path):
        print(f"Error: Dataset not found in '{DATA_DIR}'. Run 'python scripts/generate_dataset.py' first.")
        return

    manager = IndexManager()
    manager.store.load(DATA_DIR)

    queries = np.load(queries_path)
    print(f"Loaded {len(queries)} query vectors. Computing ground truth (k={TOP_K})...")

    ground_truth = generate_ground_truth(manager.store, queries, k=TOP_K)

    gt_path = os.path.join(DATA_DIR, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Saved ground truth for {len(ground_truth)} queries to '{gt_path}'.")


if __name__ == "__main__":
    main()
