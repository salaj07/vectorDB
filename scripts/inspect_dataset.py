"""
VectorForge Script — Inspect Vector Datasets

Provides an interactive CLI summary to inspect the 50,000 synthetic dataset
or the 5,000 text dataset.

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py data_text
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def inspect_dir(target_dir: str = "data", sample_count: int = 3):
    print("=" * 80)
    print(f"               VectorForge — Dataset Inspector ({target_dir}/)")
    print("=" * 80)

    vec_file = os.path.join(target_dir, "vectors.npy")
    ids_file = os.path.join(target_dir, "ids.npy")
    queries_file = os.path.join(target_dir, "queries.npy")
    meta_file = os.path.join(target_dir, "metadata.json")
    gt_file = os.path.join(target_dir, "ground_truth.json")

    # 1. Vectors
    if os.path.exists(vec_file):
        vectors = np.load(vec_file)
        size_mb = os.path.getsize(vec_file) / (1024 * 1024)
        print(f"\n[1] Vectors Array:        '{vec_file}'")
        print(f"    • Total Vectors:      {vectors.shape[0]:,}")
        print(f"    • Dimension (D):      {vectors.shape[1]}")
        print(f"    • Data Type:          {vectors.dtype}")
        print(f"    • File Size:          {size_mb:.2f} MB")
        print(f"    • Memory in RAM:      {(vectors.nbytes / (1024*1024)):.2f} MB")
        print(f"    • Norm check (avg):   {np.mean(np.linalg.norm(vectors[:100], axis=1)):.4f}")
    else:
        print(f"\n[1] Vectors Array:        NOT FOUND in '{target_dir}/'")

    # 2. Query Set
    if os.path.exists(queries_file):
        queries = np.load(queries_file)
        print(f"\n[2] Queries Array:        '{queries_file}'")
        print(f"    • Total Queries:      {queries.shape[0]:,}")
        print(f"    • Query Dimension:    {queries.shape[1]}")

    # 3. Ground Truth
    if os.path.exists(gt_file):
        with open(gt_file, "r", encoding="utf-8") as f:
            gt = json.load(f)
        print(f"\n[3] Ground Truth Baseline: '{gt_file}'")
        if isinstance(gt, dict):
            k_len = len(list(gt.values())[0]) if gt else 0
        elif isinstance(gt, list):
            k_len = len(gt[0]) if gt else 0
        else:
            k_len = 0
        print(f"    • Neighbors / Query:  {k_len}")

    # 4. Metadata & Samples
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"\n[4] Metadata Store:       '{meta_file}' ({len(meta):,} entries)")
        
        # Category breakdown if present
        categories = {}
        for m in meta.values():
            cat = m.get("category", f"Cluster {m.get('cluster', 'N/A')}")
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"    • Categories/Clusters ({len(categories)} total):")
        for cat, count in list(categories.items())[:6]:
            print(f"        - {cat}: {count:,} items")
        if len(categories) > 6:
            print(f"        - ... and {len(categories) - 6} more")

        print(f"\n[5] Sample Vector Entries (First {sample_count}):")
        for i, (vid, m) in enumerate(list(meta.items())[:sample_count], 1):
            if "text" in m:
                print(f"    {i}. ID: {vid} | [{m.get('category', 'N/A')}]")
                print(f"       \"{m.get('text')}\"")
            else:
                print(f"    {i}. ID: {vid} | Cluster: {m.get('cluster')} | Center Sim: {m.get('cluster_center_sim', 'N/A')}")
                if os.path.exists(vec_file):
                    print(f"       Raw Coordinates (first 5 dims): {vectors[i-1][:5].round(4)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "data"
    inspect_dir(target)
