"""
VectorForge Interactive Text Similarity Search Demo

Searches across 5,000 embedded text statements using Brute Force, IVF-Flat, and HNSW indexes.

Usage:
    python scripts/demo_text_search.py "express js framework"
    python scripts/demo_text_search.py "docker containers"
    python scripts/demo_text_search.py "human emotions and empathy"
"""

import os
import sys
import hashlib
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR
from app.core.distance import normalize
from app.indexes.manager import IndexManager
from scripts.generate_text_dataset import text_to_vector, generate_5k_text_dataset


def main():
    print("=" * 80)
    print("        VectorForge — Interactive Text Similarity Search Demo (5,000 Texts)")
    print("=" * 80)

    TEXT_DATA_DIR = "data_text"
    target_dir = TEXT_DATA_DIR if os.path.exists(os.path.join(TEXT_DATA_DIR, "vectors.npy")) else DATA_DIR

    # Ensure dataset exists
    vec_path = os.path.join(target_dir, "vectors.npy")
    if not os.path.exists(vec_path):
        generate_5k_text_dataset(total_count=5000, output_dir=target_dir)

    # Initialize IndexManager and load 5,000 text vectors
    manager = IndexManager(dimension=128)
    print(f"\nLoading text statements from '{target_dir}/' into VectorStore...")
    manager.load_data(target_dir)
    print(f"Loaded {manager.store.count()} text vectors (dim={manager.store.dimension}).")

    # Get search query from CLI arguments or default
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = "express js framework for node backend"

    print(f"\nQuery Statement: \"{query_text}\"")
    print("-" * 80)

    # Embed query statement into 128D vector
    query_vec = text_to_vector(query_text, dim=128)

    # Search across all 3 indexes
    for index_name in ["brute", "ivf", "hnsw"]:
        results = manager.search(query_vec, k=3, index=index_name)
        print(f"\n--- [{index_name.upper()} INDEX RESULTS] ---")
        for rank, res in enumerate(results, 1):
            doc = manager.store.get(res.id)
            meta = doc.get("metadata") or {}
            text_str = meta.get("text", f"Vector {res.id}")
            cat_str = meta.get("category", f"Cluster {meta.get('cluster', 'N/A')}")
            print(f"  {rank}. [Score: {res.score:.4f}] Category: {cat_str}")
            print(f"     \"{text_str}\"")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
