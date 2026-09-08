"""
VectorForge Script — Build All Indexes

Loads vectors from data/ and builds Brute Force, IVF, and HNSW indexes.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR
from app.indexes.manager import IndexManager


def main():
    if not os.path.exists(os.path.join(DATA_DIR, "vectors.npy")):
        print(f"Error: Dataset not found in '{DATA_DIR}'. Run 'python scripts/generate_dataset.py' first.")
        return

    manager = IndexManager()
    print("Loading vectors into VectorStore...")
    manager.store.load(DATA_DIR)
    print(f"Loaded {manager.store.count()} active vectors (dim={manager.store.dimension}).")

    print("\nBuilding indexes...")
    times = manager.build_all()

    for idx_name, build_time in times.items():
        print(f"  [{idx_name.upper()}] built in {build_time:.4f}s")

    stats = manager.stats()
    print("\nIndex Status:")
    for idx_name, state in stats["indexes"].items():
        print(f"  - {idx_name}: {state}")


if __name__ == "__main__":
    main()
