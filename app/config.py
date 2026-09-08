"""
VectorForge — Centralized Configuration

All configurable constants in one place.
Do not scatter magic numbers throughout the codebase.
"""

# ---------------------------------------------------------------------------
# Vector dimensions
# ---------------------------------------------------------------------------
VECTOR_DIM: int = 128
DEFAULT_D: int = VECTOR_DIM

# ---------------------------------------------------------------------------
# Search defaults
# ---------------------------------------------------------------------------
DEFAULT_K: int = 10
TOP_K: int = DEFAULT_K

# ---------------------------------------------------------------------------
# IVF-Flat parameters
# ---------------------------------------------------------------------------
IVF_NLIST: int = 100          # number of clusters (Voronoi cells)
IVF_NPROBE: int = 5           # clusters probed per query
IVF_KMEANS_MAX_ITER: int = 20 # K-Means iterations

# ---------------------------------------------------------------------------
# HNSW parameters
# ---------------------------------------------------------------------------
HNSW_M: int = 16              # max neighbors per node per layer
HNSW_EF_CONSTRUCTION: int = 100  # build-time search width
HNSW_EF_SEARCH: int = 50      # query-time search width

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
DATASET_SIZE: int = 50_000
DEFAULT_N: int = DATASET_SIZE
QUERY_SIZE: int = 500
DEFAULT_NUM_QUERIES: int = QUERY_SIZE
NUM_CLUSTERS: int = 20
DEFAULT_NUM_CLUSTERS: int = NUM_CLUSTERS
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: str = "data"
BENCHMARK_DIR: str = "benchmarks"


# ---------------------------------------------------------------------------
# Load .env file automatically
# ---------------------------------------------------------------------------
def _load_env_file(filepath: str = ".env") -> None:
    import os
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'").strip('"')
                        if key and key not in os.environ:
                            os.environ[key] = val
        except Exception:
            pass

_load_env_file()

