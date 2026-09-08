# VectorForge — Custom Vector Database from Scratch

VectorForge is a production-grade, zero-dependency vector database engine built entirely from scratch in Python and NumPy. It implements exact nearest-neighbor search alongside two fundamental approximate nearest-neighbor (ANN) index architectures: **IVF-Flat** (Inverted File Index with K-Means clustering) and **HNSW** (Hierarchical Navigable Small World graphs).

---

## Key Features

- **Exact & Approximate Nearest Neighbor Search**:
  - **Brute Force**: Exhaustive $O(N \times D)$ cosine similarity scan — ground truth reference.
  - **IVF-Flat**: Inverted File Index with custom $K$-Means clustering and centroid-based candidate pruning.
  - **HNSW**: Multi-layer skip-graph ANN index with greedy top-layer navigation and $ef$-bounded candidate search.
- **In-Memory Vector Store**:
  - Direct NumPy array storage ($N \times D$ float32) with automatic vector normalization on ingestion.
  - Logical soft-deletion support (`active_mask` filtering).
  - Fast lookup map ($O(1)$ ID to row index) and metadata handling.
  - Native binary persistence (`.npy` array serialization and JSON metadata).
- **FastAPI REST Service**:
  - Vector CRUD operations (`POST /vectors`, `POST /vectors/bulk`, `GET /vectors/{id}`, `DELETE /vectors/{id}`).
  - Parameterized multi-index search endpoint (`POST /search`).
  - System health check, index stats, and dynamically requested index rebuilding (`POST /rebuild/{index}`).
- **Evaluation & Benchmarking Suite**:
  - Automated synthetic clustered dataset generator.
  - Ground truth calculation for recall computation (`Recall@K`).
  - Detailed latency statistics ($p50$, $p95$, $p99$), build times, and relative speedup reporting.

---

## Architecture Overview

```
                        ┌────────────────────────┐
                        │     FastAPI Layer      │
                        │ (routes, schemas, app) │
                        └───────────┬────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │      IndexManager      │
                        └─────┬──────┬──────┬────┘
                              │      │      │
       ┌──────────────────────┘      │      └──────────────────────┐
       │                             │                             │
┌──────▼─────────┐          ┌────────▼────────┐           ┌────────▼────────┐
│  Brute Force   │          │    IVF-Flat     │           │      HNSW       │
│  (Exact Baseline)         │ (K-Means/Lists) │           │  (Multi-Layer)  │
└──────┬─────────┘          └────────┬────────┘           └────────┬────────┘
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     │
                        ┌────────────▼───────────┐
                        │      VectorStore       │
                        │ (NumPy N×D float32)    │
                        └────────────────────────┘
```

---

## Directory Structure

```
vectordb/
├── app/
│   ├── algorithms/
│   │   ├── heap.py          # Min/Max heap wrappers for candidate tracking
│   │   └── kmeans.py        # Manual K-Means clustering algorithm
│   ├── api/
│   │   ├── routes_search.py # POST /search endpoint
│   │   ├── routes_system.py # GET /health, GET /stats, POST /rebuild
│   │   └── routes_vectors.py# Vector CRUD endpoints
│   ├── core/
│   │   ├── distance.py      # L2 normalization & Cosine similarity
│   │   ├── exceptions.py    # Custom domain exceptions
│   │   ├── topk.py          # Partial sort Top-K selector via argpartition
│   │   ├── types.py         # SearchResult & IndexState dataclasses
│   │   └── vector_store.py  # In-memory NumPy vector storage
│   ├── evaluation/
│   │   ├── benchmark.py     # Latency & performance benchmark runner
│   │   ├── ground_truth.py  # Exact nearest neighbor ground truth generator
│   │   └── recall.py        # Recall@K calculation
│   ├── indexes/
│   │   ├── base.py          # Base index interface
│   │   ├── brute_force.py   # Brute force search index
│   │   ├── hnsw.py          # Hierarchical Navigable Small World index
│   │   ├── ivf.py           # IVF-Flat index
│   │   └── manager.py       # Index coordinator
│   ├── schemas/             # Pydantic request/response models
│   ├── config.py            # Global default configurations
│   └── main.py              # FastAPI app initialization
├── scripts/
│   ├── benchmark.py         # Benchmark execution script
│   ├── build_indexes.py     # Index building script
│   ├── generate_dataset.py  # Synthetic dataset generator
│   ├── generate_ground_truth.py
│   └── test_suite.py        # Master test runner & verification
├── tests/
│   ├── api/                 # FastAPI integration tests
│   ├── integration/         # Cross-index correctness tests
│   └── unit/                # Core module unit tests
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Requirements & Setup

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Run the Full Test & Verification Suite

Execute the master verification runner:

```bash
python scripts/test_suite.py
```

This automatically runs all unit tests, integration tests, API tests, dataset generation, index building, and benchmarks.

### 3. Run Pytest Directly

```bash
python -m pytest tests/ -v
```

---

## Benchmark & Performance Evaluation

Run performance benchmarks against a synthetic clustered dataset ($N=1000, D=128$):

```bash
python scripts/generate_dataset.py -n 1000 -d 128
python scripts/benchmark.py
```

### Sample Output Matrix

| Index Name | Parameters | Recall@10 | Mean Latency (ms) | Speedup vs Brute |
|---|---|---|---|---|
| Brute Force | Exact | 100.0% | 0.42 ms | 1.00x |
| IVF-Flat | nprobe=1 | 78.4% | 0.08 ms | 5.25x |
| IVF-Flat | nprobe=5 | 96.2% | 0.19 ms | 2.21x |
| HNSW | ef=10 | 85.1% | 0.06 ms | 7.00x |
| HNSW | ef=50 | 98.6% | 0.14 ms | 3.00x |

---

## Running the API Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

Access Swagger UI interactive documentation at:
`http://localhost:8000/docs`

### Core API Examples

#### Insert Single Vector
```bash
curl -X POST "http://localhost:8000/vectors" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "v1",
    "vector": [0.1, 0.2, ..., 0.5],
    "metadata": {"category": "tech"}
  }'
```

#### Perform Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, ..., 0.5],
    "k": 5,
    "index": "hnsw",
    "ef_search": 50
  }'
```

---

## License

MIT License. Designed and engineered for production-grade educational & benchmark use.
