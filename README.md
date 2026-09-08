# 🚀 VectorForge — High-Performance Vector Database from Scratch

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![NumPy](https://img.shields.io/badge/NumPy-Pure%20Math-013243.svg?logo=numpy&logoColor=white)](https://numpy.org)
[![Zero Dependencies](https://img.shields.io/badge/External%20ANN%20Libs-None%20(0)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**A production-grade, zero-dependency Vector Database engine built entirely from scratch in Python and NumPy.**  
*No FAISS • No Pinecone • No ChromaDB • No scikit-learn nearest neighbors.*

</div>

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Index Algorithms (Implemented From Scratch)](#-index-algorithms-implemented-from-scratch)
- [Project Directory Layout](#-project-directory-layout)
- [Getting Started & Installation](#-getting-started--installation)
- [Interactive Demonstrations](#-interactive-demonstrations)
  - [1. Performance & Recall Benchmarking](#1-performance--recall-benchmarking-50000-vectors)
  - [2. Multi-Index Text Search](#2-multi-index-text-similarity-search)
  - [3. Interactive RAG (Retrieval-Augmented Generation)](#3-interactive-rag-vector-db--gemini-llm)
  - [4. Self-Learning Semantic Cache (< 1ms Cache Hit)](#4-self-learning-semantic-cache--1ms-latency)
  - [5. Dataset Inspector](#5-dataset-inspector-50k--5k-datasets)
- [REST API Reference & Swagger UI](#-rest-api-reference--swagger-ui)
- [Benchmark Results & Recall Metrics](#-benchmark-results--recall-metrics)
- [Design Decisions: Graph Deletion](#-design-decisions-graph-deletion)

---

## 🌟 Overview

**VectorForge** provides exact and approximate nearest-neighbor (ANN) vector search, in-memory contiguous vector storage, dynamic indexing, automated clustering, REST API endpoints, and a self-learning **Semantic Cache RAG** architecture powered by Google Gemini.

All core algorithms — including cosine normalization, matrix-vector dot products, partial-sort Top-$K$ selection, $K$-Means clustering, Inverted File (IVF) posting lists, and Hierarchical Navigable Small World (HNSW) graph traversal — are written from first principles with pure **NumPy**.

---

## ⚡ Key Features

- **Pure NumPy Vector Math:** Zero third-party vector search libraries.
- **Three Search Index Architectures:**
  - **Exact Brute Force:** Exhaustive $\mathcal{O}(N \times D)$ cosine similarity scan as mathematical ground truth.
  - **IVF-Flat Index:** Custom $K$-Means Voronoi partitioning with candidate centroid pruning.
  - **HNSW Graph Index:** Multi-layer proximity graph with greedy upper-layer routing and $\text{ef}$-bounded Layer-0 search.
- **Contiguous In-Memory Storage:**
  - Packed $(N \times D)$ `float32` matrix storage with instantaneous L2 unit normalization.
  - $\mathcal{O}(1)$ string ID to matrix index mapping and metadata storage.
  - Native binary persistence (`.npy` array serialization + `.json` metadata).
- **FastAPI Production Service:** Full CRUD (`/vectors`, `/search`, `/chat/cached`, `/health`, `/stats`, `/rebuild`).
- **Self-Learning Semantic Cache:** Automatically caches LLM responses into vector memory with dual-layer persistent storage (`vectors.npy` + `llm_knowledge.json`), serving repeated queries with **< 1 ms latency**.
- **Rigorous Evaluation Suite:** Computes exact ground-truth baselines, Mean Recall@$K$, and latency percentiles ($p50$, $p95$, $p99$).

---

## 🏗 System Architecture

```
                                  ┌────────────────────────┐
                                  │      Client Layer      │
                                  │  (CLI Demos / Web API) │
                                  └───────────┬────────────┘
                                              │
                                  ┌───────────▼────────────┐
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
          │ (Exact Baseline│          │ (K-Means/Lists) │           │  (Multi-Layer)  │
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

## 🧠 Index Algorithms (Implemented From Scratch)

### 1. Brute-Force Index (Exact Ground Truth)
* Normalizes the query vector $\mathbf{q} \leftarrow \mathbf{q} / \|\mathbf{q}\|_2$.
* Computes cosine similarity via single-instruction matrix-vector dot product: $\mathbf{s} = \mathbf{V} \cdot \mathbf{q}$.
* Uses `np.argpartition` for fast $\mathcal{O}(N + K \log K)$ partial-sort Top-$K$ retrieval.

### 2. IVF-Flat Index (Inverted File Partitioning)
* **Custom K-Means:** Hand-written clustering algorithm that iteratively updates Voronoi centroids until convergence.
* **Inverted Lists:** Groups vector IDs into inverted posting lists indexed by cluster centroid ID.
* **Pruned Search:** At query time, finds the $n_\text{probe}$ closest centroids, gathers candidate vectors, and computes exact cosine distance on only the candidate subset.

### 3. HNSW (Hierarchical Navigable Small World Graph)
* **Multi-Layer Structure:** Vectors are assigned exponential probabilistic levels $\ell = \lfloor -\ln(\text{uniform}(0,1)) \cdot m_L \rfloor$.
* **Top Layers (Sparse):** Rapid 1-nearest neighbor greedy routing across long-range graph connections.
* **Bottom Layer 0 (Dense):** Full $\text{ef}$-search expansion across fine-grained local neighborhoods.
* **Bidirectional Edges & Pruning:** Maintains up to $M$ neighbors per node ($M_{\max0} = 2M$ on Layer 0).

---

## 📁 Project Directory Layout

```
vectordb/
├── app/
│   ├── algorithms/
│   │   ├── heap.py                 # Min/Max heap priority queue helpers
│   │   └── kmeans.py               # From-scratch K-Means clustering algorithm
│   ├── api/
│   │   ├── routes_chat.py          # POST /chat/cached (Semantic Cache endpoint)
│   │   ├── routes_search.py        # POST /search, POST /search/text
│   │   ├── routes_system.py        # GET /health, GET /stats, POST /rebuild/{index}
│   │   └── routes_vectors.py       # POST /vectors, POST /vectors/bulk, DELETE /vectors/{id}
│   ├── core/
│   │   ├── distance.py             # L2 unit normalization & cosine metric
│   │   ├── exceptions.py           # Domain exception classes
│   │   ├── topk.py                 # Fast argpartition Top-K selector
│   │   ├── types.py                # SearchResult & IndexState dataclasses
│   │   └── vector_store.py         # Contiguous NumPy vector store + metadata
│   ├── evaluation/
│   │   ├── benchmark.py            # Latency (p50/p95/p99) & speedup runner
│   │   ├── ground_truth.py         # Precomputed ground truth generator
│   │   └── recall.py               # Exact Recall@K computation
│   ├── indexes/
│   │   ├── base.py                 # Base index abstract class
│   │   ├── brute_force.py          # Exact Brute-Force index
│   │   ├── hnsw.py                 # Multi-layer HNSW index
│   │   ├── ivf.py                  # IVF-Flat index
│   │   └── manager.py              # Central IndexManager coordinator
│   ├── services/
│   │   └── llm_service.py          # Google Gemini LLM API client
│   ├── config.py                   # Centralized configuration & constants
│   └── main.py                     # FastAPI application factory
├── data/                           # 50,000 synthetic benchmark dataset
├── data_text/                      # 5,000 text statements & persistent knowledge
├── scripts/
│   ├── benchmark.py                # 50K benchmark & Recall@K verification
│   ├── build_indexes.py            # Index pre-building script
│   ├── demo_cached_rag.py          # Interactive Semantic Cache RAG demo
│   ├── demo_rag.py                 # Interactive Vector DB + Gemini RAG demo
│   ├── demo_text_search.py         # Side-by-side Brute vs IVF vs HNSW search
│   ├── generate_dataset.py         # 50K synthetic clustered dataset generator
│   ├── generate_ground_truth.py    # Ground truth generation script
│   ├── generate_text_dataset.py    # 5K text corpus & embedding generator
│   ├── inspect_dataset.py          # Interactive dataset inspector
│   └── test_suite.py               # Automated verification test suite
├── tests/                          # Pytest test suite (unit, api, integration)
├── requirements.txt                # Minimal dependencies (numpy, fastapi, uvicorn)
├── .env.example                    # Environment variable template
└── README.md
```

---

## 🛠 Getting Started & Installation

### 1. Clone the Repository & Create Virtual Environment
```bash
git clone https://github.com/salaj07/vectorDB.git
cd vectorDB

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional for Gemini RAG)
Copy `.env.example` to `.env` and insert your Gemini API Key if you want to use the live LLM RAG features:
```bash
copy .env.example .env     # Windows
# or: cp .env.example .env  # Linux/macOS
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🎯 Interactive Demonstrations

### 1. Performance & Recall Benchmarking (50,000 Vectors)
Runs the full evaluation suite across 500 test queries against the 50,000 vector dataset:
```powershell
python scripts/benchmark.py
```
*Outputs Mean Recall@10, latency percentiles ($p50, p95, p99$), speedup multiplier vs. Brute-Force, and RAM usage.*

---

### 2. Multi-Index Text Similarity Search
Compares **Brute Force**, **IVF-Flat**, and **HNSW** side-by-side on 5,000 text statements:
```powershell
python scripts/demo_text_search.py "express js framework"
```
Or query other topics:
```powershell
python scripts/demo_text_search.py "docker containers and kubernetes"
python scripts/demo_text_search.py "deep neural networks and transformers"
```

---

### 3. Interactive RAG (Vector DB + Gemini LLM)
Retrieves grounding facts from VectorForge and synthesizes a verified response using Gemini:
```powershell
python scripts/demo_rag.py "what is next js?"
```

---

### 4. Self-Learning Semantic Cache (< 1ms Latency)
Demonstrates intelligent query caching with dual-layer persistent storage.

```powershell
# Run 1: Cache Miss -> Retrieves Context -> Calls Gemini -> Saves to DB & Knowledge file
python scripts/demo_cached_rag.py "explain docker containers"

# Run 2: CACHE HIT -> Retrieves from Vector DB in 0.5ms -> Skips Gemini API call completely!
python scripts/demo_cached_rag.py "explain docker containers"
```

---

### 5. Dataset Inspector (50K & 5K Datasets)
Inspect raw dimensions, metadata distributions, and sample coordinates:
```powershell
# Inspect 50,000 synthetic dataset:
python scripts/inspect_dataset.py

# Inspect 5,000 text dataset:
python scripts/inspect_dataset.py data_text
```

---

## 🌐 REST API Reference & Swagger UI

Start the production FastAPI server:
```powershell
uvicorn app.main:app --reload --port 8000
```
Open **`http://localhost:8000/docs`** in your browser to access the interactive Swagger UI.

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/vectors` | Ingest a single vector with custom metadata and string ID. |
| `POST` | `/vectors/bulk` | Batch insert multiple vectors and metadata in one request. |
| `GET` | `/vectors/{id}` | Retrieve a stored vector array and metadata by ID. |
| `DELETE` | `/vectors/{id}` | Logically delete a vector via tombstone mask. |
| `POST` | `/search` | Search top-$K$ nearest neighbors by raw float array (`brute`, `ivf`, or `hnsw`). |
| `POST` | `/search/text` | Search top-$K$ nearest neighbors by natural language text query. |
| `POST` | `/chat/cached` | Production semantic cache endpoint with Gemini LLM integration. |
| `GET` | `/stats` | View total vector counts, dimensions, and index build states. |
| `POST` | `/rebuild/{index}` | Manually trigger a background rebuild of `ivf` or `hnsw`. |
| `GET` | `/health` | System health check and memory status. |

---

## 📊 Benchmark Results & Recall Metrics

Evaluated on **50,000 synthetic vectors** ($D = 128$) across **500 independent queries** ($K = 10$):

| Index | Configuration | Recall@10 | Latency ($p50$) | Latency ($p95$) | Speedup vs Brute |
|---|---|:---:|:---:|:---:|:---:|
| **Brute Force** | Exhaustive Scan | **100.0%** | 3.42 ms | 3.89 ms | 1.0&times; *(Baseline)* |
| **IVF-Flat** | $n_{\text{probe}} = 1$ | 72.4% | 0.28 ms | 0.35 ms | **12.2&times;** |
| **IVF-Flat** | $n_{\text{probe}} = 5$ | **93.8%** | 0.74 ms | 0.89 ms | **4.6&times;** |
| **IVF-Flat** | $n_{\text{probe}} = 10$ | **98.2%** | 1.25 ms | 1.48 ms | **2.7&times;** |
| **HNSW** | $\text{ef}_{\text{search}} = 10$ | 86.1% | 0.38 ms | 0.49 ms | **9.0&times;** |
| **HNSW** | $\text{ef}_{\text{search}} = 50$ | **97.4%** | 0.92 ms | 1.15 ms | **3.7&times;** |
| **HNSW** | $\text{ef}_{\text{search}} = 100$ | **99.6%** | 1.62 ms | 1.94 ms | **2.1&times;** |

---

## 📌 Design Decisions: Graph Deletion

### Why Logical Deletion (Tombstoning) is Used in HNSW

In VectorForge, deletion is handled via a Boolean `active_mask` filter (logical tombstoning):

1. **Avoids Catastrophic Graph Disconnection:** Physical node deletion in a small-world graph requires complex re-wiring to prevent graph partitioning into isolated sub-graphs.
2. **Zero Ingestion Latency Penalty:** Toggling an active bit in NumPy takes $\mathcal{O}(1)$ time.
3. **Search Invariant Protection:** During search, traversed candidate nodes filter out tombstones before returning Top-$K$ results.
4. **Rebuild on Dirty:** When the deletion ratio exceeds a configured threshold, the index marks itself as `DIRTY` and triggers an optimal background graph rebuild via `POST /rebuild/hnsw`.

---

## 🧪 Running Tests

Execute the comprehensive automated test suite:
```powershell
# Run all unit, integration, and API tests
pytest

# Or run the master test runner script
python scripts/test_suite.py
```

---

## 📜 License
MIT License. Free for educational, research, and commercial use.
