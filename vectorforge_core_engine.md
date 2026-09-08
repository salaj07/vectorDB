# VectorForge — From-Scratch Vector Database

> **Project goal:** Build a small but technically credible vector database/search engine from scratch using Python and NumPy, without FAISS, Pinecone, Chroma, sklearn.neighbors, or another ready-made vector index.
>
> **Scope decision:** Implement **Brute Force + IVF-Flat + HNSW**. Brute Force is the exact ground truth engine; IVF-Flat and HNSW are approximate indexes.

---

# 1. Problem Statement

Modern applications often convert text, images, documents, or other data into high-dimensional vectors called **embeddings**. A vector database stores these vectors and answers queries such as:

> “Which stored vectors are most similar to this query vector?”

A naive solution compares a query against every vector. This is accurate but becomes expensive as the dataset grows.

VectorForge demonstrates how a vector database can be built from first principles:

1. Store vectors efficiently.
2. Calculate cosine similarity using NumPy.
3. Perform exact top-k search with Brute Force.
4. Build an **IVF-Flat** approximate index manually.
5. Build an **HNSW** approximate index manually.
6. Expose all operations through a small FastAPI API.
7. Compare approximate results against exact ground truth.
8. Measure **latency, speedup, recall@10, and memory usage**.
9. Keep the architecture ready for a future 3D visualization frontend.

---

# 2. Assignment Requirements

The implementation must satisfy the following constraints:

- Do **not** use Pinecone.
- Do **not** use FAISS.
- Do **not** use Chroma.
- Do **not** use `sklearn.neighbors`.
- Implement exact cosine similarity search yourself using NumPy.
- Support at least **50,000 vectors**.
- Implement at least one approximate index manually.
- VectorForge intentionally implements **both IVF-Flat and HNSW**.
- Support insert, search, and delete operations.
- Use a real short-text corpus of approximately 5,000 texts or deterministic clustered synthetic vectors.
- Create a query set of approximately 500 vectors.
- Compute exact top-10 neighbors using Brute Force.
- Evaluate approximate search using **Recall@10**.
- Provide a working API and demo.
- Keep the repository public and reproducible.

---

# 3. Scope

## 3.1 In Scope

### Core Engine
- Vector storage
- Vector normalization
- Cosine similarity
- Exact Brute Force search
- Top-k optimization
- K-Means clustering
- IVF-Flat
- HNSW
- Logical deletion
- Index rebuilding

### Evaluation
- Ground-truth generation
- Recall@K
- Search latency
- Build time
- Speedup
- Memory estimates

### API
- Health
- Stats
- Insert
- Delete
- Search
- Index selection

### Testing
- Unit tests
- Integration tests
- API tests
- Cross-index correctness tests
- Recall tests
- Test-suite runner

### Future UI
- 3D vector visualization
- Cluster visualization
- HNSW graph visualization
- Search-path visualization

---

# 4. Out of Scope for the First Version

Do not over-engineer the first implementation.

- PostgreSQL
- MongoDB
- Redis
- Distributed storage
- Replication
- Authentication
- Multi-node clustering
- GPU acceleration
- Production-grade persistence
- Full-text search
- Advanced filtering
- Streaming ingestion
- Cloud deployment
- Fancy frontend before the engine works

The vector database itself is the main project.

---

# 5. Engineering Principles

The code should be:

- DRY
- Modular
- Test-driven
- Vectorized with NumPy
- Easy to benchmark
- Easy to extend
- Free from unnecessary abstraction
- Free from duplicated search logic
- Explicit about index-specific behavior

## Important optimization principle

Do not repeatedly implement:

- cosine similarity
- normalization
- top-k selection
- ID mapping
- deletion checks

Create reusable utilities and use them across Brute Force, IVF, and HNSW.

---

# 6. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Numerical computation | NumPy |
| API | FastAPI |
| Server | Uvicorn |
| Validation | Pydantic |
| Testing | Pytest |
| HTTP testing | FastAPI TestClient |
| Data format | `.npy`, `.npz`, JSON |
| Visualization later | React + Three.js / React Three Fiber |
| Version control | Git + GitHub |

Recommended `requirements.txt`:

```txt
numpy
fastapi
uvicorn
pydantic
pytest
httpx
```

No vector database library should be added.

---

# 7. High-Level Architecture

```text
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Index Manager    │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
      ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
      │ Brute Force │   │  IVF-Flat   │   │    HNSW     │
      │    Index    │   │    Index    │   │    Index    │
      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Vector Store     │
                    │  NumPy + Metadata   │
                    └─────────────────────┘
```

---

# 8. Final Project Structure

```text
vectorforge/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── routes_search.py
│   │   ├── routes_vectors.py
│   │   └── routes_system.py
│   │
│   ├── core/
│   │   ├── vector_store.py
│   │   ├── distance.py
│   │   ├── topk.py
│   │   ├── types.py
│   │   └── exceptions.py
│   │
│   ├── indexes/
│   │   ├── base.py
│   │   ├── brute_force.py
│   │   ├── ivf.py
│   │   └── hnsw.py
│   │
│   ├── algorithms/
│   │   ├── kmeans.py
│   │   └── heap.py
│   │
│   ├── evaluation/
│   │   ├── ground_truth.py
│   │   ├── recall.py
│   │   └── benchmark.py
│   │
│   ├── schemas/
│   │   ├── vector.py
│   │   ├── search.py
│   │   └── response.py
│   │
│   └── config.py
│
├── tests/
│   ├── unit/
│   │   ├── test_distance.py
│   │   ├── test_topk.py
│   │   ├── test_vector_store.py
│   │   ├── test_kmeans.py
│   │   ├── test_brute_force.py
│   │   ├── test_ivf.py
│   │   └── test_hnsw.py
│   │
│   ├── integration/
│   │   ├── test_indexes.py
│   │   └── test_search_flow.py
│   │
│   ├── api/
│   │   ├── test_health.py
│   │   ├── test_vectors.py
│   │   └── test_search.py
│   │
│   └── conftest.py
│
├── scripts/
│   ├── generate_dataset.py
│   ├── build_indexes.py
│   ├── generate_ground_truth.py
│   ├── benchmark.py
│   └── test_suite.py
│
├── data/
│   ├── vectors.npy
│   ├── ids.npy
│   └── metadata.json
│
├── benchmarks/
│   └── results.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 9. Core Data Model

A vector is represented as:

```text
v = [x1, x2, x3, ..., xd]
```

For example:

```python
[0.12, -0.43, 0.91, 0.22]
```

A vector has:

- ID
- embedding
- optional metadata
- active/deleted state

Example:

```json
{
  "id": "doc_102",
  "metadata": {
    "text": "pressure increased during drilling",
    "source": "well_12"
  }
}
```

---

# 10. Vector Storage

The initial storage layer should be simple.

Use:

```text
vectors.npy
ids.npy
metadata.json
```

For the in-memory engine:

```python
vectors: np.ndarray
ids: np.ndarray
metadata: dict
active_mask: np.ndarray
```

Recommended vector matrix shape:

```text
(N, D)
```

Where:

- `N` = number of vectors
- `D` = embedding dimension

For example:

```text
50,000 × 128
```

Use `float32` to reduce memory usage.

---

# 11. Embeddings

An embedding is a numerical representation of information.

Example:

```text
"Oil pressure increased"
          ↓
[0.21, -0.11, 0.84, ..., 0.07]
```

The exact semantic meaning of individual dimensions is not important.

What matters is that semantically similar inputs should produce vectors that are close in vector space.

For the first implementation, synthetic clustered vectors are recommended because they are:

- deterministic
- reproducible
- fast
- easy to control
- ideal for testing ANN recall

Later, real embeddings can be plugged into the same storage/index API.

---

# 12. Cosine Similarity

Cosine similarity:

```text
cos(q, x) = (q · x) / (||q|| ||x||)
```

Where:

- `q` = query vector
- `x` = stored vector
- `q · x` = dot product
- `||q||` = magnitude of q
- `||x||` = magnitude of x

If vectors are normalized:

```text
||q|| = ||x|| = 1
```

then:

```text
cos(q, x) = q · x
```

Therefore the search can become a matrix-vector multiplication.

---

# 13. Distance Module

File:

```text
app/core/distance.py
```

Responsibilities:

- normalize vectors
- normalize batches
- calculate cosine similarity
- calculate similarity matrix where needed

Example design:

```python
def normalize(vectors: np.ndarray) -> np.ndarray:
    ...

def cosine_similarity(
    query: np.ndarray,
    vectors: np.ndarray
) -> np.ndarray:
    ...
```

Do not duplicate this implementation inside each index.

---

# 14. Top-K Search

For N vectors:

```python
scores = vectors @ query
```

Sorting all N results:

```python
np.argsort(scores)[-k:]
```

is correct but can be unnecessarily expensive.

Prefer:

```python
np.argpartition(...)
```

to find the top-k candidates.

Then sort only those k candidates.

Conceptually:

```text
50,000 scores
      ↓
argpartition
      ↓
top 10 candidates
      ↓
sort only 10
```

This utility should live in:

```text
app/core/topk.py
```

---

# 15. Brute Force Index

File:

```text
app/indexes/brute_force.py
```

Purpose:

- exact search
- ground truth
- correctness reference
- benchmark baseline

Algorithm:

```text
query
  ↓
normalize query
  ↓
compare with every active vector
  ↓
calculate cosine similarity
  ↓
top-k
  ↓
return IDs + scores
```

Complexity:

```text
O(ND)
```

For:

```text
N = 50,000
D = 128
```

the operation is still practical using NumPy, but ANN indexes should reduce the number of candidate comparisons.

---

# 16. Common Index Interface

File:

```text
app/indexes/base.py
```

All indexes should expose a consistent interface.

Example:

```python
class BaseIndex:
    def build(self, vectors, ids):
        ...

    def add(self, vector_id, vector):
        ...

    def search(self, query, k=10):
        ...

    def delete(self, vector_id):
        ...

    def stats(self):
        ...
```

The exact implementation differs by index.

This prevents the API layer from containing index-specific logic.

---

# 17. K-Means

File:

```text
app/algorithms/kmeans.py
```

IVF requires clustering.

Implement K-Means manually.

Algorithm:

```text
1. Initialize K centroids
2. Assign each vector to nearest centroid
3. Recalculate centroid means
4. Repeat
5. Stop after max iterations or convergence
```

Distance for normalized vectors can be based on:

```text
1 - cosine_similarity
```

For the first version, Euclidean distance can also be used for centroid assignment if the dataset strategy is designed consistently.

Recommended initial configuration:

```text
nlist = 100
max_iterations = 20
```

The values should be configurable.

---

# 18. IVF-Flat

File:

```text
app/indexes/ivf.py
```

IVF means:

> Inverted File Index

The vector space is divided into clusters.

Example:

```text
50,000 vectors
      ↓
K-Means
      ↓
100 clusters
      ↓
cluster 0 → vectors
cluster 1 → vectors
...
cluster 99 → vectors
```

The index stores:

```text
centroids
inverted_lists
```

Example:

```python
inverted_lists = {
    0: [id1, id8, id21],
    1: [id2, id5],
    ...
}
```

---

# 19. IVF Search

Query:

```text
query
  ↓
compare query against all centroids
  ↓
select nearest nprobe clusters
  ↓
collect vectors from those clusters
  ↓
exact cosine similarity on candidates
  ↓
top-k
```

Important parameter:

```text
nprobe
```

Meaning:

> Number of clusters searched per query.

Example:

```text
nlist = 100
nprobe = 5
```

Instead of searching all 100 clusters, search only 5.

Trade-off:

```text
higher nprobe
    ↓
higher recall
    ↓
higher latency
```

and:

```text
lower nprobe
    ↓
lower latency
    ↓
lower recall
```

---

# 20. Why IVF is Called IVF-Flat

The cluster selection is approximate.

But once the relevant clusters are selected, the vectors inside them are compared using exact cosine similarity.

Therefore:

```text
Approximate candidate selection
+
Exact vector comparison
=
IVF-Flat
```

---

# 21. IVF Insert

When inserting a new vector:

```text
new vector
   ↓
find nearest centroid
   ↓
append vector ID to that inverted list
```

If the system allows vectors to be inserted before an index is built, either:

1. mark the index as stale and require rebuild, or
2. support dynamic insertion carefully.

For the first version, prefer:

```text
insert into store
      ↓
mark indexes dirty
      ↓
rebuild index
```

This is simpler and safer.

---

# 22. IVF Delete

Do not immediately shift large arrays.

Use logical deletion:

```text
active[id] = False
```

Search ignores inactive vectors.

This reduces complexity.

Later:

```text
rebuild index
```

can physically remove deleted vectors.

---

# 23. HNSW

File:

```text
app/indexes/hnsw.py
```

HNSW means:

> Hierarchical Navigable Small World graph

It represents vectors as graph nodes.

Each node:

```text
vector + neighbors
```

Example:

```text
A ─── B ─── C
│     │     │
D ─── E ─── F
```

Similar vectors tend to be connected.

---

# 24. HNSW Hierarchy

HNSW contains multiple layers.

Conceptually:

```text
Layer 2:

        A -------- F


Layer 1:

    A --- C --- F
     \   / \   /
      B ----- E


Layer 0:

A-B-C-D-E-F-G-H-I-J...
```

Higher layers:

- fewer nodes
- long-range connections
- fast navigation

Layer 0:

- many nodes
- detailed neighborhood

---

# 25. HNSW Parameters

Use configurable parameters:

```text
M
ef_construction
ef_search
```

### M

Maximum approximate number of neighbors per node per layer.

Typical initial value:

```text
M = 8 or 16
```

### ef_construction

Controls graph construction quality.

Initial:

```text
ef_construction = 100
```

### ef_search

Controls search quality.

Initial:

```text
ef_search = 50
```

Higher `ef_search` generally means:

```text
higher recall
higher latency
```

---

# 26. HNSW Search

Basic search flow:

```text
query
  ↓
start from entry point
  ↓
search from highest layer
  ↓
move toward closer nodes
  ↓
drop to next layer
  ↓
continue
  ↓
layer 0 candidate search
  ↓
top-k
```

The implementation should use a priority queue / heap.

File:

```text
app/algorithms/heap.py
```

can contain reusable heap utilities if required.

---

# 27. HNSW Build Strategy

For every vector:

```text
1. Choose random level.
2. Start from current entry point.
3. Search for good neighbors.
4. Connect the node.
5. Prune excess neighbors.
6. Update graph.
7. Update entry point if necessary.
```

Keep the first implementation simple.

Do not attempt every advanced HNSW optimization immediately.

---

# 28. HNSW Delete

Use logical deletion initially:

```text
deleted[node_id] = True
```

Search should ignore deleted nodes when producing results.

Because graph topology remains unchanged, deletion is cheap.

A later rebuild can remove deleted nodes physically.

---

# 29. Index Manager

Create a central manager responsible for:

```text
VectorStore
BruteForceIndex
IVFIndex
HNSWIndex
```

Example:

```text
IndexManager
    ├── store
    ├── brute
    ├── ivf
    └── hnsw
```

Responsibilities:

- load/build indexes
- select index
- coordinate insertion
- coordinate deletion
- expose stats
- mark stale indexes

The API should not directly manipulate index internals.

---

# 30. API Design

## GET `/health`

Response:

```json
{
  "status": "ok"
}
```

---

## GET `/stats`

Example:

```json
{
  "vectors": 50000,
  "dimension": 128,
  "indexes": {
    "brute": "ready",
    "ivf": "ready",
    "hnsw": "ready"
  }
}
```

---

## POST `/vectors`

Request:

```json
{
  "id": "vec_123",
  "vector": [0.12, 0.44, 0.72],
  "metadata": {
    "text": "sample statement"
  }
}
```

Response:

```json
{
  "id": "vec_123",
  "status": "inserted"
}
```

---

# 31. DELETE `/vectors/{id}`

Example:

```text
DELETE /vectors/vec_123
```

Response:

```json
{
  "id": "vec_123",
  "status": "deleted"
}
```

---

# 32. POST `/search`

Request:

```json
{
  "query": [0.11, 0.43, 0.71],
  "k": 10,
  "index": "ivf",
  "nprobe": 5
}
```

For HNSW:

```json
{
  "query": [0.11, 0.43, 0.71],
  "k": 10,
  "index": "hnsw",
  "ef_search": 50
}
```

For brute force:

```json
{
  "query": [0.11, 0.43, 0.71],
  "k": 10,
  "index": "brute"
}
```

Response:

```json
{
  "index": "ivf",
  "results": [
    {
      "id": "vec_21",
      "score": 0.9821
    },
    {
      "id": "vec_91",
      "score": 0.9712
    }
  ]
}
```

---

# 33. API Validation

Validate:

- vector is not empty
- vector dimension is correct
- `k > 0`
- `k <= number of active vectors`
- index name is valid
- `nprobe > 0`
- `ef_search > 0`
- vector values are numeric

Use Pydantic schemas.

---

# 34. Dataset Strategy

Two acceptable approaches:

## Option A — Real Text Corpus

Approximately:

```text
5,000 short texts
```

Convert each into embeddings using an external embedding model during dataset preparation.

The vector engine itself still performs the storage and search from scratch.

## Option B — Deterministic Synthetic Dataset

Recommended for the first build.

Generate:

```text
50,000 vectors
128 dimensions
```

around several known cluster centers.

Example:

```text
Cluster A → drilling
Cluster B → pressure
Cluster C → mud
Cluster D → casing
...
```

This makes ANN behavior easy to demonstrate.

Use a fixed random seed:

```python
np.random.default_rng(42)
```

so all experiments are reproducible.

---

# 35. Dataset Generator

File:

```text
scripts/generate_dataset.py
```

Responsibilities:

- generate vectors
- generate IDs
- generate optional metadata
- save `.npy`
- save metadata JSON

Example:

```bash
python scripts/generate_dataset.py
```

Expected output:

```text
Generated 50,000 vectors
Dimension: 128
Saved to data/
```

---

# 36. Ground Truth

File:

```text
app/evaluation/ground_truth.py
```

For each query:

```text
query
  ↓
Brute Force
  ↓
exact top-10
```

Store:

```text
query_id → [neighbor IDs]
```

For approximately 500 queries.

The brute-force result is the correctness baseline.

---

# 37. Recall@10

Recall@10 measures how many of the true top-10 results were found by the approximate index.

Formula:

```text
Recall@10 =
|Approximate Top-10 ∩ Exact Top-10|
------------------------------------
10
```

Example:

Exact:

```text
[A, B, C, D, E, F, G, H, I, J]
```

IVF:

```text
[A, B, C, X, E, Y, G, H, Z, J]
```

Intersection:

```text
A, B, C, E, G, H, J = 7
```

Therefore:

```text
Recall@10 = 0.7
```

---

# 38. Benchmarking

File:

```text
app/evaluation/benchmark.py
```

Measure:

### Build time

```text
IVF build time
HNSW build time
```

### Query latency

Measure:

```text
Brute Force
IVF
HNSW
```

### Recall

Measure:

```text
IVF Recall@10
HNSW Recall@10
```

### Speedup

```text
speedup = brute_force_latency / approximate_latency
```

---

# 39. Benchmark Matrix

Test several configurations.

### IVF

```text
nprobe = 1
nprobe = 5
nprobe = 10
nprobe = 20
```

### HNSW

```text
ef_search = 20
ef_search = 50
ef_search = 100
```

Example result:

| Index | Parameter | Avg Latency | Recall@10 | Speedup |
|---|---:|---:|---:|---:|
| Brute | — | 10 ms | 1.00 | 1.0x |
| IVF | nprobe=5 | 2 ms | 0.82 | 5.0x |
| IVF | nprobe=10 | 3 ms | 0.91 | 3.3x |
| HNSW | ef=50 | 1.5 ms | 0.94 | 6.7x |

These are illustrative values only. Actual values must come from the benchmark.

---

# 40. Benchmark Script

File:

```text
scripts/benchmark.py
```

Usage:

```bash
python scripts/benchmark.py
```

It should:

1. Load dataset.
2. Load/build indexes.
3. Load query set.
4. Load ground truth.
5. Run warm-up queries.
6. Run benchmark queries.
7. Calculate latency statistics.
8. Calculate recall.
9. Save results to `benchmarks/results.json`.

Report at least:

```text
average latency
p50 latency
p95 latency
recall@10
build time
speedup
```

---

# 41. Memory Considerations

For:

```text
50,000 × 128 × float32
```

raw vectors require approximately:

```text
50,000 × 128 × 4 bytes
≈ 25.6 MB
```

Additional memory is required for:

- IDs
- metadata
- IVF centroids
- inverted lists
- HNSW graph
- temporary arrays

Avoid unnecessary copies of the full vector matrix.

Prefer views/index arrays where practical.

---

# 42. Important Performance Rules

## Rule 1 — Normalize once

Do not normalize every stored vector on every query.

Normalize during ingestion/build.

Normalize the query once.

## Rule 2 — Use float32

Prefer:

```python
np.float32
```

unless higher precision is specifically required.

## Rule 3 — Vectorize

Prefer:

```python
vectors @ query
```

over:

```python
for vector in vectors:
    ...
```

## Rule 4 — Use partial top-k

Use:

```python
np.argpartition
```

instead of sorting all vectors.

## Rule 5 — Build once

Do not rebuild IVF/HNSW for every search request.

## Rule 6 — Separate build and query

Index construction can be expensive.

Query execution must be optimized independently.

## Rule 7 — Avoid duplicated logic

All indexes should use common:

- distance utilities
- top-k utilities
- vector-store operations
- result models

---

# 43. Testing Strategy — TDD

The project should be built using:

```text
RED
 ↓
GREEN
 ↓
REFACTOR
 ↓
BENCHMARK
```

For every major component:

1. Write the test.
2. Run it and make it fail.
3. Implement the smallest correct solution.
4. Make the test pass.
5. Refactor for DRY/performance.
6. Run the full test suite.
7. Benchmark only after correctness is stable.

---

# 44. Test Architecture

```text
tests/
├── unit/
├── integration/
├── api/
└── conftest.py
```

### Unit tests

Test one component at a time.

### Integration tests

Test interactions between:

```text
VectorStore
+
Index
+
Search
```

### API tests

Test FastAPI endpoints using `TestClient`.

---

# 45. Distance Tests

File:

```text
tests/unit/test_distance.py
```

Test:

- identical vectors have similarity ≈ 1
- orthogonal vectors have similarity ≈ 0
- opposite vectors have similarity ≈ -1
- normalization works
- zero vectors are handled correctly
- batch similarity has correct shape
- float32 inputs work

Example:

```python
def test_identical_vectors_have_similarity_one():
    ...
```

---

# 46. Top-K Tests

File:

```text
tests/unit/test_topk.py
```

Test:

- correct top-k IDs
- correct ordering
- `k=1`
- `k=N`
- duplicate scores
- invalid k
- empty input

The optimized implementation must match a simple reference implementation.

---

# 47. VectorStore Tests

File:

```text
tests/unit/test_vector_store.py
```

Test:

- insert
- retrieve
- duplicate ID
- dimension mismatch
- delete
- logical deletion
- active vector count
- metadata
- empty store

---

# 48. K-Means Tests

File:

```text
tests/unit/test_kmeans.py
```

Test:

- correct centroid shape
- deterministic output with fixed seed
- assignments have correct length
- convergence
- no invalid cluster IDs
- simple known clusters produce sensible assignments

---

# 49. Brute Force Tests

File:

```text
tests/unit/test_brute_force.py
```

Test:

- exact nearest neighbor
- top-k correctness
- deleted vectors are ignored
- query dimension validation
- result ordering
- empty index behavior

Brute Force should be treated as the reference implementation.

---

# 50. IVF Tests

File:

```text
tests/unit/test_ivf.py
```

Test:

- K-Means integration
- centroid count
- inverted list creation
- every active vector appears in exactly one list
- nearest cluster selection
- `nprobe=1`
- `nprobe=nlist`
- top-k result validity
- deleted vectors are ignored
- invalid nprobe handling

Important correctness test:

```text
IVF with nprobe=nlist
```

should behave like exact search because every cluster is searched.

---

# 51. HNSW Tests

File:

```text
tests/unit/test_hnsw.py
```

Test:

- node insertion
- level assignment
- neighbor creation
- maximum neighbor constraints
- graph connectivity on a small dataset
- search returns valid IDs
- deleted nodes are ignored
- query works with one node
- query works with a small graph

Do not require exact equality with Brute Force for every HNSW query because HNSW is approximate.

Instead test:

```text
result IDs are valid
result count is correct
recall is above a chosen threshold
```

---

# 52. Cross-Index Tests

File:

```text
tests/integration/test_indexes.py
```

Use the same small dataset for:

```text
Brute Force
IVF
HNSW
```

For each query:

```text
exact = brute.search(query)
ivf = ivf.search(query)
hnsw = hnsw.search(query)
```

Verify:

- all result IDs exist
- result count = k
- scores are ordered
- deleted IDs never appear

Then calculate recall.

---

# 53. Search Flow Tests

File:

```text
tests/integration/test_search_flow.py
```

Test:

```text
insert
  ↓
build
  ↓
search
  ↓
delete
  ↓
search again
```

Example:

```text
Insert A
Insert B
Insert C
Search
Delete B
Search
```

Expected:

```text
B never appears after deletion
```

---

# 54. API Tests

Files:

```text
tests/api/test_health.py
tests/api/test_vectors.py
tests/api/test_search.py
```

Test:

### Health

```text
GET /health → 200
```

### Insert

```text
POST /vectors → 200/201
```

### Delete

```text
DELETE /vectors/{id}
```

### Search

```text
POST /search
```

### Validation

Invalid requests should return proper 4xx responses.

---

# 55. Test Fixtures

File:

```text
tests/conftest.py
```

Provide reusable fixtures:

```text
small_vectors
small_ids
small_queries
vector_store
brute_index
ivf_index
hnsw_index
api_client
```

Use small datasets for normal tests.

Example:

```text
N = 100
D = 16
queries = 10
```

Do not run 50,000-vector benchmarks for every unit test.

---

# 56. Large Dataset Tests

A separate benchmark/validation path should test:

```text
N = 50,000
D = 128
Q = 500
```

Do not make every Pytest run expensive.

Recommended separation:

```text
pytest tests/
```

for correctness.

and:

```text
python scripts/benchmark.py
```

for performance.

---

# 57. Project Test Suite Script

File:

```text
scripts/test_suite.py
```

Purpose:

Provide one command for developers/judges to validate the project.

Usage:

```bash
python scripts/test_suite.py
```

It should execute:

```text
1. Unit tests
2. Integration tests
3. API tests
4. Final summary
```

Suggested output:

```text
========================================
        VECTORFORGE TEST SUITE
========================================

[1/3] Unit Tests
✓ Distance
✓ Top-K
✓ Vector Store
✓ K-Means
✓ Brute Force
✓ IVF
✓ HNSW

[2/3] Integration Tests
✓ Index consistency
✓ Search flow
✓ Delete flow

[3/3] API Tests
✓ Health
✓ Insert
✓ Search
✓ Delete

========================================
RESULT: ALL TESTS PASSED
========================================
```

The script should return a non-zero exit code when tests fail.

---

# 58. Recommended Test Commands

Fast unit tests:

```bash
pytest tests/unit -v
```

Integration:

```bash
pytest tests/integration -v
```

API:

```bash
pytest tests/api -v
```

Everything:

```bash
pytest tests -v
```

Project test runner:

```bash
python scripts/test_suite.py
```

Benchmark:

```bash
python scripts/benchmark.py
```

---

# 59. Postman Testing

Once the API is ready, create a Postman collection.

Suggested folders:

```text
VectorForge
│
├── System
│   ├── Health
│   └── Stats
│
├── Vectors
│   ├── Insert
│   └── Delete
│
└── Search
    ├── Brute Force
    ├── IVF
    └── HNSW
```

This is for manual/demo testing.

Pytest remains the automated regression suite.

---

# 60. Postman Demo Flow

Recommended live demo:

```text
1. GET /health
2. GET /stats
3. Insert vector
4. Search with brute force
5. Search with IVF
6. Search with HNSW
7. Delete vector
8. Search again
```

Then show:

```text
benchmark results
recall
latency
speedup
```

---

# 61. Development Order

Follow this order strictly.

## Phase 1 — Project Setup

Create:

```text
app/
tests/
scripts/
data/
benchmarks/
```

Set up:

```text
requirements.txt
pytest
FastAPI
```

Run:

```bash
pytest
```

---

## Phase 2 — Distance

Create:

```text
app/core/distance.py
tests/unit/test_distance.py
```

Implement:

```text
normalize
cosine_similarity
```

First.

---

## Phase 3 — Top-K

Create:

```text
app/core/topk.py
tests/unit/test_topk.py
```

Implement optimized top-k.

---

## Phase 4 — Vector Store

Create:

```text
app/core/vector_store.py
tests/unit/test_vector_store.py
```

Implement:

```text
insert
get
delete
active mask
metadata
```

---

## Phase 5 — Brute Force

Create:

```text
app/indexes/base.py
app/indexes/brute_force.py
tests/unit/test_brute_force.py
```

Get exact search fully correct.

This becomes the reference implementation.

---

## Phase 6 — K-Means

Create:

```text
app/algorithms/kmeans.py
tests/unit/test_kmeans.py
```

Verify clustering on synthetic data.

---

## Phase 7 — IVF-Flat

Create:

```text
app/indexes/ivf.py
tests/unit/test_ivf.py
```

Implement:

```text
build
cluster assignment
nprobe search
top-k
logical deletion
```

Then compare with Brute Force.

---

## Phase 8 — HNSW

Create:

```text
app/indexes/hnsw.py
tests/unit/test_hnsw.py
```

Implement:

```text
node
levels
neighbors
search
ef_search
logical deletion
```

Keep the implementation small and readable.

---

## Phase 9 — Integration

Create:

```text
tests/integration/
```

Verify:

```text
store → index → search → delete
```

---

## Phase 10 — Evaluation

Create:

```text
app/evaluation/
scripts/generate_ground_truth.py
scripts/benchmark.py
```

Generate:

```text
50k vectors
500 queries
exact top-10
IVF recall
HNSW recall
latency
speedup
```

---

## Phase 11 — API

Create:

```text
app/api/
app/schemas/
app/main.py
```

Expose:

```text
/health
/stats
/vectors
/search
```

---

## Phase 12 — Test Suite

Create:

```text
scripts/test_suite.py
```

Ensure one command can validate the complete project.

---

## Phase 13 — Documentation

Create:

```text
README.md
```

Include:

- architecture
- algorithms
- setup
- API examples
- benchmark results
- limitations
- screenshots later
- demo video

---

# 62. Suggested Git Commit Strategy

Keep commits meaningful.

```text
chore: initialize vectorforge
test: add cosine similarity tests
feat: implement cosine similarity
test: add top-k tests
feat: implement optimized top-k
test: add vector store tests
feat: implement vector store
test: add brute force tests
feat: implement brute force search
test: add kmeans tests
feat: implement kmeans
test: add ivf tests
feat: implement ivf-flat
test: add hnsw tests
feat: implement hnsw
test: add integration tests
feat: add fastapi api
feat: add benchmark pipeline
feat: add project test suite
docs: add architecture and benchmarks
```

---

# 63. Configuration

File:

```text
app/config.py
```

Centralize configurable values:

```python
VECTOR_DIM = 128
DEFAULT_K = 10

IVF_NLIST = 100
IVF_NPROBE = 5

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 50
```

Do not scatter magic numbers throughout the codebase.

---

# 64. Error Handling

Create:

```text
app/core/exceptions.py
```

Possible errors:

```text
VectorNotFoundError
DuplicateVectorError
DimensionMismatchError
InvalidSearchParameterError
IndexNotBuiltError
```

Convert these into appropriate API responses.

---

# 65. Search Result Model

Use a common result structure:

```python
SearchResult(
    id="vec_10",
    score=0.9321
)
```

Every index should return the same logical format.

This allows the API to remain index-agnostic.

---

# 66. Index State

Indexes can have states:

```text
EMPTY
BUILDING
READY
DIRTY
```

Example:

```text
insert vector
    ↓
store updated
    ↓
IVF/HNSW marked DIRTY
```

A search against a dirty index should either:

- rebuild automatically, or
- return a clear error.

For the first version, explicit rebuild is acceptable.

---

# 67. Persistence Strategy

First version:

```text
vectors.npy
ids.npy
metadata.json
```

Indexes can be rebuilt at startup.

This is intentionally simple.

Startup:

```text
load vectors
      ↓
load IDs
      ↓
load metadata
      ↓
build indexes
```

Later optimization:

```text
save serialized IVF/HNSW index
```

but this is not required for the hackathon MVP.

---

# 68. Frontend — Later Phase

Do not start frontend development until:

```text
Brute Force ✓
IVF ✓
HNSW ✓
Tests ✓
Benchmark ✓
API ✓
```

Then build a React frontend.

Potential UI:

```text
┌───────────────────────────────────────────────┐
│ VectorForge                                   │
│ From-Scratch Vector Database                  │
├───────────────────────────────────────────────┤
│ Search                                        │
│ [ query................................. ]    │
│ [Brute] [IVF] [HNSW]        Top K: [10]       │
├───────────────────────┬───────────────────────┤
│                       │                       │
│       3D SPACE        │   Search Results      │
│                       │                       │
│       •   •           │  1. vec_102  .98      │
│    •       •          │  2. vec_981  .96      │
│         •             │  3. vec_211  .95      │
│                       │                       │
├───────────────────────┴───────────────────────┤
│ Latency | Recall | Candidates | Speedup       │
└───────────────────────────────────────────────┘
```

---

# 69. 3D Visualization Strategy

Actual vectors may have:

```text
128 dimensions
384 dimensions
768 dimensions
```

A screen cannot directly visualize those dimensions.

Use:

```text
High-dimensional vectors
        ↓
PCA
        ↓
3 dimensions
        ↓
Three.js / React Three Fiber
```

For performance:

Do not render 50,000 individual DOM elements.

Prefer:

```text
GPU point cloud
Instanced rendering
Sampling
```

A good visualization dataset can be:

```text
2,000–5,000 points
```

while the actual engine continues operating on all 50,000 vectors.

---

# 70. Future 3D Features

Potential visualizations:

### Vector space

Show points as:

```text
● ●   ●
  ● ●
       ●
```

### IVF

Color/group points by cluster.

Show:

```text
Cluster 1
Cluster 2
Cluster 3
...
```

### Query

Highlight the query vector.

### Top-K

Highlight retrieved neighbors.

### HNSW

Display:

```text
nodes + edges
```

and animate the search path.

### Performance

Show:

```text
Brute:
50,000 comparisons

IVF:
~2,500 candidates

HNSW:
~few hundred candidate visits
```

Actual numbers should come from instrumentation.

---

# 71. API + Frontend Data Contract

The frontend should not know how IVF or HNSW is implemented internally.

API returns:

```json
{
  "query": "...",
  "index": "hnsw",
  "results": [],
  "metrics": {
    "latency_ms": 1.4,
    "candidates": 321,
    "recall": 0.94
  }
}
```

This keeps frontend and engine loosely coupled.

---

# 72. Demo Story

The final demo should communicate one clear idea:

> **“We built a vector database ourselves and can prove the trade-off between exact search and approximate search.”**

Demo sequence:

```text
1. Explain vector embeddings.
2. Show 50,000 stored vectors.
3. Run exact Brute Force.
4. Show latency.
5. Run IVF.
6. Show fewer candidates.
7. Show recall.
8. Run HNSW.
9. Show graph-based navigation.
10. Compare latency and recall.
11. Insert a vector.
12. Delete a vector.
13. Search again.
```

---

# 73. Suggested README Structure

```text
# VectorForge

## Overview

## Features

## Architecture

## Algorithms
### Brute Force
### IVF-Flat
### HNSW

## Dataset

## Installation

## Running the API

## API Endpoints

## Running Tests

## Running Benchmarks

## Results

## Performance Trade-offs

## Limitations

## Future Work

## Demo
```

---

# 74. Definition of Done

The core engine is considered complete when:

### Storage

- [ ] vectors can be inserted
- [ ] vectors can be retrieved
- [ ] vectors can be deleted
- [ ] metadata works
- [ ] 50,000 vectors can be loaded

### Exact Search

- [ ] cosine similarity works
- [ ] top-k works
- [ ] brute-force results are correct

### IVF

- [ ] K-Means works
- [ ] inverted lists work
- [ ] nprobe works
- [ ] approximate search works
- [ ] deletion works
- [ ] recall is measured

### HNSW

- [ ] graph nodes work
- [ ] levels work
- [ ] neighbors work
- [ ] ef_search works
- [ ] approximate search works
- [ ] deletion works
- [ ] recall is measured

### API

- [ ] health endpoint
- [ ] stats endpoint
- [ ] insert endpoint
- [ ] delete endpoint
- [ ] search endpoint

### Testing

- [ ] unit tests
- [ ] integration tests
- [ ] API tests
- [ ] cross-index tests
- [ ] test_suite.py

### Evaluation

- [ ] 500 queries
- [ ] exact top-10 ground truth
- [ ] Recall@10
- [ ] latency
- [ ] speedup
- [ ] build time

### Documentation

- [ ] README
- [ ] architecture diagram
- [ ] API examples
- [ ] benchmark table
- [ ] demo video

---

# 75. Recommended 12-Hour Execution Plan

Because the project includes both IVF and HNSW, treat the schedule as an aggressive MVP plan.

| Time | Work |
|---|---|
| 0:00–0:30 | Project setup + dependencies |
| 0:30–1:15 | Distance + normalization + tests |
| 1:15–2:00 | Top-k + tests |
| 2:00–2:45 | VectorStore + tests |
| 2:45–3:30 | Brute Force + tests |
| 3:30–4:30 | K-Means + tests |
| 4:30–5:45 | IVF-Flat + tests |
| 5:45–7:30 | HNSW + tests |
| 7:30–8:15 | Integration tests |
| 8:15–9:00 | Dataset + ground truth |
| 9:00–9:45 | Benchmark + recall |
| 9:45–10:45 | FastAPI |
| 10:45–11:15 | Test suite + cleanup |
| 11:15–12:00 | README + demo preparation |

If HNSW becomes unstable, **do not sacrifice Brute Force + IVF correctness**. The assignment can be satisfied with IVF alone, while HNSW can remain an additional experimental feature.

---

# 76. MVP Priority

Priority order:

```text
P0 — Must Work
────────────────────────
Brute Force
Vector Store
Cosine Similarity
Top-K
IVF-Flat
FastAPI
50k Dataset
Ground Truth
Recall@10
Tests

P1 — Strong Enhancement
────────────────────────
HNSW
Benchmark Dashboard
Detailed Metrics
Postman Collection

P2 — Visual Enhancement
────────────────────────
React UI
3D Vector Space
IVF Cluster Visualization
HNSW Graph Visualization
Animated Search Path
```

The implementation should never allow P2 work to block P0 correctness.

---

# 77. Final Architecture Summary

```text
                         VECTORFORGE
                              │
                ┌─────────────┴─────────────┐
                │                           │
             REST API                  Evaluation
                │                           │
          Index Manager               Ground Truth
                │                      Recall@10
        ┌───────┼────────┐             Benchmark
        │       │        │
      Brute     IVF     HNSW
        │       │        │
        │    K-Means     │
        │       │        │
        └───────┼────────┘
                │
           Vector Store
                │
        NumPy + Metadata
                │
       50,000+ Vectors
```

Core principle:

```text
Exact Search
    ↓
Correctness baseline
    ↓
Approximate Search
    ├── IVF-Flat
    └── HNSW
    ↓
Measure Recall + Latency
    ↓
Expose through API
    ↓
Visualize later
```

---

# 78. First Implementation Step

Start with the smallest TDD cycle.

### Step 1

Create:

```text
app/core/distance.py
tests/unit/test_distance.py
```

### Step 2

Write failing tests for:

```text
normalize()
cosine_similarity()
```

### Step 3

Implement the minimum code required to pass.

### Step 4

Refactor.

### Step 5

Run:

```bash
pytest tests/unit/test_distance.py -v
```

Then move to:

```text
Top-K
→ VectorStore
→ Brute Force
→ K-Means
→ IVF
→ HNSW
```

This keeps the project understandable and prevents the ANN implementations from becoming a large untested block of code.

---

# 79. Final Deliverables

The final repository should contain:

```text
✓ Source code
✓ Brute Force index
✓ IVF-Flat index
✓ HNSW index
✓ Vector store
✓ FastAPI service
✓ 50k+ vector dataset generation
✓ 500 query evaluation
✓ Ground truth
✓ Recall@10
✓ Benchmark results
✓ Unit tests
✓ Integration tests
✓ API tests
✓ scripts/test_suite.py
✓ Postman collection
✓ README
✓ Demo video
✓ Optional React + 3D visualization
```

The strongest final presentation is not merely:

> “Here is a vector search API.”

It is:

> **“Here is a vector database we implemented from scratch, here is the exact baseline, here are two ANN algorithms, and here is experimental evidence showing the latency/recall trade-off.”**
