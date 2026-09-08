"""
VectorForge -- Semantic Cache RAG Demo

Demonstrates LLM response caching via the vector database:
  1. Loads existing 5K text vectors from data_text/
  2. Loads any previously saved LLM knowledge from llm_knowledge.json
  3. Searches for similar content in the DB
  4. If a cached LLM response exists with high similarity -> returns it instantly
  5. Otherwise calls Gemini LLM -> inserts the response into the DB AND
     saves it to llm_knowledge.json (survives dataset resets)

Usage:
    python scripts/demo_cached_rag.py "express js framework"
    python scripts/demo_cached_rag.py "what is kubernetes"
    python scripts/demo_cached_rag.py "express js framework"   # cache hit!
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import DATA_DIR
from app.core.distance import normalize
from app.indexes.manager import IndexManager
from app.services.llm_service import gemini_service
from scripts.generate_text_dataset import text_to_vector, generate_5k_text_dataset

TEXT_DATA_DIR = "data_text"
CACHE_SIMILARITY_THRESHOLD = 0.90
LLM_KNOWLEDGE_FILE = os.path.join(TEXT_DATA_DIR, "llm_knowledge.json")


# ------------------------------------------------------------------
# Persistent LLM Knowledge Store
# ------------------------------------------------------------------
def load_llm_knowledge() -> list[dict]:
    """Load saved LLM responses from the persistent knowledge file."""
    if os.path.exists(LLM_KNOWLEDGE_FILE):
        with open(LLM_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_llm_knowledge(knowledge: list[dict]) -> None:
    """Save LLM responses to the persistent knowledge file."""
    os.makedirs(os.path.dirname(LLM_KNOWLEDGE_FILE), exist_ok=True)
    with open(LLM_KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)


def inject_knowledge_into_store(manager: IndexManager, knowledge: list[dict]) -> int:
    """
    Re-inject saved LLM responses into the vector store.
    
    This is what makes them survive dataset resets -- even if you regenerate
    the 5K dataset, the LLM knowledge gets re-inserted on load.
    Only inserts entries that aren't already in the store.
    """
    injected = 0
    for entry in knowledge:
        vid = entry["id"]
        # Skip if already loaded (from a previous save of vectors.npy)
        if vid in manager.store._id_to_idx:
            continue

        vec = text_to_vector(entry["query"], dim=manager.store.dimension)
        metadata = {
            "source": "llm_cache",
            "query": entry["query"],
            "llm_response": entry["llm_response"],
            "text": f"[Cached] {entry['query']}",
            "category": "LLM Cache",
            "timestamp": entry.get("timestamp", ""),
            "context_ids": entry.get("context_ids", []),
        }
        manager.store.insert(vid, vec, metadata)
        injected += 1

    if injected > 0:
        manager.build_index("brute")

    return injected


# ------------------------------------------------------------------
# Cache lookup & insertion
# ------------------------------------------------------------------
def find_cache_hit(manager: IndexManager, query_vec: np.ndarray, threshold: float):
    """Search the DB for a cached LLM response matching this query."""
    results = manager.search(query_vec, k=5, index="brute")

    for res in results:
        doc = manager.store.get(res.id)
        meta = doc.get("metadata") or {}

        if meta.get("source") == "llm_cache" and res.score >= threshold:
            return {
                "text": meta.get("llm_response", ""),
                "original_query": meta.get("query", ""),
                "score": res.score,
                "id": res.id,
                "cached_at": meta.get("timestamp", ""),
            }

    return None


def insert_cached_response(manager, query_text, llm_response, contexts, knowledge):
    """
    Insert LLM response into BOTH the vector store AND the persistent knowledge file.
    """
    query_vec = text_to_vector(query_text, dim=manager.store.dimension)
    cache_id = f"cache_{manager.store.total_count()}"
    timestamp = datetime.now().isoformat()

    metadata = {
        "source": "llm_cache",
        "query": query_text,
        "llm_response": llm_response,
        "text": f"[Cached] {query_text}",
        "category": "LLM Cache",
        "timestamp": timestamp,
        "context_ids": [c.get("id", "") for c in contexts],
    }

    # 1. Insert into in-memory vector store
    manager.store.insert(cache_id, query_vec, metadata)
    manager.build_index("brute")

    # 2. Append to persistent knowledge file (survives dataset resets)
    knowledge.append({
        "id": cache_id,
        "query": query_text,
        "llm_response": llm_response,
        "timestamp": timestamp,
        "context_ids": [c.get("id", "") for c in contexts],
    })
    save_llm_knowledge(knowledge)

    return cache_id


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
async def run_cached_rag():
    print("=" * 80)
    print("   VectorForge -- Semantic Cache RAG Demo (Search -> Cache -> Reuse)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Load existing text dataset
    # ------------------------------------------------------------------
    target_dir = TEXT_DATA_DIR if os.path.exists(os.path.join(TEXT_DATA_DIR, "vectors.npy")) else DATA_DIR
    if not os.path.exists(os.path.join(target_dir, "vectors.npy")):
        print("No dataset found. Generating 5K text dataset first...")
        generate_5k_text_dataset(total_count=5000, output_dir=target_dir)

    manager = IndexManager(dimension=128)
    print(f"\n[1/5] Loading Vector DB from '{target_dir}/'...")
    manager.load_data(target_dir)

    base_count = manager.store.total_count()

    # ------------------------------------------------------------------
    # Step 2: Load & inject persistent LLM knowledge
    # ------------------------------------------------------------------
    knowledge = load_llm_knowledge()
    injected = inject_knowledge_into_store(manager, knowledge)

    total = manager.store.total_count()
    cached_count = len(knowledge)
    original_count = total - cached_count

    if injected > 0:
        print(f"[2/5] Re-injected {injected} LLM responses from llm_knowledge.json")
    else:
        print(f"[2/5] LLM knowledge loaded ({cached_count} cached responses)")

    print(f"      Total: {total} vectors ({original_count} original + {cached_count} cached)")

    # ------------------------------------------------------------------
    # Step 3: Parse query from CLI
    # ------------------------------------------------------------------
    threshold = CACHE_SIMILARITY_THRESHOLD
    query_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--threshold="):
            threshold = float(arg.split("=", 1)[1])
        else:
            query_args.append(arg)

    if query_args:
        query_text = " ".join(query_args)
    else:
        query_text = "What is the best web framework for Node.js?"

    print(f"\n[3/5] Query: \"{query_text}\"")
    print(f"      Cache threshold: {threshold}")

    # ------------------------------------------------------------------
    # Step 4: Search DB -- check for cache hit first
    # ------------------------------------------------------------------
    query_vec = text_to_vector(query_text, dim=128)

    start = time.perf_counter()
    cache_hit = find_cache_hit(manager, query_vec, threshold)
    search_ms = (time.perf_counter() - start) * 1000.0

    if cache_hit:
        # -- CACHE HIT -- return cached LLM response directly --
        print(f"\n[4/5] CACHE HIT! (similarity: {cache_hit['score']:.4f})")
        print(f"      Cached for query: \"{cache_hit['original_query']}\"")
        print(f"      Cached at: {cache_hit['cached_at']}")

        print("\n" + "=" * 80)
        print("              CACHED LLM RESPONSE (No API Call Needed!)")
        print("=" * 80)
        print(cache_hit["text"])
        print("=" * 80)
        print(f"  Performance:")
        print(f"    * Cache lookup latency:  {search_ms:.2f} ms")
        print(f"    * Gemini LLM call:       SKIPPED (served from cache)")
        print(f"    * Vector DB size:        {manager.store.total_count()} vectors")
        print(f"  Persistence:")
        print(f"    * llm_knowledge.json:    {len(knowledge)} saved responses")
        print(f"    * Survives dataset reset: YES")
        print("=" * 80 + "\n")
        return

    # -- CACHE MISS -- search original corpus, call LLM, cache result --
    print(f"\n[4/5] Cache miss. Searching original corpus...")

    results = manager.search(query_vec, k=3, index="brute")
    retrieval_ms = (time.perf_counter() - start) * 1000.0

    contexts = []
    print("\n      Retrieved contexts from Vector DB:")
    for rank, res in enumerate(results, 1):
        doc = manager.store.get(res.id)
        meta = doc.get("metadata") or {}
        text_str = meta.get("text", f"Vector {res.id}")
        cat_str = meta.get("category", f"Cluster {meta.get('cluster', 'N/A')}")
        source_tag = " [CACHED]" if meta.get("source") == "llm_cache" else ""
        print(f"        {rank}. [Sim: {res.score:.4f}] ({cat_str}){source_tag}: \"{text_str}\"")
        contexts.append({
            "id": res.id,
            "text": text_str,
            "similarity": float(res.score),
        })

    # ------------------------------------------------------------------
    # Step 5: Call Gemini LLM -> Cache in DB + persistent knowledge file
    # ------------------------------------------------------------------
    print(f"\n[5/5] Calling Gemini LLM ({gemini_service.model})...")

    answer, llm_ms, model_used = await gemini_service.generate_rag_response(
        query=query_text,
        contexts=contexts,
    )

    # Insert into both vector store AND persistent knowledge file
    cache_id = insert_cached_response(manager, query_text, answer, contexts, knowledge)

    # Also save updated vectors to disk
    manager.store.save(target_dir)

    new_total = manager.store.total_count()

    print("\n" + "=" * 80)
    print("                        GEMINI LLM RESPONSE")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    print(f"  Performance:")
    print(f"    * Vector retrieval:      {retrieval_ms:.2f} ms")
    print(f"    * Gemini LLM latency:    {llm_ms:.2f} ms")
    print(f"    * Model:                 {model_used}")
    print(f"  Storage:")
    print(f"    * Response cached as:    {cache_id}")
    print(f"    * Vector DB:             {original_count} original + {len(knowledge)} cached = {new_total} total")
    print(f"    * Saved to vectors:      {target_dir}/vectors.npy")
    print(f"    * Saved to knowledge:    {LLM_KNOWLEDGE_FILE}")
    print(f"    * Survives dataset reset: YES (llm_knowledge.json is separate)")
    print(f"    > Next similar query will be served from cache!")
    print("=" * 80 + "\n")


def main():
    asyncio.run(run_cached_rag())


if __name__ == "__main__":
    main()
