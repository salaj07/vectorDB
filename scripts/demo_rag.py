"""
VectorForge — Interactive RAG (Retrieval-Augmented Generation) CLI Demo

Demonstrates end-to-end vector search + Google Gemini LLM synthesis.

Usage:
    python scripts/demo_rag.py "What backend framework should I use for Node.js?"
    python scripts/demo_rag.py "How does vector similarity search work?"
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR
from app.indexes.manager import IndexManager
from app.services.llm_service import gemini_service
from scripts.generate_text_dataset import text_to_vector, generate_5k_text_dataset

TEXT_DATA_DIR = "data_text"


async def run_rag_demo():
    print("=" * 80)
    print("      VectorForge — Interactive RAG (Vector DB + Gemini LLM) Demo")
    print("=" * 80)

    target_dir = TEXT_DATA_DIR if os.path.exists(os.path.join(TEXT_DATA_DIR, "vectors.npy")) else DATA_DIR
    if not os.path.exists(os.path.join(target_dir, "vectors.npy")):
        print("Generating dataset...")
        generate_5k_text_dataset(total_count=5000, output_dir=target_dir)

    manager = IndexManager(dimension=128)
    print(f"\n[1/3] Loading Vector DB store from '{target_dir}/'...")
    manager.load_data(target_dir)
    print(f"      Loaded {manager.store.count()} text vectors.")

    # Get query from CLI args and optional index flag
    index_name = "brute"
    query_args = []
    
    for arg in sys.argv[1:]:
        if arg.startswith("--index="):
            index_name = arg.split("=", 1)[1].lower()
        elif arg in ("--brute", "--ivf", "--hnsw"):
            index_name = arg.replace("--", "").lower()
        else:
            query_args.append(arg)

    if query_args:
        query_text = " ".join(query_args)
    else:
        query_text = "What is the best web development framework for Node.js backend?"

    print(f"\n[2/3] User Query: \"{query_text}\"")
    print(f"      Executing vector search across [{index_name.upper()}] index...")

    start_retrieval = time.perf_counter()
    query_vec = text_to_vector(query_text, dim=128)
    
    # Configure optimal parameters for maximum recall
    kwargs = {}
    if index_name == "ivf":
        kwargs["nprobe"] = 10
    elif index_name == "hnsw":
        kwargs["ef_search"] = 100

    results = manager.search(query_vec, k=3, index=index_name, **kwargs)
    retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0

    contexts = []
    print("\n      Retrieved Ground Truth Contexts:")
    for rank, res in enumerate(results, 1):
        doc = manager.store.get(res.id)
        meta = doc.get("metadata") or {}
        text_str = meta.get("text", f"Vector {res.id}")
        cat_str = meta.get("category", f"Cluster {meta.get('cluster', 'N/A')}")
        print(f"        {rank}. [Similarity: {res.score:.4f}] ({cat_str}): \"{text_str}\"")
        contexts.append({
            "id": res.id,
            "text": text_str,
            "similarity": float(res.score),
        })

    print(f"\n[3/3] Sending context to Google Gemini LLM ({gemini_service.model})...")
    answer, llm_ms, model_used = await gemini_service.generate_rag_response(
        query=query_text,
        contexts=contexts,
    )

    print("\n" + "=" * 80)
    print("                        GEMINI LLM RESPONSE")
    print("=" * 80)
    print(f"{answer}")
    print("=" * 80)
    print(f" Performance Metrics:")
    print(f"   • Vector Retrieval Latency (HNSW): {retrieval_ms:.2f} ms")
    print(f"   • Gemini LLM Synthesis Latency:  {llm_ms:.2f} ms")
    print(f"   • Model Identifier:             {model_used}")
    print("=" * 80 + "\n")


def main():
    asyncio.run(run_rag_demo())


if __name__ == "__main__":
    main()
