"""
VectorForge — RAG Chat Endpoint (Gemini LLM Integration)

POST /chat        — Standard RAG pipeline (Vector DB retrieval + Gemini LLM)
POST /chat/cached — Semantic cache RAG (checks cache first, caches new responses)
"""

import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, status
import numpy as np

from app.core.exceptions import (
    DimensionMismatchError,
    IndexNotBuiltError,
    InvalidSearchParameterError,
)
from app.schemas.chat import ChatRequest, ChatResponse, CachedChatResponse, SourceContext
from app.services.llm_service import gemini_service
from scripts.generate_text_dataset import text_to_vector

router = APIRouter(prefix="/chat", tags=["chat"])

CACHE_SIMILARITY_THRESHOLD = 0.90


@router.post("", response_model=ChatResponse)
async def rag_chat(req: ChatRequest, request: Request):
    """
    RAG Chat endpoint:
    1. Vectorizes user text query.
    2. Performs top-k vector search in selected VectorForge index (brute, ivf, hnsw).
    3. Fetches ground truth source text statements & metadata.
    4. Generates grounded answer using Google Gemini LLM.
    """
    manager = request.app.state.manager

    # Step 1: Measure Retrieval Latency
    start_retrieval = time.perf_counter()
    try:
        query_vec = text_to_vector(req.query, dim=manager.store.dimension)
        search_results = manager.search(
            query=query_vec,
            k=req.k,
            index=req.index,
        )

        sources: list[SourceContext] = []
        context_dicts: list[dict] = []

        for res in search_results:
            doc = manager.store.get(res.id)
            meta = doc.get("metadata") or {}
            text_val = meta.get("text", f"Vector {res.id}")

            source_ctx = SourceContext(
                id=res.id,
                text=text_val,
                similarity=float(res.score),
            )
            sources.append(source_ctx)
            context_dicts.append({
                "id": res.id,
                "text": text_val,
                "similarity": float(res.score),
            })

    except InvalidSearchParameterError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DimensionMismatchError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Retrieval error: {str(e)}")

    retrieval_latency_ms = (time.perf_counter() - start_retrieval) * 1000.0

    # Step 2: Query Gemini LLM for synthesis
    answer, llm_latency_ms, model_used = await gemini_service.generate_rag_response(
        query=req.query,
        contexts=context_dicts,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )

    return ChatResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        index_used=req.index,
        retrieval_latency_ms=round(retrieval_latency_ms, 3),
        llm_latency_ms=round(llm_latency_ms, 3),
        model_used=model_used,
    )


@router.post("/cached", response_model=CachedChatResponse)
async def cached_rag_chat(req: ChatRequest, request: Request):
    """
    Semantic Cache RAG endpoint:
    1. Vectorizes user text query.
    2. Searches the DB (original data + cached responses).
    3. If a cached LLM response is found with similarity >= 0.90 → returns it instantly.
    4. Otherwise calls Gemini LLM → caches the response in the DB → returns it.
    5. The DB grows over time: 5000 → 5001 → 5002 → ...

    Next time a similar query comes in, it gets served from cache — no LLM call!
    """
    manager = request.app.state.manager

    start = time.perf_counter()
    try:
        query_vec = text_to_vector(req.query, dim=manager.store.dimension)

        # Search all vectors (original + cached)
        search_results = manager.search(
            query=query_vec, k=5, index=req.index,
        )
    except (InvalidSearchParameterError, IndexNotBuiltError, DimensionMismatchError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # ── Check for cache hit ──
    for res in search_results:
        doc = manager.store.get(res.id)
        meta = doc.get("metadata") or {}
        if meta.get("source") == "llm_cache" and res.score >= CACHE_SIMILARITY_THRESHOLD:
            retrieval_ms = (time.perf_counter() - start) * 1000.0
            return CachedChatResponse(
                query=req.query,
                answer=meta.get("llm_response", ""),
                sources=[SourceContext(
                    id=res.id,
                    text=meta.get("query", ""),
                    similarity=float(res.score),
                )],
                index_used=req.index,
                retrieval_latency_ms=round(retrieval_ms, 3),
                llm_latency_ms=0.0,
                model_used="cache",
                cache_hit=True,
                cached_at=meta.get("timestamp"),
                original_query=meta.get("query"),
                db_size=manager.store.total_count(),
            )

    # ── Cache miss — gather contexts and call LLM ──
    sources: list[SourceContext] = []
    context_dicts: list[dict] = []

    # Use top-k results as LLM context
    for res in search_results[:req.k]:
        doc = manager.store.get(res.id)
        meta = doc.get("metadata") or {}
        text_val = meta.get("text", f"Vector {res.id}")
        sources.append(SourceContext(id=res.id, text=text_val, similarity=float(res.score)))
        context_dicts.append({"id": res.id, "text": text_val, "similarity": float(res.score)})

    retrieval_ms = (time.perf_counter() - start) * 1000.0

    # Call Gemini LLM
    answer, llm_ms, model_used = await gemini_service.generate_rag_response(
        query=req.query,
        contexts=context_dicts,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )

    # ── Cache the response in the vector DB ──
    cache_id = f"cache_{manager.store.total_count()}"
    cache_meta = {
        "source": "llm_cache",
        "query": req.query,
        "llm_response": answer,
        "text": f"[Cached] {req.query}",
        "category": "LLM Cache",
        "timestamp": datetime.now().isoformat(),
        "context_ids": [c["id"] for c in context_dicts],
    }

    try:
        manager.store.insert(cache_id, query_vec, cache_meta)
        manager.build_index("brute")
    except Exception:
        pass  # Non-critical — caching failure shouldn't break the response

    return CachedChatResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        index_used=req.index,
        retrieval_latency_ms=round(retrieval_ms, 3),
        llm_latency_ms=round(llm_ms, 3),
        model_used=model_used,
        cache_hit=False,
        cached_at=None,
        original_query=None,
        db_size=manager.store.total_count(),
    )
