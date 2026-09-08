"""
VectorForge — Search Route

POST /search       — Search nearest neighbors using numerical query vector
POST /search/text  — Search nearest neighbors using text string (e.g. "express js")
"""

from fastapi import APIRouter, HTTPException, Request, status
import numpy as np

from app.core.exceptions import (
    DimensionMismatchError,
    IndexNotBuiltError,
    InvalidSearchParameterError,
)
from app.schemas.response import (
    SearchResponse,
    SearchResultItem,
    TextSearchResponse,
    TextSearchResultItem,
)
from app.schemas.search import SearchRequest, TextSearchRequest
from scripts.generate_text_dataset import text_to_vector

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_vectors(req: SearchRequest, request: Request):
    """
    Search for nearest neighbors using the selected index.

    Query parameters inside body:
    - query: list[float]
    - k: int (default 10)
    - index: "brute", "ivf", or "hnsw"
    - nprobe: int (optional, for IVF)
    - ef_search: int (optional, for HNSW)
    """
    manager = request.app.state.manager

    # Build kwargs for search
    kwargs = {}
    if req.nprobe is not None:
        kwargs["nprobe"] = req.nprobe
    if req.ef_search is not None:
        kwargs["ef_search"] = req.ef_search

    try:
        query_vec = np.array(req.query, dtype=np.float32)
        results = manager.search(
            query=query_vec,
            k=req.k,
            index=req.index,
            **kwargs,
        )

        items = [SearchResultItem(id=res.id, score=res.score) for res in results]
        return SearchResponse(
            index=req.index,
            results=items,
            total=len(items),
        )
    except InvalidSearchParameterError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DimensionMismatchError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/text", response_model=TextSearchResponse)
async def search_text(req: TextSearchRequest, request: Request):
    """
    Search for nearest neighbor statements by supplying a text string directly.

    Query parameters inside body:
    - text: str (e.g. "express js framework for node backend")
    - k: int (default 10)
    - index: "brute", "ivf", or "hnsw"
    - nprobe: int (optional, for IVF)
    - ef_search: int (optional, for HNSW)
    """
    manager = request.app.state.manager

    # Build kwargs for search
    kwargs = {}
    if req.nprobe is not None:
        kwargs["nprobe"] = req.nprobe
    if req.ef_search is not None:
        kwargs["ef_search"] = req.ef_search

    try:
        # Convert text query into vector embedding
        query_vec = text_to_vector(req.text, dim=manager.store.dimension)
        results = manager.search(
            query=query_vec,
            k=req.k,
            index=req.index,
            **kwargs,
        )

        items = []
        for res in results:
            doc = manager.store.get(res.id)
            meta = doc.get("metadata") or {}
            text_val = meta.get("text", f"Vector {res.id}")
            cat_val = meta.get("category", f"Cluster {meta.get('cluster', 'N/A')}")
            items.append(
                TextSearchResultItem(
                    id=res.id,
                    score=res.score,
                    text=text_val,
                    category=cat_val,
                    metadata=meta,
                )
            )

        return TextSearchResponse(
            index=req.index,
            query_text=req.text,
            results=items,
            total=len(items),
        )
    except InvalidSearchParameterError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexNotBuiltError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DimensionMismatchError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
