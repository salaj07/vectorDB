"""
VectorForge — Vector Management Routes

POST /vectors          — Insert a single vector
POST /vectors/bulk     — Bulk insert vectors
GET /vectors/{id}      — Retrieve a vector by ID
DELETE /vectors/{id}   — Delete a vector by ID
"""

from fastapi import APIRouter, HTTPException, Request, status
import numpy as np

from app.core.exceptions import (
    DimensionMismatchError,
    DuplicateVectorError,
    VectorNotFoundError,
)
from app.schemas.vector import (
    BulkInsertRequest,
    BulkInsertResponse,
    VectorGetResponse,
    VectorInsertRequest,
    VectorResponse,
)

router = APIRouter(prefix="/vectors", tags=["vectors"])


@router.post("", response_model=VectorResponse, status_code=status.HTTP_201_CREATED)
async def insert_vector(req: VectorInsertRequest, request: Request):
    """Insert a single vector into the vector store."""
    manager = request.app.state.manager
    try:
        vec_arr = np.array(req.vector, dtype=np.float32)
        manager.insert_vector(req.id, vec_arr, req.metadata)
        return VectorResponse(id=req.id, status="inserted")
    except DuplicateVectorError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DimensionMismatchError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/bulk", response_model=BulkInsertResponse, status_code=status.HTTP_201_CREATED)
async def bulk_insert_vectors(req: BulkInsertRequest, request: Request):
    """Bulk insert multiple vectors into the vector store."""
    manager = request.app.state.manager
    ids = [item.id for item in req.vectors]
    vectors = np.array([item.vector for item in req.vectors], dtype=np.float32)
    metadatas = [item.metadata for item in req.vectors]

    try:
        inserted_count = manager.bulk_insert(ids, vectors, metadatas)
        return BulkInsertResponse(inserted=inserted_count, status="inserted")
    except DuplicateVectorError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DimensionMismatchError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{vector_id}", response_model=VectorGetResponse)
async def get_vector(vector_id: str, request: Request):
    """Retrieve a vector by ID."""
    manager = request.app.state.manager
    try:
        data = manager.get_vector(vector_id)
        if not data.get("active", True):
            raise VectorNotFoundError(vector_id)
        return VectorGetResponse(
            id=vector_id,
            vector=data["vector"].tolist(),
            metadata=data["metadata"],
        )
    except VectorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/{vector_id}", response_model=VectorResponse)
async def delete_vector(vector_id: str, request: Request):
    """Soft delete a vector by ID."""
    manager = request.app.state.manager
    try:
        manager.delete_vector(vector_id)
        return VectorResponse(id=vector_id, status="deleted")
    except VectorNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
