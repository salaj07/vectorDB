"""
VectorForge — Pydantic Schemas for Vector Operations
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class VectorInsertRequest(BaseModel):
    """Request body for POST /vectors"""
    id: str = Field(..., description="Unique vector identifier")
    vector: list[float] = Field(..., description="Embedding vector", min_length=1)
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata")


class BulkVectorItem(BaseModel):
    """Item for bulk insert"""
    id: str = Field(..., description="Unique vector identifier")
    vector: list[float] = Field(..., description="Embedding vector", min_length=1)
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata")


class BulkInsertRequest(BaseModel):
    """Request body for POST /vectors/bulk"""
    vectors: list[BulkVectorItem] = Field(..., min_length=1)


class VectorResponse(BaseModel):
    """Response for vector insert/delete operations"""
    id: str
    status: str


class BulkInsertResponse(BaseModel):
    """Response for bulk insert operations"""
    inserted: int
    status: str


class VectorGetResponse(BaseModel):
    """Response for GET /vectors/{vector_id}"""
    id: str
    vector: list[float]
    metadata: Optional[dict[str, Any]] = None
