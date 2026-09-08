"""
VectorForge — Pydantic Schemas for API Responses
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    vectors: int
    total_vectors: int
    dimension: int
    indexes: dict[str, str]


class SearchResultItem(BaseModel):
    id: str
    score: float


class SearchResponse(BaseModel):
    index: str
    results: list[SearchResultItem]
    total: int


class TextSearchResultItem(BaseModel):
    id: str
    score: float
    text: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class TextSearchResponse(BaseModel):
    index: str
    query_text: str
    results: list[TextSearchResultItem]
    total: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
