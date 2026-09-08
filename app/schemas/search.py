"""
VectorForge — Pydantic Schemas for Search Operations
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for POST /search"""
    query: list[float] = Field(..., description="Query vector", min_length=1)
    k: int = Field(default=10, ge=1, description="Number of results")
    index: str = Field(default="brute", description="Index to use: brute, ivf, hnsw")
    nprobe: Optional[int] = Field(default=None, ge=1, description="IVF clusters to search")
    ef_search: Optional[int] = Field(default=None, ge=1, description="HNSW search width")


class TextSearchRequest(BaseModel):
    """Request body for POST /search/text"""
    text: str = Field(..., description="Query text string to search")
    k: int = Field(default=10, ge=1, description="Number of results")
    index: str = Field(default="brute", description="Index to use: brute, ivf, hnsw")
    nprobe: Optional[int] = Field(default=None, ge=1, description="IVF clusters to search")
    ef_search: Optional[int] = Field(default=None, ge=1, description="HNSW search width")
