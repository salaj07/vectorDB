"""
VectorForge — Pydantic Schemas for RAG Chat & LLM Integration
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for POST /chat"""
    query: str = Field(..., description="User question or prompt for RAG")
    index: str = Field(default="brute", description="Vector index to search context: brute, ivf, hnsw")
    k: int = Field(default=3, ge=1, le=20, description="Number of context items to retrieve")
    system_prompt: Optional[str] = Field(default=None, description="Custom system instruction for LLM")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM sampling temperature")


class SourceContext(BaseModel):
    """Retrieved source context item"""
    id: str = Field(..., description="Vector ID")
    text: str = Field(..., description="Source text statement")
    similarity: float = Field(..., description="Cosine similarity score")


class ChatResponse(BaseModel):
    """Response payload from POST /chat"""
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Generated answer from Gemini LLM")
    sources: list[SourceContext] = Field(default_factory=list, description="Retrieved ground truth contexts")
    index_used: str = Field(..., description="Index used for retrieval")
    retrieval_latency_ms: float = Field(..., description="Context retrieval duration in ms")
    llm_latency_ms: float = Field(..., description="LLM generation duration in ms")
    model_used: str = Field(..., description="LLM model identifier")


class CachedChatResponse(BaseModel):
    """Response payload from POST /chat/cached — includes cache status"""
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="LLM answer (from cache or freshly generated)")
    sources: list[SourceContext] = Field(default_factory=list, description="Retrieved contexts")
    index_used: str = Field(..., description="Index used for retrieval")
    retrieval_latency_ms: float = Field(..., description="Retrieval duration in ms")
    llm_latency_ms: float = Field(..., description="LLM duration (0 if cache hit)")
    model_used: str = Field(..., description="LLM model identifier")
    cache_hit: bool = Field(..., description="True if response was served from cache")
    cached_at: Optional[str] = Field(default=None, description="Timestamp when response was cached")
    original_query: Optional[str] = Field(default=None, description="Original query that generated the cached response")
    db_size: int = Field(default=0, description="Current total vectors in DB")
