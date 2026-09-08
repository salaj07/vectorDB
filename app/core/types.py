"""
VectorForge — Common Types

Shared data structures used across all indexes and the API layer.
"""

from dataclasses import dataclass
from enum import Enum


class IndexState(Enum):
    """Lifecycle state of an index."""
    EMPTY = "empty"
    BUILDING = "building"
    READY = "ready"
    DIRTY = "dirty"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    A single search result.

    Attributes:
        id:    The vector identifier (string).
        score: Cosine similarity score (higher = more similar).
    """
    id: str
    score: float

    def to_dict(self) -> dict:
        return {"id": self.id, "score": round(self.score, 6)}
