"""
VectorForge — Heap Utilities for HNSW

Min-heap and max-heap wrappers for efficient candidate tracking
during HNSW search and construction.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class HeapItem:
    """A scored item for heap operations. Ordered by priority (distance)."""
    priority: float
    item: int = field(compare=False)


class MinHeap:
    """Min-heap: smallest priority on top."""

    def __init__(self):
        self._heap: list[HeapItem] = []

    def push(self, priority: float, item: int) -> None:
        heapq.heappush(self._heap, HeapItem(priority, item))

    def pop(self) -> tuple[float, int]:
        h = heapq.heappop(self._heap)
        return h.priority, h.item

    def peek(self) -> tuple[float, int]:
        h = self._heap[0]
        return h.priority, h.item

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0


class MaxHeap:
    """Max-heap: largest priority on top (via negation)."""

    def __init__(self):
        self._heap: list[HeapItem] = []

    def push(self, priority: float, item: int) -> None:
        heapq.heappush(self._heap, HeapItem(-priority, item))

    def pop(self) -> tuple[float, int]:
        h = heapq.heappop(self._heap)
        return -h.priority, h.item

    def peek(self) -> tuple[float, int]:
        h = self._heap[0]
        return -h.priority, h.item

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def to_sorted_list(self) -> list[tuple[float, int]]:
        """Return all items sorted by priority descending."""
        items = [(-h.priority, h.item) for h in self._heap]
        items.sort(key=lambda x: x[0], reverse=True)
        return items
