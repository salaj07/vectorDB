"""
VectorForge — System Routes

GET /health
GET /stats
POST /rebuild/{index_name}
POST /seed
"""

import os
from fastapi import APIRouter, Request

from app.config import DATA_DIR
from app.schemas.response import HealthResponse, StatsResponse
from scripts.generate_dataset import generate_clustered_dataset

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request):
    """Return system statistics."""
    manager = request.app.state.manager
    s = manager.stats()
    return StatsResponse(**s)


@router.post("/rebuild/{index_name}")
async def rebuild_index(index_name: str, request: Request):
    """Rebuild a specific index."""
    manager = request.app.state.manager
    manager.build_index(index_name)
    return {"index": index_name, "status": "rebuilt"}


@router.post("/seed")
async def seed_data(request: Request):
    """Generate and load synthetic dataset into the server."""
    if not os.path.exists(os.path.join(DATA_DIR, "vectors.npy")):
        generate_clustered_dataset()
    manager = request.app.state.manager
    manager.load_data(DATA_DIR)
    return {"status": "seeded", "vectors": manager.store.count()}
