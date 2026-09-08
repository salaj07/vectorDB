"""
VectorForge — FastAPI Application Main Entry Point
"""

from contextlib import asynccontextmanager
import os
from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_search import router as search_router
from app.api.routes_system import router as system_router
from app.api.routes_vectors import router as vectors_router
from app.config import DATA_DIR
from app.indexes.manager import IndexManager

TEXT_DATA_DIR = "data_text"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize IndexManager on startup — fast load in <0.2s."""
    manager = IndexManager()
    
    target_dir = TEXT_DATA_DIR if os.path.exists(os.path.join(TEXT_DATA_DIR, "vectors.npy")) else DATA_DIR

    if (
        os.path.exists(os.path.join(target_dir, "vectors.npy"))
        and not os.environ.get("VECTORFORGE_SKIP_AUTOLOAD")
    ):
        manager.store.load(target_dir)
        manager.build_all()
    else:
        manager.build_index("brute")

    app.state.manager = manager
    yield


app = FastAPI(
    title="VectorForge API",
    description="Custom Vector Database from Scratch — Exact & Approximate Nearest Neighbor Search",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(system_router)
app.include_router(vectors_router)
app.include_router(search_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": "Welcome to VectorForge API", "docs": "/docs"}
