"""
FastAPI application — the single entry-point.

    uvicorn app.main:app --reload

Lifespan:
  • Ensures the pgvector table exists
  • Starts the file-watcher in a background thread
  • Serves the chat UI on ``/``
"""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.services.ingestion.embedding import get_connection_db, setup_table
from app.watcher import start_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Ensure pgvector table exists
    conn = get_connection_db()
    try:
        setup_table(conn)
        logger.info("✓ document_chunks table ready")
    finally:
        conn.close()

    # 2. Start file-watcher in a daemon thread
    def _watch():
        observer = start_watcher()
        observer.join()       # blocks this thread until shutdown

    watcher_thread = threading.Thread(target=_watch, daemon=True)
    watcher_thread.start()
    logger.info("✓ File watcher started")

    yield  # ← app is running

    logger.info("Shutting down…")


# ── App factory ───────────────────────────────────────────────────

app = FastAPI(
    title="Portfolio Bot",
    description="RAG-powered portfolio assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(chat_router, prefix="/api")

# Serve static assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the chat UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))
