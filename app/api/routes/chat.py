import logging
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.retrieval import ask


logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.perf_counter()

    try:
        answer = ask(req.question, req.top_k)

        elapsed = time.perf_counter() - start
        logger.info("RAG query completed in %.2fs", elapsed)

        return QueryResponse(answer=answer)

    except Exception as exc:
        logger.exception("RAG query failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from exc
