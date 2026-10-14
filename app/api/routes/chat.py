from psycopg2.errors import RaiseException
from fastapi import APIRouter, HTTPException
import time 
import logging
from pydantic import BaseModel, Field
from ...services.retrieval import ask


logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel): 
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=10)

class QueryResponse(BaseModel): 
    answer: str


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest): 
    start = time.perf_counter()

    try: 
        answer = ask(req.question, req.top_k)
    except Exception: 
        logger.exception("RAG query failed")
        raise HTTPException("question=%r  top_k=%d  took=%.2fs", req.question, req.top_k, time.perf_counter() -> start)

    return QueryResponse(answer=answer)





