"""
Chat API — SSE streaming endpoint.

POST /api/chat  { "question": "...", "top_k": 5 }

Returns a ``text/event-stream`` where each ``data:`` line carries a
JSON object ``{"token": "..."}`` and the final line is ``data: [DONE]``.
"""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.retrieval import stream_ask

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/chat")
async def chat(req: ChatRequest):
    """Stream an RAG-grounded answer as Server-Sent Events."""

    async def event_stream():
        try:
            async for token in stream_ask(req.question, req.top_k):
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception:
            logger.exception("Streaming error for question: %s", req.question)
            error = json.dumps({"error": "Something went wrong. Please try again."})
            yield f"data: {error}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
