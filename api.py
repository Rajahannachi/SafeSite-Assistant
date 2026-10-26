"""FastAPI service exposing the RAG pipeline.

Endpoints:
    GET  /health                 liveness + number of indexed chunks
    POST /ask                    {"question", "session_id"?} -> {"answer", "sources", "latency_ms"}
    POST /ask/stream             same input, answer streamed as Server-Sent Events
    DELETE /sessions/{id}        forget a conversation

Run:  python -m uvicorn api:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from config import get_settings
from rag_chain import RETRYABLE_ERRORS, RAGPipeline, build_pipeline, select_sources
from store import get_vectorstore

logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["Which PPE is required for concrete cutting?"])
    session_id: str | None = Field(None, max_length=64, description="Optional id to enable follow-up questions.")


class Source(BaseModel):
    file: str
    page: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    pipeline: RAGPipeline
    # In-memory conversation store: session_id -> alternating Human/AI messages.
    sessions: defaultdict[str, list[BaseMessage]] = defaultdict(list)


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load the embedding model and build the chain once, not per request.
    state.pipeline = build_pipeline()
    logger.info("pipeline ready (%d chunks indexed)", len(get_vectorstore().get(include=[])["ids"]))
    yield


app = FastAPI(title="SafeSite Assistant", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error handling: map LLM provider failures to meaningful HTTP codes
# ---------------------------------------------------------------------------

@app.exception_handler(openai.RateLimitError)
async def _rate_limited(_: Request, exc: openai.RateLimitError) -> JSONResponse:
    # Reached only when every model (primary + fallbacks) is still rate-limited after retries.
    return JSONResponse(status_code=503, headers={"Retry-After": "30"},
                        content={"detail": "All free LLMs are rate-limited right now. Retry in ~30 s."})


@app.exception_handler(openai.APITimeoutError)
async def _timeout(_: Request, exc: openai.APITimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": "The LLM provider timed out."})


@app.exception_handler(openai.APIError)
async def _provider_error(_: Request, exc: openai.APIError) -> JSONResponse:
    logger.exception("LLM provider error")
    return JSONResponse(status_code=502, content={"detail": f"LLM provider error: {exc.__class__.__name__}"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _history(session_id: str | None) -> list[BaseMessage]:
    if not session_id:
        return []
    max_messages = 2 * get_settings().max_history_turns
    return state.sessions[session_id][-max_messages:]


def _remember(session_id: str | None, question: str, answer: str) -> None:
    if session_id:
        state.sessions[session_id] += [HumanMessage(question), AIMessage(answer)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "chunks_indexed": len(get_vectorstore().get(include=[])["ids"]),
        "llm_model": settings.llm_model,
        "fallback_models": settings.llm_fallback_models,
        "search_type": settings.search_type,
        "rerank": settings.rerank,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    question = req.question.strip()
    result = await state.pipeline.chain.ainvoke({"question": question, "chat_history": _history(req.session_id)})
    if not result["docs"]:
        raise HTTPException(status_code=503, detail="The vector store is empty. Run `python ingest.py` first.")
    answer: str = result["answer"].strip()
    _remember(req.session_id, question, answer)
    return AskResponse(
        answer=answer,
        sources=[Source(**s) for s in select_sources(answer, result["docs"])],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream tokens as SSE events: `token` events, then one final `done` event with sources."""
    start = time.perf_counter()
    question = req.question.strip()
    # Retrieval happens before streaming starts so that errors surface as HTTP codes.
    retrieved = await state.pipeline.retrieval.ainvoke({"question": question, "chat_history": _history(req.session_id)})

    async def events() -> AsyncIterator[str]:
        parts: list[str] = []
        attempts = get_settings().llm_max_attempts
        for attempt in range(1, attempts + 1):
            try:
                async for token in state.pipeline.answer.astream(retrieved):
                    parts.append(token)
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
                break
            except RETRYABLE_ERRORS as exc:
                # .with_retry() only wraps invoke/batch, not streaming, so transient errors are
                # retried here, and only before the first token: sent text cannot be taken back.
                if parts or attempt == attempts:
                    yield f"event: error\ndata: {json.dumps(exc.__class__.__name__)}\n\n"
                    return
                await asyncio.sleep(2**attempt)
            except openai.APIError as exc:
                yield f"event: error\ndata: {json.dumps(exc.__class__.__name__)}\n\n"
                return
        answer = "".join(parts).strip()
        _remember(req.session_id, question, answer)
        payload = {"sources": select_sources(answer, retrieved["docs"]),
                   "latency_ms": int((time.perf_counter() - start) * 1000)}
        yield f"event: done\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}", status_code=204)
def reset_session(session_id: str) -> None:
    state.sessions.pop(session_id, None)
