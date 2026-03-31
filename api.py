# run:  uvicorn api:api --reload --host 127.0.0.1 --port 8000

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import db
import embeddings as emb
from app import chat, clear_session, get_raw_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: initialise the DB pool and embedding model when the server starts
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise PostgreSQL connection pool and embedding model on startup."""

    # DB pool
    try:
        db.init_pool()
        logger.info("DB pool initialised successfully.")
    except RuntimeError as exc:
        logger.error("DB init failed: %s", exc)

    # Force-load embedding model now so we know immediately if it works
    vector = emb.embed("startup test")
    if vector is not None:
        logger.info("Embedding model ready (dim=%d).", len(vector))
    else:
        logger.warning(
            "Embedding model NOT loaded — run: pip install sentence-transformers"
        )
    yield


api = FastAPI(title="HF LangChain Chat API", version="2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    question:   str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    answer:     str

class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session identifier to clear")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        answer = chat(req.question, req.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return ChatResponse(session_id=req.session_id, answer=answer)


@api.get("/history/{session_id}")
def history(session_id: str):
    """Return the full chronological chat history for a session."""
    try:
        raw = get_raw_history(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"session_id": session_id, "history": raw}


@api.get("/history/{session_id}/search")
def history_search(
    session_id: str,
    query: str = Query(..., min_length=1, description="Natural-language search query"),
    k:     int = Query(5,  ge=1, le=50,   description="Number of results to return"),
):
    """
    Semantic similarity search over a session's message history.

    Returns the top-k messages most similar to *query*, ranked by
    cosine similarity (highest first).  Only messages that have a
    stored embedding are considered.

    Returns 422 if the embedding model is unavailable.
    """
    query_vec = emb.embed(query)
    if query_vec is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Embedding model is unavailable. "
                "Install sentence-transformers and check EMBEDDING_MODEL in .env."
            ),
        )
    try:
        results = db.search_similar(session_id, query_vec, k=k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Serialise datetime objects to ISO strings for JSON
    for r in results:
        if "created_at" in r and r["created_at"] is not None:
            r["created_at"] = r["created_at"].isoformat()

    return {
        "session_id": session_id,
        "query":      query,
        "k":          k,
        "results":    results,
    }


@api.post("/clear")
def clear(req: ClearRequest):
    try:
        clear_session(req.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "cleared", "session_id": req.session_id}