"""
embeddings.py — Embedding generation pipeline.

Uses a local sentence-transformers model (configurable via EMBEDDING_MODEL env var).
Falls back gracefully: if the model is unavailable or inference fails, it logs the
error and returns None so the rest of the system keeps working without embeddings.

Default model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  - 384-dimensional vectors
  - Supports Arabic + English (important for this project)
  - Small enough to run on CPU

Change EMBEDDING_MODEL in .env to use a different model.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Lazy-load the model so startup is fast even if the library is missing.
_model = None
_load_attempted = False


def _get_model():
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
    except Exception as exc:
        logger.warning(
            "Could not load embedding model '%s': %s. "
            "Semantic search will be unavailable.",
            EMBEDDING_MODEL,
            exc,
        )
        _model = None
    return _model


def embed(text: str) -> Optional[list[float]]:
    """
    Generate an embedding vector for *text*.

    Returns:
        list[float]  – the embedding vector, or
        None         – if the model is unavailable or inference fails.
    """
    if not text or not text.strip():
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()
    except Exception as exc:
        logger.warning("Embedding inference failed: %s", exc)
        return None
