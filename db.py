"""
db.py — PostgreSQL + pgvector persistence layer.

Responsibilities:
  - Manage the connection pool (psycopg2)
  - Create the schema on first run (pgvector + chat_messages table)
  - Insert / query / delete messages
  - Store and search message embeddings

All functions raise RuntimeError with a clear message when the DB is
unavailable, so callers (api.py / app.py) can return HTTP 503 instead
of crashing silently.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (read from environment / .env loaded by app.py)
# ---------------------------------------------------------------------------

PG_HOST     = os.getenv("PG_HOST",     "127.0.0.1")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DB       = os.getenv("PG_DB",       "chatbot")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")

DSN = (
    f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} "
    f"user={PG_USER} password={PG_PASSWORD}"
)

_pool: Optional[pg_pool.ThreadedConnectionPool] = None


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

def init_pool(minconn: int = 1, maxconn: int = 10) -> None:
    """Create the connection pool and initialise the DB schema."""
    global _pool
    try:
        _pool = pg_pool.ThreadedConnectionPool(minconn, maxconn, dsn=DSN)
        logger.info("PostgreSQL connection pool created (%s–%s conns)", minconn, maxconn)
        _create_schema()
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to PostgreSQL: {exc}") from exc


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Yield a connection from the pool and return it afterwards."""
    if _pool is None:
        raise RuntimeError("DB pool is not initialised. Call init_pool() first.")
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Enable pgvector extension (requires pgvector to be installed in Postgres)
CREATE EXTENSION IF NOT EXISTS vector;

-- Main messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    role        TEXT        NOT NULL CHECK (role IN ('human', 'ai')),
    content     TEXT        NOT NULL,
    embedding   vector(384),           -- dimension matches EMBEDDING_MODEL default
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast per-session lookups
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
    ON chat_messages (session_id);

-- Index for vector similarity search (IVFFlat – fast approximate NN)
-- We use IF NOT EXISTS via DO block because CREATE INDEX IF NOT EXISTS
-- is only supported for regular indexes in older PG versions.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'chat_messages'
          AND indexname  = 'idx_chat_messages_embedding'
    ) THEN
        CREATE INDEX idx_chat_messages_embedding
            ON chat_messages
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    END IF;
END
$$;
"""


def _create_schema() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
    logger.info("DB schema verified / created.")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_message(
    session_id: str,
    role: str,
    content: str,
    embedding: Optional[list[float]] = None,
) -> int:
    """Insert a chat message and return its id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, embedding)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (session_id, role, content, embedding),
            )
            row = cur.fetchone()
            return row[0]


def get_messages(session_id: str) -> list[dict]:
    """Return all messages for a session ordered chronologically."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM   chat_messages
                WHERE  session_id = %s
                ORDER  BY created_at ASC, id ASC
                """,
                (session_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def delete_session(session_id: str) -> int:
    """Delete all messages for a session. Returns number of rows deleted."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_messages WHERE session_id = %s",
                (session_id,),
            )
            return cur.rowcount


def update_embedding(message_id: int, embedding: list[float]) -> None:
    """Patch the embedding column for an already-inserted row."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_messages SET embedding = %s WHERE id = %s",
                (embedding, message_id),
            )


def search_similar(
    session_id: str,
    query_embedding: list[float],
    k: int = 5,
) -> list[dict]:
    """
    Return the top-k most semantically similar messages for a session,
    ordered by cosine distance (closest first).

    Only messages that have an embedding are considered.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT   id, role, content, created_at,
                         1 - (embedding <=> %s::vector) AS similarity
                FROM     chat_messages
                WHERE    session_id = %s
                  AND    embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT    %s
                """,
                (query_embedding, session_id, query_embedding, k),
            )
            return [dict(r) for r in cur.fetchall()]