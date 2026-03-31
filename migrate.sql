-- migrate.sql
-- Run once against your PostgreSQL database to prepare it for the chatbot.
-- Example:  psql -U postgres -d chatbot -f migrate.sql

-- 1. Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the messages table
--    embedding dimension = 384 (matches paraphrase-multilingual-MiniLM-L12-v2)
--    Change to 768 if you switch to a larger model.
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL    PRIMARY KEY,
    session_id  TEXT         NOT NULL,
    role        TEXT         NOT NULL CHECK (role IN ('human', 'ai')),
    content     TEXT         NOT NULL,
    embedding   vector(384),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- 3. Fast lookup by session
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
    ON chat_messages (session_id);

-- 4. Approximate nearest-neighbour index for similarity search
--    Requires at least one row in the table before it can be built.
--    The application (db.py) creates this automatically on startup.
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
