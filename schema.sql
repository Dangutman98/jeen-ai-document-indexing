-- Document index schema.
--
-- Embeddings are stored at 1536 dimensions rather than gemini-embedding-001's
-- 3072-dim default. pgvector's HNSW and IVFFlat indexes only support up to
-- 2000 dimensions, so a 3072-dim column can never be indexed -- every
-- similarity query would fall back to a full sequential scan regardless of
-- table size. We request output_dimensionality=1536 from the Gemini API
-- (Matryoshka truncation) and L2-renormalize the result, as required by
-- Google for any non-default output size. See src/embeddings.py.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id              BIGSERIAL PRIMARY KEY,
    chunk_text      TEXT        NOT NULL,
    embedding       VECTOR(1536) NOT NULL,
    filename        TEXT        NOT NULL,
    split_strategy  TEXT        NOT NULL CHECK (split_strategy IN ('fixed', 'sentence', 'paragraph')),
    chunk_index     INTEGER     NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW over cosine distance. Built after the table exists; safe to run
-- again on an already-indexed table because of IF NOT EXISTS.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS document_chunks_filename_idx
    ON document_chunks (filename);
