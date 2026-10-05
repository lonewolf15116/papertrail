-- Runs once when the Postgres volume is first created.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    paper_id   TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    arxiv_id   TEXT,
    venue      TEXT,
    grp        TEXT
);

-- Embedding width matches the default model (bge-small-en-v1.5 = 384). Change it with the model.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    paper_id   TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    section    TEXT NOT NULL,
    page       INT  NOT NULL CHECK (page >= 1),
    text       TEXT NOT NULL,
    tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding  VECTOR(384),
    -- same model, embedded as "<paper title>. <section>. <text>" (contextual chunk header)
    embedding_header VECTOR(384)
);

-- Existing databases created before the header column existed.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_header VECTOR(384);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_paper_idx ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_embedding_header_idx
    ON chunks USING hnsw (embedding_header vector_cosine_ops);
