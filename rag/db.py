import os
import psycopg

DSN = os.environ.get("RAG_DSN", "postgresql://postgres:rag@localhost:5433/rag")
EMBED_DIM = 384  # BAAI/bge-small-en-v1.5

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    id        serial PRIMARY KEY,
    doc_id    text NOT NULL,
    title     text NOT NULL,
    chunk_ix  int  NOT NULL,
    content   text NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED,
    embedding vector({EMBED_DIM}) NOT NULL
);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN, autocommit=True)
