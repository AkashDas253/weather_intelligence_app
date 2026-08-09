-- Create dedicated database schema
CREATE SCHEMA IF NOT EXISTS weather;

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Unstructured raw weather documents table
CREATE TABLE IF NOT EXISTS weather.weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
    headline TEXT,
    narrative_text TEXT NOT NULL,
    issued_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    payload JSONB,
    synced_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Chunked vector embeddings table
CREATE TABLE IF NOT EXISTS weather.weather_embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES weather.weather_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_doc_chunk UNIQUE (document_id, chunk_index)
);

-- Indexes inside the weather schema
CREATE INDEX IF NOT EXISTS idx_weather_documents_location 
    ON weather.weather_documents(location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_issued_at 
    ON weather.weather_documents(issued_at DESC);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_doc_id 
    ON weather.weather_embeddings(document_id);

-- Hierarchical Navigable Small World (HNSW) index for vector cosine distance search
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_cosine 
    ON weather.weather_embeddings USING hnsw (embedding vector_cosine_ops);