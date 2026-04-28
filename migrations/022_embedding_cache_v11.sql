-- v11.0 Phase 7 — Embedding cache (multi-space aware).
--
-- Supersedes the v9 `embedding_cache` table from migration 014. The old
-- table keys cache entries by sha256(text) only — fine when the project
-- had a single embedder, but now (v11 §J) every chunk also carries an
-- `embedding_space` plus an explicit provider/model. Different spaces or
-- model swaps must NEVER collide on the same key, so the cache key must
-- mix in (provider, model, space, normalized_text).
--
-- A NEW table name (`embedding_cache_v11`) is used so both caches can
-- coexist while the v10 paths still depend on the original. Phase 5+
-- can deprecate the old table.
--
-- Vector blob format: same struct.pack('f'*dim, *vec) used by the SQLite
-- binary store (Store._float32_to_blob in server.py).

CREATE TABLE IF NOT EXISTS embedding_cache_v11 (
    cache_key       TEXT PRIMARY KEY,            -- sha256(provider||model||space||normalized_text)
    provider        TEXT NOT NULL,               -- fastembed | sentence-transformers | openai | ...
    model           TEXT NOT NULL,               -- e.g. BAAI/bge-small-en-v1.5
    embedding_space TEXT NOT NULL,               -- text | code | log | config
    dim             INTEGER NOT NULL,            -- vector length
    vector_blob     BLOB NOT NULL,               -- packed float32 (4*dim bytes)
    hit_count       INTEGER NOT NULL DEFAULT 0,  -- bumped by get()
    last_used_at    TEXT NOT NULL,               -- ISO-8601 UTC, LRU eviction key
    created_at      TEXT NOT NULL                -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_emb_cache_v11_last_used
    ON embedding_cache_v11(last_used_at);

CREATE INDEX IF NOT EXISTS idx_emb_cache_v11_space
    ON embedding_cache_v11(embedding_space);

INSERT OR IGNORE INTO migrations (version, description)
VALUES ('022', 'Embedding cache v11 — multi-space aware, sha256(provider||model||space||text) key');
