-- ══════════════════════════════════════════════════════════
-- v9.0 — Two-level cache: L2 embedding cache
--
-- Stores persisted fastembed / provider outputs keyed by sha256(text).
-- Avoids re-running the embedding model for repeated inputs, which is
-- by far the dominant cost in the recall hot path (p50 ~38ms baseline).
--
-- Key design:
--   • key = sha256(normalized_text)  — deterministic, collision-free
--   • embedding stored as packed float32 BLOB (struct-packed, N*4 bytes)
--   • model + dim are kept alongside so a loader can refuse mismatches
--     when the active provider changes (avoids silent shape mix-ups).
--   • created_at is ISO-8601 UTC for ad-hoc TTL eviction via cron.
--
-- Safe to (re)run: CREATE TABLE IF NOT EXISTS is idempotent.
-- ══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS embedding_cache (
    key        TEXT PRIMARY KEY,            -- sha256 of text
    embedding  BLOB NOT NULL,               -- packed float32 vector
    created_at TEXT NOT NULL,               -- ISO-8601 UTC
    model      TEXT NOT NULL,               -- provider/model tag
    dim        INTEGER NOT NULL             -- vector length
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_model
    ON embedding_cache(model);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_created
    ON embedding_cache(created_at);

INSERT OR IGNORE INTO migrations (version, description)
VALUES ('014', 'Embedding cache table — v9 A2 L2 persistent cache keyed by sha256(text)');
