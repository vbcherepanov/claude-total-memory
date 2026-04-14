-- Migration 002: Multi-representation embeddings (GEM-RAG)
--
-- Adds knowledge_representations table to store multiple embedding "views"
-- of the same knowledge record:
--   - raw:       original content (already lives in `embeddings`, mirrored here
--                optionally when multi-repr enabled)
--   - summary:   LLM-generated short summary
--   - keywords:  LLM-extracted salient keywords
--   - questions: LLM-generated utility questions this record answers
--
-- Matching via any representation boosts recall. Fuse scores with RRF.
--
-- Backward compatible: existing `embeddings` table untouched.

CREATE TABLE IF NOT EXISTS knowledge_representations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id    INTEGER NOT NULL,
    representation  TEXT NOT NULL,          -- 'raw' | 'summary' | 'keywords' | 'questions'
    content         TEXT NOT NULL,          -- the transformed text (for audit/debug)
    binary_vector   BLOB NOT NULL,
    float32_vector  BLOB NOT NULL,
    embed_model     TEXT NOT NULL,
    embed_dim       INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE(knowledge_id, representation)
);

CREATE INDEX IF NOT EXISTS idx_krepr_kid ON knowledge_representations(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_krepr_type ON knowledge_representations(representation);
CREATE INDEX IF NOT EXISTS idx_krepr_kid_type
    ON knowledge_representations(knowledge_id, representation);
