-- Migration 004: Deep enrichment (entities/intent/topics) — async queue + storage
--
-- Analogous to triple_extraction_queue: memory_save enqueues a knowledge_id,
-- reflection.agent drains the queue with deep_enricher.deep_enrich() and
-- persists the result into knowledge_enrichment.
--
-- Enrichment data is indexable for metadata filtering at retrieval time
-- (e.g. "find facts tagged topic=authentication").

CREATE TABLE IF NOT EXISTS knowledge_enrichment (
    knowledge_id  INTEGER PRIMARY KEY,
    entities      TEXT NOT NULL DEFAULT '[]',  -- JSON array: [{"name","type"}, ...]
    intent        TEXT NOT NULL DEFAULT 'unknown',
    topics        TEXT NOT NULL DEFAULT '[]',  -- JSON array: ["auth", "security", ...]
    updated_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kenr_intent ON knowledge_enrichment(intent);

CREATE TABLE IF NOT EXISTS deep_enrichment_queue (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id   INTEGER NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | done | failed
    attempts       INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT,
    created_at     TEXT NOT NULL,
    claimed_at     TEXT,
    processed_at   TEXT,
    UNIQUE(knowledge_id, status)
);

CREATE INDEX IF NOT EXISTS idx_deq_status ON deep_enrichment_queue(status);
CREATE INDEX IF NOT EXISTS idx_deq_created ON deep_enrichment_queue(created_at);
