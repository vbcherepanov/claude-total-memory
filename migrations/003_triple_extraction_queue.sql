-- Migration 003: Triple extraction queue
--
-- memory_save pushes knowledge_id here; a background worker (reflection.agent
-- or dedicated cron) calls ConceptExtractor.extract_and_link(deep=True) which
-- populates graph_edges with (subject, predicate, object) triples. Keeps the
-- hot save path fast (<1ms enqueue) while still getting real KG extraction.

CREATE TABLE IF NOT EXISTS triple_extraction_queue (
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

CREATE INDEX IF NOT EXISTS idx_teq_status ON triple_extraction_queue(status);
CREATE INDEX IF NOT EXISTS idx_teq_created ON triple_extraction_queue(created_at);
