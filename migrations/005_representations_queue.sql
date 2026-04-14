-- Migration 005: Async generation queue for multi-representation embeddings
--
-- memory_save enqueues knowledge_id here; reflection.agent drains it, calls
-- representations.generate_representations() (LLM: summary/keywords/questions),
-- embeds each with the same embedder used for raw, and stores in
-- knowledge_representations (migration 002).

CREATE TABLE IF NOT EXISTS representations_queue (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id   INTEGER NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    attempts       INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT,
    created_at     TEXT NOT NULL,
    claimed_at     TEXT,
    processed_at   TEXT,
    UNIQUE(knowledge_id, status)
);

CREATE INDEX IF NOT EXISTS idx_repq_status ON representations_queue(status);
CREATE INDEX IF NOT EXISTS idx_repq_created ON representations_queue(created_at);
