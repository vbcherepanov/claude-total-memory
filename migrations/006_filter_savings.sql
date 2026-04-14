-- Migration 006: Filter savings metrics
--
-- When memory_save is called with a filter_name (e.g. "pytest", "cargo"),
-- the content_filter pipeline reduces noise before storage. We track the
-- savings here for operator visibility and aggregate metrics.

CREATE TABLE IF NOT EXISTS filter_savings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id   INTEGER NOT NULL,
    filter_name    TEXT NOT NULL,
    input_chars    INTEGER NOT NULL,
    output_chars   INTEGER NOT NULL,
    reduction_pct  REAL NOT NULL,
    safety         TEXT NOT NULL DEFAULT 'strict',
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fs_kid  ON filter_savings(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_fs_name ON filter_savings(filter_name);
CREATE INDEX IF NOT EXISTS idx_fs_at   ON filter_savings(created_at);
