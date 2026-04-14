-- Migration 007: Performance indexes for hot dashboard / queue queries
--
-- Driven by perf-engineer audit findings:
--   - api_graph_delta WHERE last_reinforced_at > ? OR created_at > ?
--     was a SCAN graph_edges + TEMP B-TREE ORDER BY (~300ms at 35k rows).
--   - claim_next on triple_extraction_queue benefits from covering index.
--   - knowledge_representations JOIN by (representation, knowledge_id) needs index.

CREATE INDEX IF NOT EXISTS idx_ge_lastref      ON graph_edges(last_reinforced_at);
CREATE INDEX IF NOT EXISTS idx_ge_created_at   ON graph_edges(created_at);
CREATE INDEX IF NOT EXISTS idx_gn_lastseen     ON graph_nodes(last_seen_at);

CREATE INDEX IF NOT EXISTS idx_krepr_repr_kid
    ON knowledge_representations(representation, knowledge_id);

CREATE INDEX IF NOT EXISTS idx_teq_status_id
    ON triple_extraction_queue(status, id);
CREATE INDEX IF NOT EXISTS idx_deq_status_id
    ON deep_enrichment_queue(status, id);
CREATE INDEX IF NOT EXISTS idx_repq_status_id
    ON representations_queue(status, id);
