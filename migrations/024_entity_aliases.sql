-- v11.0 Phase W1-F — Global Entity Resolver.
--
-- Cross-session coreference: "Sarah", "she", "mom" (within a discourse
-- where context resolves "mom" → "Sarah") should all map to the same
-- canonical entity_id within a project. The session-local
-- `coref_resolver.py` only rewrites pronouns/deictics inside one save —
-- it has no memory across sessions and cannot tell the retriever that
-- two surface forms refer to the same person/thing.
--
-- This migration introduces a project-scoped alias index:
--
--   canonical_entities — one row per real-world entity (person,
--     technology, project, …). Identity = (project, type, name_norm).
--
--   entity_aliases     — many surface forms pointing at one canonical.
--     Includes the canonical's own name as alias (so a single lookup
--     against alias_norm covers exact + alias matches).
--
-- Idempotent — every CREATE uses IF NOT EXISTS, the unique constraint
-- on (project, type, name_norm) prevents duplicate canonicals on
-- replay, and entity_aliases is keyed by an autoincrement id so repeated
-- inserts are tolerated by the application layer (entity_resolver.add_alias
-- guards against duplicates before INSERT).

CREATE TABLE IF NOT EXISTS canonical_entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT NOT NULL,
    type        TEXT NOT NULL,           -- person, technology, project, ...
    name        TEXT NOT NULL,
    name_norm   TEXT NOT NULL,           -- lowercased, stripped, no punct, accents folded
    embedding   BLOB,                    -- struct.pack('f'*dim, *vec) of canonical name
    attrs       TEXT,                    -- JSON metadata (free-form)
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(project, type, name_norm)
);

CREATE INDEX IF NOT EXISTS idx_canon_proj_type
    ON canonical_entities(project, type);

CREATE TABLE IF NOT EXISTS entity_aliases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id  INTEGER NOT NULL,
    alias         TEXT NOT NULL,
    alias_norm    TEXT NOT NULL,
    source        TEXT,                  -- "explicit" | "coref" | "embedding_match"
    confidence    REAL DEFAULT 1.0,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (canonical_id) REFERENCES canonical_entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_aliases_norm  ON entity_aliases(alias_norm);
CREATE INDEX IF NOT EXISTS idx_aliases_canon ON entity_aliases(canonical_id);

INSERT OR IGNORE INTO migrations (version, description)
VALUES ('024', 'Global Entity Resolver (W1-F) — canonical_entities + entity_aliases');
