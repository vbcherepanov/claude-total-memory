-- v11.0 Phase W1-A — First-class Episode layer for LoCoMo multi-hop / temporal QA.
--
-- This migration adds a NEW episode model that sits on top of the flat
-- `knowledge` (fact) store. Each row captures a coherent (when, who,
-- where, what, why, outcome) unit so retrieval can return whole episodes
-- instead of scattered facts.
--
-- Naming note: the v5 schema (migration 001) already owns a table named
-- `episodes` for the Beever-style narrative episode store
-- (TEXT id, narrative, outcome enum, used by EpisodeStore + dashboard +
-- reflection + cognitive engine). The W1-A model is a different concept
-- with a different shape (INTEGER id, started_at/ended_at window,
-- participants, summary embedding). Stomping on the existing table would
-- break every consumer above, so this migration uses the suffix `_v11`
-- on the new table and its FTS mirror. They are designed to coexist.

CREATE TABLE IF NOT EXISTS episodes_v11 (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  project         TEXT NOT NULL,
  session_id      TEXT,
  started_at      TEXT NOT NULL,             -- ISO 8601, first fact timestamp
  ended_at        TEXT NOT NULL,             -- ISO 8601, last fact timestamp
  participants    TEXT,                      -- JSON array of canonical entity tags
  location        TEXT,                      -- optional, free-form
  summary         TEXT NOT NULL,             -- compact narrative (LLM or fallback)
  outcome         TEXT,                      -- optional, free-form
  embedding_blob  BLOB,                      -- struct.pack('f'*dim, *vec) of summary
  created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_v11_project_time
    ON episodes_v11(project, started_at);

CREATE INDEX IF NOT EXISTS idx_episodes_v11_session
    ON episodes_v11(session_id);

-- Idempotency anchor — segment boundaries are a deterministic function of
-- (project, session_id, started_at), so we use that triple as a re-run
-- guard. UNIQUE allows extractor to use INSERT OR IGNORE.
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_v11_unique
    ON episodes_v11(project, IFNULL(session_id, ''), started_at);

CREATE TABLE IF NOT EXISTS episode_facts (
  episode_id   INTEGER NOT NULL,
  knowledge_id INTEGER NOT NULL,
  PRIMARY KEY (episode_id, knowledge_id),
  FOREIGN KEY (episode_id) REFERENCES episodes_v11(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episode_facts_kid
    ON episode_facts(knowledge_id);

-- FTS5 mirror over the episode summary so the retriever can BM25-score
-- it. content='' (contentless FTS) keeps the index in sync via explicit
-- INSERT/DELETE in extractor.py rather than triggers; this avoids the
-- chicken-and-egg of triggers firing before the row's id is available.
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_v11_fts USING fts5(
    summary,
    participants,
    outcome,
    content=''
);

INSERT OR IGNORE INTO migrations (version, description)
VALUES ('023', 'Episode layer (W1-A) — first-class episodes_v11 + episode_facts + FTS mirror');
