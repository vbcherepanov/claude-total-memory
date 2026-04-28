-- v11.0 Phase W2-G — Idle-project consolidation daemon state.
--
-- Two tables:
--
--   project_activity     — heartbeat for "this project is being touched
--                          right now". Updated by hot-path code (recall /
--                          save / session events). The daemon reads it to
--                          decide which projects are idle (>30 min) and
--                          must NEVER consolidate the project a user is
--                          currently working in.
--
--   consolidation_state  — per-project bookkeeping for the daemon:
--                          last run, status, advisory lock TTL, JSON stats.
--                          The lock is advisory (TTL field) rather than a
--                          BEGIN EXCLUSIVE because we run on SQLite WAL
--                          and need recall to keep reading freely.
--
-- Idempotent: every CREATE uses IF NOT EXISTS so re-running this migration
-- on a partly-applied DB is a no-op.

CREATE TABLE IF NOT EXISTS project_activity (
  project          TEXT PRIMARY KEY,
  last_touched_at  TEXT NOT NULL,
  touch_count_24h  INTEGER NOT NULL DEFAULT 0,
  updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_proj_activity_touched
    ON project_activity(last_touched_at);

CREATE TABLE IF NOT EXISTS consolidation_state (
  project              TEXT PRIMARY KEY,
  last_consolidated_at TEXT,
  last_status          TEXT,            -- 'ok' | 'failed' | 'paused' | 'in_progress'
  last_error           TEXT,
  locked_until         TEXT,            -- advisory lock TTL; reused by recall to detect ongoing consolidation
  stats_json           TEXT,            -- last run summary (JSON of ConsolidationStats)
  updated_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_consol_locked
    ON consolidation_state(locked_until);

INSERT OR IGNORE INTO migrations (version, description)
VALUES ('025', 'Consolidation daemon state (W2-G) — project_activity + consolidation_state');
