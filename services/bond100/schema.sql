-- Bond 100 Hall — SQLite schema (bridge model).
--
-- arona.icu is the single source of truth. The DB is just a small cache:
--   * bond100_meta        — the cached wall ('wall_summary' + 'entries') + 'snapshot_date'
--   * bond100_refresh_log — abuse-limit bookkeeping for the submission /refresh flow


-- Cached wall. Bridge model holds 'wall_summary' + 'entries' (the aggregation
-- of arona's user-info endpoint) and 'snapshot_date', surfaced by /summary and
-- /students/<id>/entries.
CREATE TABLE IF NOT EXISTS bond100_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);


-- Abuse limiting for the submission ("add me") flow, which triggers an arona
-- /refresh. code_hash = sha256(server|friend_code) — the raw friend code is
-- never stored. Used for the per-code cooldown + the global hourly cap.
CREATE TABLE IF NOT EXISTS bond100_refresh_log (
  code_hash    TEXT PRIMARY KEY,
  server       TEXT,
  refreshed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_refresh_at ON bond100_refresh_log (refreshed_at);
