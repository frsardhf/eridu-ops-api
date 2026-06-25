-- Bond 100 Hall — SQLite schema (bridge model).
--
-- arona.icu is the single source of truth. The DB is just a small cache:
--   * bond100_meta         — the assembled wall ('wall_summary' + 'entries') + 'snapshot_date'
--   * bond100_student_rank — per-student bond-100 store the wall is assembled from
--   * bond100_refresh_log  — abuse-limit bookkeeping for the submission /refresh flow


-- Assembled wall served by /summary and /students/<id>/entries. Holds the
-- 'wall_summary' + 'entries' blobs plus 'snapshot_date'. No longer written
-- directly by a single _info call: wall_store.assemble_wall() rebuilds these
-- blobs from bond100_student_rank, so the read path (repository.py) is unchanged.
CREATE TABLE IF NOT EXISTS bond100_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);


-- Per-student bond-100 store (rolling /rank model). One row per student holds
-- its current bond-100 entries from whatever source last fetched it: the 'info'
-- baseline (one rank_by_max_favor_user_info call for everyone) or a per-student
-- 'rank' sweep (friends/rank, real-time). assemble_wall() builds the bond100_meta
-- blobs from these rows; fetched_at drives per-student freshness on the wall and
-- the staleness-ordered sweep (stalest swept first).
CREATE TABLE IF NOT EXISTS bond100_student_rank (
  student_id INTEGER PRIMARY KEY,   -- primary (linked-collapsed) SchaleDB id
  count      INTEGER NOT NULL,      -- authoritative bond-100 count for the student
  entries    TEXT NOT NULL,         -- JSON: [{serverRegion, playerName}]
  source     TEXT NOT NULL,         -- 'info' | 'rank'
  fetched_at TEXT NOT NULL          -- ISO8601; stalest rows get swept first
);

CREATE INDEX IF NOT EXISTS idx_student_rank_fetched ON bond100_student_rank (fetched_at);


-- Abuse limiting for the submission ("add me") flow, which triggers an arona
-- /refresh. code_hash = sha256(server|friend_code) — the raw friend code is
-- never stored. Used for the per-code cooldown + the global hourly cap.
CREATE TABLE IF NOT EXISTS bond100_refresh_log (
  code_hash    TEXT PRIMARY KEY,
  server       TEXT,
  refreshed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_refresh_at ON bond100_refresh_log (refreshed_at);
