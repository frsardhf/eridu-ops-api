-- Durable anonymous feedback submitted from the Eridu Ops frontend.

CREATE TABLE IF NOT EXISTS feedback_submissions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  category   TEXT NOT NULL CHECK (category IN ('suggestion', 'bug_report', 'incorrect_data', 'other')),
  message    TEXT NOT NULL,
  page       TEXT NOT NULL,
  locale     TEXT NOT NULL CHECK (locale IN ('en', 'jp', 'kr')),
  status     TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'reviewed', 'archived')),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_status_created
  ON feedback_submissions (status, created_at DESC);
