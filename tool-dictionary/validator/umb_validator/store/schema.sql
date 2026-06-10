-- umb-validator state store schema. SPEC v2 §2.
-- SQLite, WAL mode, synchronous=NORMAL. All tables idempotent (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS tools (
  server_name      TEXT NOT NULL,
  tool_name        TEXT NOT NULL,
  added_at         TEXT NOT NULL,
  status           TEXT NOT NULL,
  current_hash     TEXT,
  upstream_url     TEXT,
  upstream_pinned  TEXT,
  notes            TEXT,
  PRIMARY KEY (server_name, tool_name)
);

CREATE TABLE IF NOT EXISTS state_events (
  id             INTEGER PRIMARY KEY,
  server_name    TEXT NOT NULL,
  tool_name      TEXT NOT NULL,
  event_at       TEXT NOT NULL,
  new_status     TEXT NOT NULL,
  metadata_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_state_events_tool
  ON state_events (server_name, tool_name, event_at);

CREATE TABLE IF NOT EXISTS research_artifacts (
  id             INTEGER PRIMARY KEY,
  server_name    TEXT NOT NULL,
  tool_name      TEXT NOT NULL,
  kind           TEXT NOT NULL,
  source_url     TEXT,
  source_pinned  TEXT,
  content        TEXT NOT NULL,
  fetched_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_tool
  ON research_artifacts (server_name, tool_name);

CREATE TABLE IF NOT EXISTS benchmark_prompts (
  id              INTEGER PRIMARY KEY,
  server_name     TEXT NOT NULL,
  tool_name       TEXT NOT NULL,
  prompt_idx      INTEGER NOT NULL,
  prompt_text     TEXT NOT NULL,
  expected        TEXT NOT NULL,        -- 'positive' | 'negative'
  source          TEXT NOT NULL,        -- 'auto_research' | 'hand' | 'log_extracted'
  generator_model TEXT NOT NULL,
  admitted        INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL,
  UNIQUE (server_name, tool_name, prompt_idx)
);

CREATE TABLE IF NOT EXISTS prompt_self_validation (
  id           INTEGER PRIMARY KEY,
  prompt_id    INTEGER REFERENCES benchmark_prompts(id),
  juror_model  TEXT NOT NULL,
  juror_pick   TEXT,
  agrees       INTEGER NOT NULL,
  ran_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_psv_prompt
  ON prompt_self_validation (prompt_id);

CREATE TABLE IF NOT EXISTS validation_runs (
  id               INTEGER PRIMARY KEY,
  run_id           TEXT NOT NULL,        -- the umb-validator run id
  server_name      TEXT NOT NULL,
  tool_name        TEXT NOT NULL,
  iteration        INTEGER NOT NULL,     -- 0 = baseline; 1..N = shortened
  description      TEXT NOT NULL,
  description_hash TEXT NOT NULL,
  model            TEXT NOT NULL,
  model_class      TEXT NOT NULL,        -- 'local' | 'cloud'
  backend          TEXT,
  n_prompts        INTEGER NOT NULL,
  n_positive       INTEGER NOT NULL,
  n_negative       INTEGER NOT NULL,
  n_correct_pos    INTEGER NOT NULL,
  n_correct_neg    INTEGER NOT NULL,
  accuracy         REAL NOT NULL,
  token_count      INTEGER NOT NULL,
  p50_latency_ms   REAL,
  p95_latency_ms   REAL,
  cost_usd         REAL NOT NULL DEFAULT 0,
  ran_at           TEXT NOT NULL,
  UNIQUE (run_id, server_name, tool_name, iteration, model)
);
CREATE INDEX IF NOT EXISTS idx_runs_tool
  ON validation_runs (server_name, tool_name);

CREATE TABLE IF NOT EXISTS llm_calls (
  id           INTEGER PRIMARY KEY,
  run_id       INTEGER,
  purpose      TEXT NOT NULL,            -- research|generate|self_validate|shorten|benchmark
  tool_key     TEXT,                     -- 'server/tool' for idempotency dedupe
  prompt_idx   INTEGER,
  model        TEXT NOT NULL,
  model_class  TEXT NOT NULL,
  iteration    INTEGER NOT NULL DEFAULT 0,
  mode         TEXT NOT NULL DEFAULT '', -- baseline|shortened|jury|''
  prompt_in    TEXT NOT NULL,
  response_raw TEXT,
  tokens_in    INTEGER,
  tokens_out   INTEGER,
  cost_usd     REAL NOT NULL DEFAULT 0,
  started_at   TEXT NOT NULL,
  finished_at  TEXT,
  error        TEXT,
  UNIQUE (tool_key, model, prompt_idx, mode, iteration, purpose)
);

CREATE TABLE IF NOT EXISTS pending_diffs (
  id               INTEGER PRIMARY KEY,
  server_name      TEXT NOT NULL,
  tool_name        TEXT NOT NULL,
  proposed_short   TEXT NOT NULL,
  proposed_hash    TEXT NOT NULL,
  pending_path     TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  reviewed_at      TEXT,
  reviewed_outcome TEXT,                 -- 'merged' | 'rejected'
  reviewer         TEXT,
  reason           TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_tool
  ON pending_diffs (server_name, tool_name);

-- daily cloud-cost ledger for the MAX_DAILY_COST_USD cap (UTC date keyed).
CREATE TABLE IF NOT EXISTS cost_ledger (
  utc_date   TEXT PRIMARY KEY,
  cloud_usd  REAL NOT NULL DEFAULT 0
);

-- run metadata: lets `status` and resume find the active run.
CREATE TABLE IF NOT EXISTS runs (
  run_id         TEXT PRIMARY KEY,
  started_at     TEXT NOT NULL,
  finished_at    TEXT,
  status         TEXT NOT NULL,          -- running|complete|wall_cap|crashed
  with_cloud     INTEGER NOT NULL DEFAULT 0,
  tools_total    INTEGER NOT NULL DEFAULT 0,
  metadata_json  TEXT
);
