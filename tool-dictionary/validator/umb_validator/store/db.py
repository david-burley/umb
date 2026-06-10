"""SQLite WAL state store. SPEC v2 §1/§2.

Current state of a tool = `MAX(event_at) WHERE server=… AND tool=…` over
`state_events` (append-only). Checkpoint/resume: every long-running step
writes a `state_event` row + flushes WAL BEFORE inference dispatch; the
`llm_calls` row is upserted by `(tool_key, model, prompt_idx, mode, iteration,
purpose)`. On restart, a stamped row with no response is re-dispatched.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from umb_validator.states import State

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _utcnow() -> str:
    """ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StateStore:
    """Durable per-tool state + run artifacts.

    Thread-/coroutine-safety: a single connection is used; SQLite serializes
    writes. The harness scheduler runs one DB-touching coroutine at a time for
    state mutations (asyncio is single-threaded), so explicit locking is not
    required, but `check_same_thread=False` is set for defensiveness.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        """Flush WAL + close. Safe to call multiple times."""
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------- tool corpus ----------------

    def upsert_tool(
        self,
        server: str,
        tool: str,
        *,
        status: State | None = None,
        current_hash: str | None = None,
        upstream_url: str | None = None,
        upstream_pinned: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Insert a tool row or update its mutable fields. The `status` column
        is a denormalized cache of the latest `state_events` row."""
        existing = self.get_tool(server, tool)
        if existing is None:
            self.conn.execute(
                "INSERT INTO tools (server_name, tool_name, added_at, status, "
                "current_hash, upstream_url, upstream_pinned, notes) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    server, tool, _utcnow(),
                    str(status or State.PENDING),
                    current_hash, upstream_url, upstream_pinned, notes,
                ),
            )
        else:
            self.conn.execute(
                "UPDATE tools SET status=COALESCE(?,status), "
                "current_hash=COALESCE(?,current_hash), "
                "upstream_url=COALESCE(?,upstream_url), "
                "upstream_pinned=COALESCE(?,upstream_pinned), "
                "notes=COALESCE(?,notes) "
                "WHERE server_name=? AND tool_name=?",
                (
                    str(status) if status else None,
                    current_hash, upstream_url, upstream_pinned, notes,
                    server, tool,
                ),
            )
        self.conn.commit()

    def get_tool(self, server: str, tool: str) -> sqlite3.Row | None:
        """Fetch one tool row, or None."""
        cur = self.conn.execute(
            "SELECT * FROM tools WHERE server_name=? AND tool_name=?",
            (server, tool),
        )
        row: sqlite3.Row | None = cur.fetchone()
        return row

    def all_tools(self) -> list[sqlite3.Row]:
        """All tool rows."""
        return list(
            self.conn.execute("SELECT * FROM tools ORDER BY server_name, tool_name")
        )

    def tools_in_state(self, state: State) -> list[sqlite3.Row]:
        """All tools whose CURRENT state is `state` (derived from events)."""
        return [t for t in self.all_tools() if self.current_state(
            t["server_name"], t["tool_name"]) == state]

    # ---------------- state machine ----------------

    def record_event(
        self,
        server: str,
        tool: str,
        new_status: State,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a state_event AND flush WAL. This is the idempotency stamp:
        the event is durable before any subsequent inference is dispatched."""
        self.conn.execute(
            "INSERT INTO state_events (server_name, tool_name, event_at, "
            "new_status, metadata_json) VALUES (?,?,?,?,?)",
            (server, tool, _utcnow(), str(new_status),
             json.dumps(metadata) if metadata else None),
        )
        # Keep the denormalized cache in sync.
        self.conn.execute(
            "UPDATE tools SET status=? WHERE server_name=? AND tool_name=?",
            (str(new_status), server, tool),
        )
        self.conn.commit()
        # Flush WAL so the stamp survives power loss before inference dispatch.
        self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def current_state(self, server: str, tool: str) -> State | None:
        """Current state = the latest state_events row for the tool. Falls
        back to the `tools.status` cache if no events exist yet."""
        cur = self.conn.execute(
            "SELECT new_status FROM state_events "
            "WHERE server_name=? AND tool_name=? "
            "ORDER BY event_at DESC, id DESC LIMIT 1",
            (server, tool),
        )
        row = cur.fetchone()
        if row is not None:
            return State(row["new_status"])
        t = self.get_tool(server, tool)
        return State(t["status"]) if t is not None else None

    def events_for(self, server: str, tool: str) -> list[sqlite3.Row]:
        """Full event history for a tool, oldest first."""
        return list(self.conn.execute(
            "SELECT * FROM state_events WHERE server_name=? AND tool_name=? "
            "ORDER BY event_at ASC, id ASC",
            (server, tool),
        ))

    # ---------------- research artifacts ----------------

    def add_research_artifact(
        self, server: str, tool: str, kind: str, content: str,
        source_url: str | None = None, source_pinned: str | None = None,
    ) -> int:
        """Store one grounding artifact; returns its row id."""
        cur = self.conn.execute(
            "INSERT INTO research_artifacts (server_name, tool_name, kind, "
            "source_url, source_pinned, content, fetched_at) VALUES (?,?,?,?,?,?,?)",
            (server, tool, kind, source_url, source_pinned, content, _utcnow()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def research_artifacts(self, server: str, tool: str) -> list[sqlite3.Row]:
        """All grounding artifacts for a tool."""
        return list(self.conn.execute(
            "SELECT * FROM research_artifacts WHERE server_name=? AND tool_name=? "
            "ORDER BY id ASC",
            (server, tool),
        ))

    # ---------------- benchmark prompts ----------------

    def add_prompt(
        self, server: str, tool: str, prompt_idx: int, text: str,
        expected: str, source: str, generator_model: str,
    ) -> int:
        """Insert a candidate prompt (admitted=0). Idempotent on
        (server, tool, prompt_idx) — re-insert returns the existing id."""
        existing = self.conn.execute(
            "SELECT id FROM benchmark_prompts "
            "WHERE server_name=? AND tool_name=? AND prompt_idx=?",
            (server, tool, prompt_idx),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        cur = self.conn.execute(
            "INSERT INTO benchmark_prompts (server_name, tool_name, prompt_idx, "
            "prompt_text, expected, source, generator_model, admitted, created_at) "
            "VALUES (?,?,?,?,?,?,?,0,?)",
            (server, tool, prompt_idx, text, expected, source,
             generator_model, _utcnow()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def set_prompt_admitted(self, prompt_id: int, admitted: bool) -> None:
        """Flip a prompt's admitted flag after self-validation quorum."""
        self.conn.execute(
            "UPDATE benchmark_prompts SET admitted=? WHERE id=?",
            (1 if admitted else 0, prompt_id),
        )
        self.conn.commit()

    def prompts_for(
        self, server: str, tool: str, only_admitted: bool = False,
    ) -> list[sqlite3.Row]:
        """Candidate prompts for a tool; `only_admitted` filters to the oracle."""
        q = ("SELECT * FROM benchmark_prompts WHERE server_name=? AND tool_name=?"
             + (" AND admitted=1" if only_admitted else "")
             + " ORDER BY prompt_idx ASC")
        return list(self.conn.execute(q, (server, tool)))

    def add_self_validation(
        self, prompt_id: int, juror_model: str, juror_pick: str | None,
        agrees: bool,
    ) -> None:
        """Record one juror verdict for a prompt."""
        self.conn.execute(
            "INSERT INTO prompt_self_validation (prompt_id, juror_model, "
            "juror_pick, agrees, ran_at) VALUES (?,?,?,?,?)",
            (prompt_id, juror_model, juror_pick, 1 if agrees else 0, _utcnow()),
        )
        self.conn.commit()

    def self_validations(self, prompt_id: int) -> list[sqlite3.Row]:
        """All juror verdicts for one prompt."""
        return list(self.conn.execute(
            "SELECT * FROM prompt_self_validation WHERE prompt_id=?",
            (prompt_id,),
        ))

    # ---------------- validation runs ----------------

    def add_validation_run(self, **fields: Any) -> int:
        """Insert (or replace, idempotent) a per-(tool,iteration,model) run row."""
        cols = (
            "run_id", "server_name", "tool_name", "iteration", "description",
            "description_hash", "model", "model_class", "backend", "n_prompts",
            "n_positive", "n_negative", "n_correct_pos", "n_correct_neg",
            "accuracy", "token_count", "p50_latency_ms", "p95_latency_ms",
            "cost_usd", "ran_at",
        )
        fields.setdefault("ran_at", _utcnow())
        fields.setdefault("cost_usd", 0.0)
        fields.setdefault("backend", None)
        fields.setdefault("p50_latency_ms", None)
        fields.setdefault("p95_latency_ms", None)
        placeholders = ",".join("?" for _ in cols)
        cur = self.conn.execute(
            f"INSERT OR REPLACE INTO validation_runs ({','.join(cols)}) "
            f"VALUES ({placeholders})",
            tuple(fields[c] for c in cols),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def runs_for(
        self, server: str, tool: str, iteration: int | None = None,
    ) -> list[sqlite3.Row]:
        """Validation runs for a tool, optionally filtered to one iteration."""
        if iteration is None:
            return list(self.conn.execute(
                "SELECT * FROM validation_runs WHERE server_name=? AND tool_name=? "
                "ORDER BY iteration, model",
                (server, tool),
            ))
        return list(self.conn.execute(
            "SELECT * FROM validation_runs WHERE server_name=? AND tool_name=? "
            "AND iteration=? ORDER BY model",
            (server, tool, iteration),
        ))

    # ---------------- llm call idempotency ----------------

    def stamp_llm_call(
        self, purpose: str, model: str, model_class: str, prompt_in: str,
        tool_key: str | None = None, prompt_idx: int | None = None,
        iteration: int = 0, mode: str = "", run_id: int | None = None,
    ) -> int:
        """Idempotency stamp: upsert an llm_calls row keyed by
        (tool_key, model, prompt_idx, mode, iteration, purpose). If a row
        already has a response, that id is returned and the caller skips
        dispatch. Returns the row id."""
        existing = self.conn.execute(
            "SELECT id, response_raw FROM llm_calls WHERE "
            "tool_key IS ? AND model=? AND prompt_idx IS ? AND mode=? "
            "AND iteration=? AND purpose=?",
            (tool_key, model, prompt_idx, mode, iteration, purpose),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        cur = self.conn.execute(
            "INSERT INTO llm_calls (run_id, purpose, tool_key, prompt_idx, "
            "model, model_class, iteration, mode, prompt_in, started_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, purpose, tool_key, prompt_idx, model, model_class,
             iteration, mode, prompt_in, _utcnow()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def llm_call_response(self, call_id: int) -> str | None:
        """Return a stamped call's stored response, or None if not yet done."""
        row = self.conn.execute(
            "SELECT response_raw FROM llm_calls WHERE id=?", (call_id,),
        ).fetchone()
        return row["response_raw"] if row else None

    def complete_llm_call(
        self, call_id: int, response_raw: str | None,
        tokens_in: int | None = None, tokens_out: int | None = None,
        cost_usd: float = 0.0, error: str | None = None,
    ) -> None:
        """Record an llm_calls response (or error) + finish timestamp."""
        self.conn.execute(
            "UPDATE llm_calls SET response_raw=?, tokens_in=?, tokens_out=?, "
            "cost_usd=?, error=?, finished_at=? WHERE id=?",
            (response_raw, tokens_in, tokens_out, cost_usd, error,
             _utcnow(), call_id),
        )
        self.conn.commit()

    def incomplete_llm_calls(self) -> list[sqlite3.Row]:
        """Stamped calls with no response — re-dispatch candidates on resume."""
        return list(self.conn.execute(
            "SELECT * FROM llm_calls WHERE response_raw IS NULL AND error IS NULL"
        ))

    # ---------------- pending diffs ----------------

    def add_pending_diff(
        self, server: str, tool: str, proposed_short: str,
        proposed_hash: str, pending_path: str,
    ) -> int:
        """Record a REVIEW_READY proposal pointing at a _pending TOML."""
        cur = self.conn.execute(
            "INSERT INTO pending_diffs (server_name, tool_name, proposed_short, "
            "proposed_hash, pending_path, created_at) VALUES (?,?,?,?,?,?)",
            (server, tool, proposed_short, proposed_hash, pending_path, _utcnow()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def pending_diffs(self, unreviewed_only: bool = True) -> list[sqlite3.Row]:
        """Pending diffs; by default only those awaiting owner review."""
        q = "SELECT * FROM pending_diffs"
        if unreviewed_only:
            q += " WHERE reviewed_at IS NULL"
        q += " ORDER BY created_at"
        return list(self.conn.execute(q))

    def latest_pending_diff(self, server: str, tool: str) -> sqlite3.Row | None:
        """Most recent pending diff for one tool."""
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM pending_diffs WHERE server_name=? AND tool_name=? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (server, tool),
        ).fetchone()
        return row

    def review_pending_diff(
        self, diff_id: int, outcome: str, reviewer: str = "owner",
        reason: str | None = None,
    ) -> None:
        """Mark a pending diff merged/rejected."""
        self.conn.execute(
            "UPDATE pending_diffs SET reviewed_at=?, reviewed_outcome=?, "
            "reviewer=?, reason=? WHERE id=?",
            (_utcnow(), outcome, reviewer, reason, diff_id),
        )
        self.conn.commit()

    # ---------------- cost ledger ----------------

    def add_cloud_cost(self, usd: float) -> float:
        """Add to today's (UTC) cloud cost ledger; returns the new daily total."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.conn.execute(
            "INSERT INTO cost_ledger (utc_date, cloud_usd) VALUES (?, ?) "
            "ON CONFLICT(utc_date) DO UPDATE SET cloud_usd = cloud_usd + ?",
            (today, usd, usd),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT cloud_usd FROM cost_ledger WHERE utc_date=?", (today,),
        ).fetchone()
        return float(row["cloud_usd"]) if row else 0.0

    def cloud_cost_today(self) -> float:
        """Today's (UTC) accumulated cloud cost."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT cloud_usd FROM cost_ledger WHERE utc_date=?", (today,),
        ).fetchone()
        return float(row["cloud_usd"]) if row else 0.0

    # ---------------- runs ----------------

    def start_run(
        self, run_id: str, with_cloud: bool, tools_total: int,
    ) -> None:
        """Record a new run as 'running'."""
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, status, "
            "with_cloud, tools_total) VALUES (?,?,?,?,?)",
            (run_id, _utcnow(), "running", 1 if with_cloud else 0, tools_total),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str,
                   metadata: dict[str, Any] | None = None) -> None:
        """Mark a run finished (complete|wall_cap|crashed)."""
        self.conn.execute(
            "UPDATE runs SET finished_at=?, status=?, metadata_json=? "
            "WHERE run_id=?",
            (_utcnow(), status, json.dumps(metadata) if metadata else None,
             run_id),
        )
        self.conn.commit()

    def active_run(self) -> sqlite3.Row | None:
        """The most recent run still in 'running' state, or None."""
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM runs WHERE status='running' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row

    def latest_run(self) -> sqlite3.Row | None:
        """The most recent run regardless of status."""
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row
