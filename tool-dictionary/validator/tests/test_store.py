"""SQLite store CRUD + checkpoint/resume tests. SPEC §1 / §2."""

from __future__ import annotations

from pathlib import Path

from umb_validator.states import State
from umb_validator.store import StateStore


def test_upsert_and_get_tool(store: StateStore) -> None:
    store.upsert_tool("filesystem", "read_file", status=State.PENDING)
    row = store.get_tool("filesystem", "read_file")
    assert row is not None
    assert row["server_name"] == "filesystem"
    assert row["status"] == "PENDING"


def test_upsert_tool_idempotent_update(store: StateStore) -> None:
    store.upsert_tool("github", "create_issue", status=State.PENDING)
    store.upsert_tool("github", "create_issue", current_hash="abc123")
    row = store.get_tool("github", "create_issue")
    assert row["current_hash"] == "abc123"
    # status preserved via COALESCE.
    assert row["status"] == "PENDING"


def test_state_event_is_current_state(store: StateStore) -> None:
    store.upsert_tool("time", "get_current_time", status=State.PENDING)
    store.record_event("time", "get_current_time", State.CANONICAL_FETCHED)
    store.record_event("time", "get_current_time", State.HASH_COMPUTED)
    assert store.current_state("time", "get_current_time") == State.HASH_COMPUTED


def test_events_for_is_ordered(store: StateStore) -> None:
    store.upsert_tool("s", "t", status=State.PENDING)
    store.record_event("s", "t", State.CANONICAL_FETCHED)
    store.record_event("s", "t", State.RESEARCHED)
    events = store.events_for("s", "t")
    assert [e["new_status"] for e in events] == ["CANONICAL_FETCHED",
                                                 "RESEARCHED"]


def test_tools_in_state(store: StateStore) -> None:
    store.upsert_tool("a", "x", status=State.PENDING)
    store.upsert_tool("b", "y", status=State.PENDING)
    store.record_event("a", "x", State.MERGED)
    pending = store.tools_in_state(State.PENDING)
    merged = store.tools_in_state(State.MERGED)
    assert [(t["server_name"], t["tool_name"]) for t in pending] == [("b", "y")]
    assert [(t["server_name"], t["tool_name"]) for t in merged] == [("a", "x")]


def test_resume_after_reopen(tmp_path: Path) -> None:
    """Checkpoint/resume: state survives a close + reopen (SPEC §1).

    Simulates a SIGTERM (close) + restart (reopen) — the tool's state must be
    recovered exactly, with no double-work needed.
    """
    db = tmp_path / "state.sqlite"
    s1 = StateStore(db)
    s1.upsert_tool("filesystem", "read_file", status=State.PENDING)
    s1.record_event("filesystem", "read_file", State.CANONICAL_FETCHED)
    s1.record_event("filesystem", "read_file", State.RESEARCHED)
    s1.close()

    s2 = StateStore(db)
    assert s2.current_state("filesystem", "read_file") == State.RESEARCHED
    # History is intact too.
    assert len(s2.events_for("filesystem", "read_file")) == 2
    s2.close()


def test_llm_call_idempotency_stamp(store: StateStore) -> None:
    """Stamping the same (tool, model, idx, mode, iter, purpose) twice returns
    the SAME row id — the dedupe key prevents double-storage on resume."""
    id1 = store.stamp_llm_call("benchmark", "qwen", "local", "prompt-A",
                               tool_key="fs/read", prompt_idx=3,
                               iteration=0, mode="baseline")
    id2 = store.stamp_llm_call("benchmark", "qwen", "local", "prompt-A",
                               tool_key="fs/read", prompt_idx=3,
                               iteration=0, mode="baseline")
    assert id1 == id2


def test_incomplete_llm_calls_redispatch(store: StateStore) -> None:
    """A stamped call with no response is a re-dispatch candidate on resume;
    once completed it drops off the incomplete list."""
    cid = store.stamp_llm_call("benchmark", "qwen", "local", "p",
                               tool_key="fs/read", prompt_idx=1)
    assert any(r["id"] == cid for r in store.incomplete_llm_calls())
    store.complete_llm_call(cid, response_raw='{"ok":true}')
    assert not any(r["id"] == cid for r in store.incomplete_llm_calls())
    assert store.llm_call_response(cid) == '{"ok":true}'


def test_prompt_crud_and_admission(store: StateStore) -> None:
    pid = store.add_prompt("fs", "read", 0, "Read foo.txt", "positive",
                           "auto_research", "qwen")
    # Idempotent on (server, tool, prompt_idx).
    pid2 = store.add_prompt("fs", "read", 0, "DIFFERENT", "positive",
                            "auto_research", "qwen")
    assert pid == pid2
    store.set_prompt_admitted(pid, True)
    admitted = store.prompts_for("fs", "read", only_admitted=True)
    assert len(admitted) == 1 and admitted[0]["admitted"] == 1


def test_validation_run_roundtrip(store: StateStore) -> None:
    rid = store.add_validation_run(
        run_id="run-1", server_name="fs", tool_name="read", iteration=0,
        description="full desc", description_hash="h", model="qwen",
        model_class="local", n_prompts=40, n_positive=25, n_negative=15,
        n_correct_pos=23, n_correct_neg=14, accuracy=0.87, token_count=120)
    assert rid > 0
    runs = store.runs_for("fs", "read", iteration=0)
    assert len(runs) == 1 and runs[0]["accuracy"] == 0.87


def test_pending_diff_and_review(store: StateStore) -> None:
    did = store.add_pending_diff("fs", "read", "Read a file", "hash",
                                 "/tmp/_pending/fs.toml")
    assert len(store.pending_diffs(unreviewed_only=True)) == 1
    store.review_pending_diff(did, "merged")
    assert len(store.pending_diffs(unreviewed_only=True)) == 0


def test_cloud_cost_ledger(store: StateStore) -> None:
    assert store.cloud_cost_today() == 0.0
    store.add_cloud_cost(1.25)
    store.add_cloud_cost(0.75)
    assert abs(store.cloud_cost_today() - 2.0) < 1e-9


def test_run_lifecycle(store: StateStore) -> None:
    store.start_run("run-1", with_cloud=False, tools_total=30)
    assert store.active_run()["run_id"] == "run-1"
    store.finish_run("run-1", "complete", {"passed": 26})
    assert store.active_run() is None
    assert store.latest_run()["status"] == "complete"
