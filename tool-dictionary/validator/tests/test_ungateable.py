"""§2 regression: a gateway HTTP 400 on forced tool-use must mark the result
UNGATEABLE — never silently fall back to an ambiguous no-pick, never let an
un-runnable gate fabricate a LOCAL_GATE_PASS or a `_pending/` TOML write.

This is the test gap the adversary identified: the prior `ScriptedChat` /
`MockChat` mocks always returned a valid 200 tool-call, so the §2 defect
(forced-400 -> unforced fallback -> `ToolPick(picked=None, error=None)` ->
fabricated ~0.4 accuracy -> LOCAL_GATE_PASS -> `_pending` write) passed
self-verification. These tests drive a real 400 and assert the harness
REFUSES rather than fabricates.

SPEC v2 §2 (fail-safe under gateway error) / §13 (K-of-N).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from umb_validator.config import Config
from umb_validator.integration.gateway import ModelInfo, SessionPool
from umb_validator.integration.llm import LLMClient, ToolPick
from umb_validator.integration.umb_dev import CanonicalTool
from umb_validator.pipeline import (
    PendingWriter, PipelineDeps, ToolContext, ToolPipeline, run_tool_pipeline,
)
from umb_validator.states import State
from umb_validator.store import StateStore
from umb_validator.subsystems.benchmark import (
    BenchmarkRunner, ModelUngateable, OraclePrompt,
)

from test_pipeline import _CANONICAL_DESC, StubResearch, _universe


class _BadRequest(Exception):
    """An `openai.BadRequestError`-shaped exception.

    The real SDK raises `BadRequestError` (an `APIStatusError` subclass) with
    `status_code == 400` when a backend rejects `tool_choice="required"`.
    `_is_forced_tool_use_rejection` duck-types on `status_code`, so this stand
    -in faithfully reproduces the production 400 without importing `openai`.
    """

    def __init__(self, msg: str = "tool_choice not supported") -> None:
        super().__init__(msg)
        self.status_code = 400


class ForcedToolUse400Chat:
    """A chat_fn that 400s on EVERY forced tool-use call.

    Models the homelab backends as they are today: started without
    `--enable-auto-tool-choice` / `--tool-call-parser`, so any
    `tool_choice="required"` request is rejected. Free-form completions
    (no `tool_choice`) still succeed — that is what makes the original bug
    so dangerous: the unforced fallback "worked", just uselessly.
    """

    def __init__(self, short_desc: str = "Read a file's contents") -> None:
        self.short_desc = short_desc
        self.forced_calls = 0
        self.unforced_calls = 0

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if "tool_choice" in kwargs:
            # Forced tool-use — the backend rejects it.
            self.forced_calls += 1
            raise _BadRequest()
        if kwargs.get("tools"):
            # An UNFORCED call that still carries a tool list. If the harness
            # ever reaches here for select_tool it has wrongly fallen back —
            # the test below proves it does not.
            self.unforced_calls += 1
            return {
                "choices": [{"message": {"content": "", "tool_calls": []}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 4},
            }
        # Free-form completion (generation / shortener) — still works.
        user = kwargs["messages"][-1]["content"]
        if "SHORT description" in user:
            text = self.short_desc
        elif "objects" in user and "correct_tool" in user:
            text = json.dumps([
                {"prompt": f"neg prompt {i}", "correct_tool": "other_tool"}
                for i in range(30)])
        else:
            text = json.dumps([f"positive prompt {i}" for i in range(45)])
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 400},
        }


class MixedForcedToolUseChat(ForcedToolUse400Chat):
    """Some models support forced tool-use, some 400 (SPEC §13 mixed case).

    `ok_models` accept `tool_choice="required"` and return a valid pick;
    every other model 400s. Used to prove the K-of-roster floor: a pass
    cannot be fabricated from an under-strength jury of survivors.
    """

    def __init__(self, candidate: str, ok_models: set[str],
                 short_desc: str = "Read a file's contents") -> None:
        super().__init__(short_desc)
        self.candidate = candidate
        self.ok_models = ok_models

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if "tool_choice" in kwargs:
            self.forced_calls += 1
            if kwargs["model"] not in self.ok_models:
                raise _BadRequest()
            # A gateable model: pick candidate for positives, a distractor
            # for negatives (faithful), matching ScriptedChat semantics.
            user = kwargs["messages"][-1]["content"]
            if "neg" in user:
                pick = next(
                    (t["function"]["name"] for t in kwargs["tools"]
                     if t["function"]["name"] != self.candidate), None)
            else:
                pick = self.candidate
            tool_calls = ([{"function": {"name": pick}}]
                          if pick is not None else [])
            return {
                "choices": [{"message": {
                    "content": "", "tool_calls": tool_calls}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 4},
            }
        return await super().__call__(**kwargs)


def _roster() -> list[ModelInfo]:
    return [ModelInfo("qwen-35b", "qwen-35b", "backend-a"),
            ModelInfo("glm", "glm", "backend-c"),
            ModelInfo("qwen-4b", "qwen-4b", "backend-b")]


def _deps_with_chat(store: StateStore, tmp_path: Path, candidate: str,
                     chat: Any) -> PipelineDeps:
    """Pipeline deps wired with an arbitrary chat mock (parallels test_pipeline
    `_deps` but lets the §2 tests inject a 400-raising chat)."""
    from umb_validator.integration.gateway import GatewayClient
    cfg = Config()
    cfg.paths.pending_dir = str(tmp_path / "_pending")
    cfg.paths.tool_dictionary_dir = str(tmp_path / "dict")
    cfg.oracle.gen_positive = 30
    cfg.oracle.gen_negative = 20
    cfg.oracle.oracle_min_positive = 10
    cfg.oracle.oracle_min_negative = 6
    pool = SessionPool(cfg)
    llm = LLMClient(pool, chat, max_retries=0)
    roster = _roster()
    return PipelineDeps(
        config=cfg, store=store, gateway=GatewayClient(cfg), pool=pool,
        llm=llm, research=StubResearch(), universe=_universe(candidate),
        roster=roster, jury_models=roster,
        generator=ModelInfo("generator", "generator", "backend-a"))


def _seed_oracle(ctx: ToolContext, candidate: str) -> None:
    """Pre-seed an admitted oracle + baseline so a test can drive the pipeline
    straight into the gate stage (the §2-defective stage)."""
    ctx.oracle = [OraclePrompt(i, f"positive prompt {i}", "positive",
                               candidate) for i in range(12)] + \
                 [OraclePrompt(100 + i, f"neg prompt {i}", "negative",
                               candidate) for i in range(8)]
    ctx.baseline_acc = {"qwen-35b": 0.9, "glm": 0.9, "qwen-4b": 0.9}


# --- (a) select_tool marks ungateable, never an ambiguous no-pick ----------


@pytest.mark.asyncio
async def test_select_tool_forced_400_returns_ungateable_signal() -> None:
    """A forced-tool-use 400 yields `forced_unavailable=True` — DISTINCT from
    a legitimate `picked=None` no-pick — and does NOT fall back to unforced."""
    chat = ForcedToolUse400Chat()
    llm = LLMClient(SessionPool(Config()), chat, max_retries=0)
    model = ModelInfo("qwen-35b", "qwen-35b", "backend-a")

    pick = await llm.select_tool(
        model, [{"name": "read_file", "description": "Read a file"}],
        "Read foo.txt")

    assert isinstance(pick, ToolPick)
    assert pick.forced_unavailable is True
    assert pick.gradeable is False
    assert pick.error == "tool_choice_unsupported"
    # The forced call was attempted; the harness did NOT fall back to an
    # unforced completion (which would have produced an ambiguous no-pick).
    assert chat.forced_calls == 1
    assert chat.unforced_calls == 0


@pytest.mark.asyncio
async def test_genuine_no_pick_is_not_ungateable() -> None:
    """A real model no-pick (200 OK, empty tool_calls) is gradeable — it must
    NOT be confused with the §2 ungateable signal."""
    class NoPickChat:
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": "", "tool_calls": []}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }

    llm = LLMClient(SessionPool(Config()), NoPickChat(), max_retries=0)
    pick = await llm.select_tool(
        ModelInfo("m", "m", "backend-a"),
        [{"name": "read_file", "description": "x"}], "irrelevant prompt")
    assert pick.picked is None
    assert pick.forced_unavailable is False
    assert pick.error is None
    assert pick.gradeable is True


# --- benchmark excludes ungateable models from N ---------------------------


@pytest.mark.asyncio
async def test_benchmark_run_model_raises_on_ungateable() -> None:
    """`run_model` aborts (raises `ModelUngateable`) when forced tool-use is
    unavailable — it must NOT score a fabricated accuracy."""
    candidate = "read_file"
    chat = ForcedToolUse400Chat()
    llm = LLMClient(SessionPool(Config()), chat, max_retries=0)
    runner = BenchmarkRunner(llm)
    oracle = [OraclePrompt(0, "positive p", "positive", candidate)]
    universe = _universe(candidate)
    with pytest.raises(ModelUngateable):
        await runner.run_model(
            ModelInfo("qwen-35b", "qwen-35b", "backend-a"),
            "filesystem", candidate, _CANONICAL_DESC, oracle, universe)


@pytest.mark.asyncio
async def test_benchmark_run_all_models_all_400_excludes_everyone() -> None:
    """When every model 400s, `run_all_models` returns an EMPTY result set —
    nothing is scored, so no fabricated accuracy reaches the gate."""
    candidate = "read_file"
    chat = ForcedToolUse400Chat()
    llm = LLMClient(SessionPool(Config()), chat, max_retries=0)
    runner = BenchmarkRunner(llm)
    oracle = [OraclePrompt(0, "positive p", "positive", candidate)]
    results = await runner.run_all_models(
        _roster(), "filesystem", candidate, _CANONICAL_DESC, oracle,
        _universe(candidate))
    assert results == []


# --- (b)+(c)+(d) full pipeline: all-400 -> NEEDS_MANUAL, NO _pending TOML ---


@pytest.mark.asyncio
async def test_all_400_pipeline_terminates_needs_manual_no_pending(
    store: StateStore, tmp_path: Path,
) -> None:
    """THE §2 REGRESSION. With every backend 400ing on forced tool-use, a tool
    driven into the gate stage MUST terminate at NEEDS_MANUAL with reason
    `gateway_ungateable` and MUST NOT write a `_pending/<server>.toml`.

    Pre-bug: forced-400 -> unforced fallback -> ambiguous no-pick -> ~0.4
    fabricated accuracy -> LOCAL_GATE_PASS -> REVIEW_READY -> a `_pending`
    TOML of a gate that never actually ran.
    """
    candidate = "read_file"
    chat = ForcedToolUse400Chat()
    deps = _deps_with_chat(store, tmp_path, candidate, chat)
    ctx = ToolContext(server="filesystem", tool=candidate,
                      canonical=CanonicalTool(candidate, _CANONICAL_DESC, {},
                                              "filesystem"))
    _seed_oracle(ctx, candidate)
    store.upsert_tool("filesystem", candidate, status=State.BASELINE_RUN)
    store.record_event("filesystem", candidate, State.BASELINE_RUN)
    writer = PendingWriter(deps.config, store)
    pipeline = ToolPipeline(deps, "run-400", with_cloud=False)

    final = await run_tool_pipeline(pipeline, ctx, writer, State.BASELINE_RUN)

    # (b) terminal state is NEEDS_MANUAL — never LOCAL_GATE_PASS / REVIEW_READY.
    assert final == State.NEEDS_MANUAL
    assert store.current_state("filesystem", candidate) == State.NEEDS_MANUAL
    events = [e["new_status"] for e in store.events_for("filesystem", candidate)]
    assert "LOCAL_GATE_PASS" not in events
    assert "REVIEW_READY" not in events
    # The terminal event carries the explicit ungateable reason.
    last = store.events_for("filesystem", candidate)[-1]
    assert json.loads(last["metadata_json"])["reason"] == "gateway_ungateable"

    # (c) NO `_pending/` TOML was written — not for this tool, not at all.
    pending = Path(deps.config.resolve_pending_dir())
    assert not (pending / "filesystem.toml").exists()
    assert not pending.exists() or list(pending.glob("*.toml")) == []
    # (d) no fabricated accuracy: no pending_diff row recorded either.
    assert store.latest_pending_diff("filesystem", candidate) is None


@pytest.mark.asyncio
async def test_all_400_from_pending_never_writes_pending(
    store: StateStore, tmp_path: Path,
) -> None:
    """End-to-end from PENDING: even when self-validation is reached first,
    an all-400 fleet still terminates NEEDS_MANUAL and writes NO `_pending`
    TOML. Proves there is no path from a 400 fleet to a dictionary entry."""
    candidate = "read_file"
    chat = ForcedToolUse400Chat()
    deps = _deps_with_chat(store, tmp_path, candidate, chat)
    ctx = ToolContext(server="filesystem", tool=candidate,
                      canonical=CanonicalTool(candidate, _CANONICAL_DESC, {},
                                              "filesystem"),
                      upstream_source="https://github.com/x/y")
    store.upsert_tool("filesystem", candidate, status=State.PENDING)
    writer = PendingWriter(deps.config, store)
    pipeline = ToolPipeline(deps, "run-400-e2e", with_cloud=False)

    final = await run_tool_pipeline(pipeline, ctx, writer, State.PENDING)

    assert final == State.NEEDS_MANUAL
    pending = Path(deps.config.resolve_pending_dir())
    assert not (pending / "filesystem.toml").exists()
    assert not pending.exists() or list(pending.glob("*.toml")) == []
    assert store.latest_pending_diff("filesystem", candidate) is None


@pytest.mark.asyncio
async def test_mixed_under_strength_jury_goes_needs_manual(
    store: StateStore, tmp_path: Path,
) -> None:
    """SPEC §13 mixed case: only 1 of 3 roster models supports forced
    tool-use. The K-of-roster floor (K=ceil(0.75*3)=3) is NOT met by a jury
    of 1 survivor -> NEEDS_MANUAL `gateway_ungateable`, NO `_pending` write.
    A pass must NOT be fabricated from an under-strength jury."""
    candidate = "read_file"
    # Only qwen-35b accepts forced tool-use; glm + qwen-4b 400.
    chat = MixedForcedToolUseChat(candidate, ok_models={"qwen-35b"})
    deps = _deps_with_chat(store, tmp_path, candidate, chat)
    ctx = ToolContext(server="filesystem", tool=candidate,
                      canonical=CanonicalTool(candidate, _CANONICAL_DESC, {},
                                              "filesystem"))
    _seed_oracle(ctx, candidate)
    store.upsert_tool("filesystem", candidate, status=State.BASELINE_RUN)
    store.record_event("filesystem", candidate, State.BASELINE_RUN)
    writer = PendingWriter(deps.config, store)
    pipeline = ToolPipeline(deps, "run-mixed", with_cloud=False)

    final = await run_tool_pipeline(pipeline, ctx, writer, State.BASELINE_RUN)

    assert final == State.NEEDS_MANUAL
    last = store.events_for("filesystem", candidate)[-1]
    meta = json.loads(last["metadata_json"])
    assert meta["reason"] == "gateway_ungateable"
    assert meta["n_gateable_models"] == 1
    assert meta["min_gateable_required"] == 3
    pending = Path(deps.config.resolve_pending_dir())
    assert not (pending / "filesystem.toml").exists()
    assert store.latest_pending_diff("filesystem", candidate) is None


@pytest.mark.asyncio
async def test_mixed_quorum_jury_supported_models_pass(
    store: StateStore, tmp_path: Path,
) -> None:
    """SPEC §13: the gate runs ONLY on models that genuinely support forced
    tool-use. When ENOUGH models support it (here all 3), the §2 fix is inert
    and a faithful run still reaches REVIEW_READY — the fix does not break the
    happy path. This is the mixed-case complement to the under-strength test.
    """
    candidate = "read_file"
    chat = MixedForcedToolUseChat(
        candidate, ok_models={"qwen-35b", "glm", "qwen-4b"})
    deps = _deps_with_chat(store, tmp_path, candidate, chat)
    ctx = ToolContext(server="filesystem", tool=candidate,
                      canonical=CanonicalTool(candidate, _CANONICAL_DESC, {},
                                              "filesystem"))
    _seed_oracle(ctx, candidate)
    store.upsert_tool("filesystem", candidate, status=State.BASELINE_RUN)
    store.record_event("filesystem", candidate, State.BASELINE_RUN)
    writer = PendingWriter(deps.config, store)
    pipeline = ToolPipeline(deps, "run-mixed-ok", with_cloud=False)

    final = await run_tool_pipeline(pipeline, ctx, writer, State.BASELINE_RUN)
    assert final == State.REVIEW_READY
    assert (Path(deps.config.resolve_pending_dir()) / "filesystem.toml").is_file()
