"""End-to-end pipeline + _pending write + merge promotion tests.

Drives a single tool through the full state machine with mocked LLM +
research, asserts a `_pending/<server>.toml` is written, then exercises the
`merge` promotion. SPEC §1 / §3 / §9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from umb_validator.config import Config
from umb_validator.integration.gateway import GatewayClient, ModelInfo, SessionPool
from umb_validator.integration.llm import LLMClient
from umb_validator.integration.umb_dev import CanonicalTool
from umb_validator.operations import merge as merge_op
from umb_validator.pipeline import (
    PendingWriter, PipelineDeps, ToolContext, ToolPipeline, run_tool_pipeline,
)
from umb_validator.prompt_structure import ToolUniverse
from umb_validator.states import State
from umb_validator.store import StateStore
from umb_validator.subsystems.research import Artifact, ResearchClient

_CANONICAL_DESC = (
    "Read the complete contents of a file from the file system. Handles "
    "various text encodings and provides detailed error messages if the file "
    "cannot be read. Use this tool when you need to examine the contents of a "
    "single file. Only works within allowed directories. This description is "
    "deliberately long so a >50% token reduction is achievable by the "
    "shortener subsystem during the local gate."
)


class ScriptedChat:
    """A mock chat_fn scripted by call purpose.

    - tool-call requests (have `tools=`): always pick the candidate tool, so
      self-validation admits positives and the benchmark scores high.
    - completion requests: route by content — generation returns a JSON array
      of prompts; the shortener returns a terse description.
    """

    def __init__(self, candidate: str, short_desc: str):
        self.candidate = candidate
        self.short_desc = short_desc

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("tools"):
            # Forced tool-use. The scripted prompts are tagged "positive" /
            # "neg" in their text, so the mock jury/benchmark can pick the
            # candidate for positives and a distractor for negatives — i.e. a
            # perfectly-faithful set of jurors. Distractor name is read off
            # the presented tool window so it is always a valid pick.
            user = kwargs["messages"][-1]["content"]
            if "neg" in user:
                other = next(
                    (t["function"]["name"] for t in kwargs["tools"]
                     if t["function"]["name"] != self.candidate),
                    None)
                pick = other
            else:
                pick = self.candidate
            tool_calls = ([{"function": {"name": pick}}]
                          if pick is not None else [])
            return {
                "choices": [{"message": {
                    "content": "", "tool_calls": tool_calls}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 4},
            }
        # Completion: inspect the user prompt.
        user = kwargs["messages"][-1]["content"]
        if "SHORT description" in user:
            text = self.short_desc
        elif "objects" in user and "correct_tool" in user:
            # negative generation
            text = json.dumps([
                {"prompt": f"neg prompt {i}", "correct_tool": "other_tool"}
                for i in range(30)])
        else:
            # positive generation
            text = json.dumps([f"positive prompt {i}" for i in range(45)])
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 400},
        }


class StubResearch(ResearchClient):
    """A research client that returns canned grounding without network."""

    def __init__(self) -> None:  # noqa: D107 — deliberately bypasses super
        pass

    async def gather(  # type: ignore[override]
        self, tool: CanonicalTool, upstream_source: str | None,
    ) -> tuple[list[Artifact], str]:
        return ([Artifact(kind="canonical_schema", content="{...}"),
                 Artifact(kind="upstream_readme", content="Reads files.",
                          source_url="x", source_pinned="abc")], "full")


def _universe(candidate: str) -> ToolUniverse:
    by_server = {
        "filesystem": [
            CanonicalTool(candidate, _CANONICAL_DESC, {}, "filesystem"),
            CanonicalTool("write_file", "Write a file", {}, "filesystem"),
        ],
        "github": [CanonicalTool("create_issue", "Create issue", {},
                                 "github")],
        "time": [CanonicalTool("get_time", "Get the time", {}, "time")],
        "slack": [CanonicalTool("post_message", "Post msg", {}, "slack")],
        "memory": [CanonicalTool("create_entities", "Create", {}, "memory")],
        "sqlite": [CanonicalTool("query", "Query db", {}, "sqlite")],
        "gitlab": [CanonicalTool("list_projects", "List", {}, "gitlab")],
        "fetch": [CanonicalTool("fetch_url", "Fetch", {}, "fetch")],
    }
    return ToolUniverse.from_canonical(by_server)


def _deps(store: StateStore, tmp_path: Path, candidate: str) -> PipelineDeps:
    cfg = Config()
    cfg.paths.pending_dir = str(tmp_path / "_pending")
    cfg.paths.tool_dictionary_dir = str(tmp_path / "dict")
    # Small oracle minimums so the mocked run admits enough prompts fast.
    cfg.oracle.gen_positive = 30
    cfg.oracle.gen_negative = 20
    cfg.oracle.oracle_min_positive = 10
    cfg.oracle.oracle_min_negative = 6
    pool = SessionPool(cfg)
    chat = ScriptedChat(candidate, "Read a file's contents")
    llm = LLMClient(pool, chat, max_retries=0)
    roster = [ModelInfo("qwen-35b", "qwen-35b", "backend-a"),
              ModelInfo("glm", "glm", "backend-c"),
              ModelInfo("qwen-4b", "qwen-4b", "backend-b")]
    jury = roster
    generator = ModelInfo("generator", "generator", "backend-a")
    return PipelineDeps(
        config=cfg, store=store, gateway=GatewayClient(cfg), pool=pool,
        llm=llm, research=StubResearch(), universe=_universe(candidate),
        roster=roster, jury_models=jury, generator=generator)


@pytest.mark.asyncio
async def test_full_pipeline_reaches_review_ready(
    store: StateStore, tmp_path: Path,
) -> None:
    """A tool driven end-to-end (mocked) reaches REVIEW_READY and writes a
    `_pending/<server>.toml` proposal — never committing, never branching."""
    candidate = "read_file"
    deps = _deps(store, tmp_path, candidate)
    ct = deps.universe.get("filesystem", candidate)
    ctx = ToolContext(server="filesystem", tool=candidate,
                      canonical=CanonicalTool(candidate, _CANONICAL_DESC, {},
                                              "filesystem"),
                      upstream_source="https://github.com/x/y")
    store.upsert_tool("filesystem", candidate, status=State.PENDING)
    writer = PendingWriter(deps.config, store)
    pipeline = ToolPipeline(deps, "run-test", with_cloud=False)

    final = await run_tool_pipeline(pipeline, ctx, writer, State.PENDING)
    assert final == State.REVIEW_READY

    pending = Path(deps.config.resolve_pending_dir()) / "filesystem.toml"
    assert pending.is_file()
    body = pending.read_text(encoding="utf-8")
    assert "read_file" in body
    assert "Read a file's contents" in body
    # A pending_diff row was recorded.
    diff = store.latest_pending_diff("filesystem", candidate)
    assert diff is not None and diff["reviewed_at"] is None


@pytest.mark.asyncio
async def test_pipeline_resume_skips_completed_steps(
    store: StateStore, tmp_path: Path,
) -> None:
    """Resuming from a mid-pipeline state must not re-run earlier steps.

    Start the tool at BASELINE_RUN with a pre-seeded oracle: the run should
    pick up at the shortener, not re-research or re-self-validate.
    """
    candidate = "read_file"
    deps = _deps(store, tmp_path, candidate)
    ctx = ToolContext(server="filesystem", tool=candidate,
                      canonical=CanonicalTool(candidate, _CANONICAL_DESC, {},
                                              "filesystem"))
    # Pre-seed an oracle (as if self-validation already ran).
    from umb_validator.subsystems.benchmark import OraclePrompt
    # Prompt text is tagged "positive"/"neg" so the ScriptedChat mock jury
    # picks the candidate for positives and a distractor for negatives.
    ctx.oracle = [OraclePrompt(i, f"positive prompt {i}", "positive",
                               candidate) for i in range(12)] + \
                 [OraclePrompt(100 + i, f"neg prompt {i}", "negative",
                               candidate) for i in range(8)]
    ctx.baseline_acc = {"qwen-35b": 0.9, "glm": 0.9, "qwen-4b": 0.9}
    store.upsert_tool("filesystem", candidate, status=State.BASELINE_RUN)
    store.record_event("filesystem", candidate, State.BASELINE_RUN)
    writer = PendingWriter(deps.config, store)
    pipeline = ToolPipeline(deps, "run-resume", with_cloud=False)

    final = await run_tool_pipeline(pipeline, ctx, writer, State.BASELINE_RUN)
    assert final == State.REVIEW_READY
    # No RESEARCHED / PROMPTS_* events were appended on resume.
    statuses = [e["new_status"] for e in
                store.events_for("filesystem", candidate)]
    assert "RESEARCHED" not in statuses
    assert "PROMPTS_SELF_VALIDATED" not in statuses
    assert "REVIEW_READY" in statuses


@pytest.mark.asyncio
async def test_merge_promotes_pending_to_live(
    store: StateStore, tmp_path: Path,
) -> None:
    """`umb-validator merge` promotes `_pending/<server>.toml` into the live
    dict TOML, working-tree-dirty (no commit). SPEC §9."""
    candidate = "read_file"
    deps = _deps(store, tmp_path, candidate)
    ctx = ToolContext(server="filesystem", tool=candidate,
                      canonical=CanonicalTool(candidate, _CANONICAL_DESC, {},
                                              "filesystem"))
    store.upsert_tool("filesystem", candidate, status=State.PENDING)
    writer = PendingWriter(deps.config, store)
    pipeline = ToolPipeline(deps, "run-merge", with_cloud=False)
    final = await run_tool_pipeline(pipeline, ctx, writer, State.PENDING)
    assert final == State.REVIEW_READY

    result = merge_op(deps.config, store, "filesystem")
    assert result.error is None
    assert "read_file" in result.merged_tools
    live = Path(deps.config.resolve_dict_dir()) / "filesystem.toml"
    assert live.is_file()
    assert "Read a file's contents" in live.read_text(encoding="utf-8")
    # The tool is now MERGED; the _pending file is consumed.
    assert store.current_state("filesystem", candidate) == State.MERGED
    assert not (Path(deps.config.resolve_pending_dir())
                / "filesystem.toml").exists()
