"""Self-validation quorum tests with mocked model responses. SPEC §3.3."""

from __future__ import annotations

from typing import Any

import pytest

from umb_validator.config import Config
from umb_validator.integration.gateway import ModelInfo, SessionPool
from umb_validator.integration.llm import LLMClient
from umb_validator.subsystems.self_validation import (
    SelfValidator, juror_agrees, select_jury,
)


def _tool_call_response(tool_name: str | None) -> dict[str, Any]:
    """Build a minimal OpenAI-shaped chat response with a forced tool call."""
    tool_calls = (
        [{"function": {"name": tool_name}}] if tool_name is not None else []
    )
    return {
        "choices": [{"message": {"content": "", "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
    }


class MockChat:
    """A mock chat_fn: maps each model id to the tool it always picks."""

    def __init__(self, picks_by_model: dict[str, str | None]):
        self.picks = picks_by_model
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        model = kwargs["model"]
        return _tool_call_response(self.picks.get(model))


def _llm(picks: dict[str, str | None]) -> LLMClient:
    cfg = Config()
    pool = SessionPool(cfg)
    return LLMClient(pool, MockChat(picks), max_retries=0)


def _jury(ids: list[str]) -> list[ModelInfo]:
    return [ModelInfo(i, i, "backend-a") for i in ids]


# --- juror_agrees logic ----------------------------------------------------


def test_juror_agrees_positive() -> None:
    assert juror_agrees("positive", "read_file", "read_file")
    assert not juror_agrees("positive", "read_file", "write_file")
    assert not juror_agrees("positive", "read_file", None)


def test_juror_agrees_negative() -> None:
    """Negative: agreement means picking ANYTHING but the candidate tool."""
    assert juror_agrees("negative", "read_file", "write_file")
    assert juror_agrees("negative", "read_file", None)
    assert not juror_agrees("negative", "read_file", "read_file")


# --- jury selection --------------------------------------------------------


def test_select_jury_excludes_generator() -> None:
    """The generator model must not judge its own prompts (SPEC §3.3 rule 2)."""
    jury_cfg = _jury(["qwen-35b", "glm", "qwen-4b"])
    roster = _jury(["qwen-35b", "glm", "qwen-4b", "extra-model"])
    jury = select_jury(jury_cfg, "qwen-35b", roster)
    ids = {m.gateway_id for m in jury}
    assert "qwen-35b" not in ids
    # Backfilled from roster to keep jury size.
    assert len(jury) == 3
    assert "extra-model" in ids


def test_select_jury_no_generator_overlap() -> None:
    """Generator not in jury -> jury unchanged."""
    jury_cfg = _jury(["glm", "qwen-4b"])
    jury = select_jury(jury_cfg, "qwen-35b", _jury(["glm", "qwen-4b"]))
    assert {m.gateway_id for m in jury} == {"glm", "qwen-4b"}


# --- quorum admission ------------------------------------------------------


@pytest.mark.asyncio
async def test_positive_prompt_admitted_on_quorum() -> None:
    """3 of 4 jurors pick the intended tool -> admitted."""
    picks = {"j1": "read_file", "j2": "read_file", "j3": "read_file",
             "j4": "write_file"}
    sv = SelfValidator(_llm(picks), quorum_q=3)
    adm = await sv.validate_prompt(
        0, "Read the file foo.txt", "positive", "read_file",
        tool_window=[{"name": "read_file", "description": "Read a file"}],
        jury=_jury(["j1", "j2", "j3", "j4"]))
    assert adm.admitted
    assert adm.quorum.n_agree == 3


@pytest.mark.asyncio
async def test_positive_prompt_rejected_below_quorum() -> None:
    """Only 2 of 4 agree -> ambiguous, not admitted."""
    picks = {"j1": "read_file", "j2": "read_file", "j3": "write_file",
             "j4": "list_directory"}
    sv = SelfValidator(_llm(picks), quorum_q=3)
    adm = await sv.validate_prompt(
        0, "ambiguous prompt", "positive", "read_file",
        tool_window=[{"name": "read_file", "description": "x"}],
        jury=_jury(["j1", "j2", "j3", "j4"]))
    assert not adm.admitted
    assert adm.quorum.n_agree == 2


@pytest.mark.asyncio
async def test_negative_prompt_admitted_when_jury_avoids_candidate() -> None:
    """A negative prompt is admitted iff >= Q jurors pick something else."""
    picks = {"j1": "write_file", "j2": "list_directory", "j3": None,
             "j4": "read_file"}
    sv = SelfValidator(_llm(picks), quorum_q=3)
    adm = await sv.validate_prompt(
        0, "near-miss prompt", "negative", "read_file",
        tool_window=[{"name": "read_file", "description": "x"}],
        jury=_jury(["j1", "j2", "j3", "j4"]))
    # 3 jurors avoided read_file -> admitted.
    assert adm.admitted
    assert adm.quorum.n_agree == 3


@pytest.mark.asyncio
async def test_quorum_adapts_when_juror_errors() -> None:
    """A juror whose call errors is dropped from N (SPEC §5 flaky handling)."""

    class FlakyChat:
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["model"] == "flaky":
                raise RuntimeError("backend down")
            return _tool_call_response("read_file")

    cfg = Config()
    llm = LLMClient(SessionPool(cfg), FlakyChat(), max_retries=0)
    sv = SelfValidator(llm, quorum_q=3)
    adm = await sv.validate_prompt(
        0, "p", "positive", "read_file",
        tool_window=[{"name": "read_file", "description": "x"}],
        jury=_jury(["j1", "j2", "j3", "flaky"]))
    # flaky dropped -> N=3, all 3 agree -> still admitted.
    assert adm.quorum.n_jurors == 3
    assert adm.admitted
