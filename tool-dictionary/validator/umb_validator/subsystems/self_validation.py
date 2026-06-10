"""Auto-research subsystem — Self-validation phase. SPEC v2 §3.3.

A generated prompt is admitted to the trusted benchmark ONLY if it passes a
cross-model quorum faithfulness check (DECISION #1: Q=3 of N=4).

The jury is N architecturally-diverse models. The generator model is EXCLUDED
from judging its own prompts. Each juror is given the full 8-tool window and
must pick a tool (forced tool-use). A positive prompt is admitted iff >= Q
jurors pick the intended tool; a negative iff >= Q jurors pick something else.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from umb_validator.gates import QuorumResult, evaluate_quorum
from umb_validator.integration.gateway import ModelInfo
from umb_validator.integration.llm import LLMClient
from umb_validator.logging_setup import get_logger

log = get_logger("self_validation")


@dataclass
class JurorVerdict:
    """One juror's verdict on one prompt."""

    juror_model: str
    picked: str | None
    agrees: bool  # picked matches the prompt's expected classification


@dataclass
class PromptAdmission:
    """Self-validation outcome for one prompt."""

    prompt_idx: int
    verdicts: list[JurorVerdict]
    quorum: QuorumResult

    @property
    def admitted(self) -> bool:
        return self.quorum.admitted


def juror_agrees(
    expected: str, intended_tool: str, picked: str | None
) -> bool:
    """Does a juror's pick agree with the prompt's expected classification?

    - positive: juror must pick `intended_tool`.
    - negative: juror must pick anything OTHER than `intended_tool`
      (a distractor, the intended-other tool, or no tool).
    """
    if expected == "positive":
        return picked == intended_tool
    return picked != intended_tool


def select_jury(
    jury_models: list[ModelInfo],
    generator_model: str | None,
    roster: list[ModelInfo],
) -> list[ModelInfo]:
    """Resolve the jury for a tool's prompts, EXCLUDING the generator model.

    If the generator is in the configured jury, it is swapped for the next
    roster model not already on the jury (SPEC §3.3 rule 2).
    """
    jury = [m for m in jury_models
            if generator_model is None or m.gateway_id != generator_model]
    if len(jury) < len(jury_models):
        # A generator was removed — backfill from the roster.
        used = {m.gateway_id for m in jury} | {generator_model}
        for cand in roster:
            if len(jury) >= len(jury_models):
                break
            if cand.gateway_id not in used:
                jury.append(cand)
                used.add(cand.gateway_id)
    return jury


class SelfValidator:
    """Runs the cross-model quorum self-validation (SPEC §3.3)."""

    def __init__(self, llm: LLMClient, quorum_q: int) -> None:
        self.llm = llm
        self.quorum_q = quorum_q

    async def validate_prompt(
        self,
        prompt_idx: int,
        prompt_text: str,
        expected: str,
        intended_tool: str,
        tool_window: list[dict[str, Any]],
        jury: list[ModelInfo],
    ) -> PromptAdmission:
        """Run all jurors on one prompt; compute the quorum admission.

        Jurors that error out (no response) are excluded from N — the quorum
        adapts, matching SPEC §5's flaky-backend handling.
        """
        async def _one(juror: ModelInfo) -> JurorVerdict | None:
            pick = await self.llm.select_tool(juror, tool_window, prompt_text)
            if not pick.gradeable:
                # Errored OR forced-tool-use-unavailable (§2): this juror's
                # pick is not a real verdict. Drop it from N so the quorum
                # never counts a fabricated agreement. A juror that 400s on
                # forced tool-use is excluded exactly like a flaky one.
                log.warning("self_validation.juror_excluded",
                            juror=juror.gateway_id, error=pick.error,
                            forced_unavailable=pick.forced_unavailable)
                return None
            return JurorVerdict(
                juror_model=juror.gateway_id,
                picked=pick.picked,
                agrees=juror_agrees(expected, intended_tool, pick.picked),
            )

        results = await asyncio.gather(*[_one(j) for j in jury])
        verdicts = [v for v in results if v is not None]
        quorum = evaluate_quorum([v.agrees for v in verdicts], self.quorum_q)
        return PromptAdmission(prompt_idx, verdicts, quorum)
