"""Definition-shortening subsystem. SPEC v2 §1 (SHORTENED_PROPOSED) / §5.

A local LLM proposes a terse `short_description` for a tool. On a gate failure
the shortener is retried (up to 5×) with feedback; an owner REJECT reason is
fed into the retry prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from umb_validator.integration.gateway import ModelInfo
from umb_validator.integration.llm import LLMClient
from umb_validator.logging_setup import get_logger

log = get_logger("shortener")

_SYS = (
    "You write terse tool descriptions for an AI tool-routing dictionary. "
    "Output ONLY the description text — no quotes, no prose, no markdown, "
    "one line."
)


@dataclass
class ShortenProposal:
    """A proposed short description."""

    short_description: str
    iteration: int
    error: str | None = None


def _build_prompt(
    tool_name: str,
    canonical_description: str,
    feedback: str | None,
    prior_attempts: list[str],
) -> str:
    """Build the shortener request, optionally with retry feedback."""
    base = (
        f"Tool name: `{tool_name}`\n\n"
        f"Canonical (verbose) description:\n{canonical_description}\n\n"
        "Write a SHORT description: 4-12 words, action-verb first, no "
        "marketing fluff. An AI agent must still be able to pick this tool "
        "correctly from the short text alone, and must NOT confuse it with "
        "similar tools. Aim to cut at least 50% of the tokens."
    )
    if prior_attempts:
        base += ("\n\nPrevious attempts that FAILED the accuracy gate "
                 "(do not repeat them, be more discriminating):\n"
                 + "\n".join(f"- {a}" for a in prior_attempts))
    if feedback:
        base += f"\n\nReviewer / gate feedback to address:\n{feedback}"
    return base


def _clean(text: str) -> str:
    """Normalize an LLM short-description response to one clean line."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.strip().strip('"').strip("'").strip()


class Shortener:
    """Proposes shortened tool descriptions (SPEC §1 / §5)."""

    def __init__(self, llm: LLMClient, model: ModelInfo) -> None:
        self.llm = llm
        self.model = model

    async def propose(
        self,
        tool_name: str,
        canonical_description: str,
        iteration: int,
        feedback: str | None = None,
        prior_attempts: list[str] | None = None,
    ) -> ShortenProposal:
        """Propose one `short_description`. `iteration` is 1..N.

        `feedback` is gate / owner-reject feedback; `prior_attempts` are
        previously-failed candidates the model should not repeat.
        """
        comp = await self.llm.complete(
            self.model,
            _build_prompt(tool_name, canonical_description, feedback,
                          prior_attempts or []),
            system_prompt=_SYS,
            max_tokens=128,
            temperature=0.3 + 0.15 * (iteration - 1),  # diversify on retries
        )
        if comp.error:
            return ShortenProposal("", iteration, error=comp.error)
        short = _clean(comp.text)
        if not short:
            return ShortenProposal("", iteration,
                                   error="empty shortener response")
        log.info("shortener.proposed", tool=tool_name, iteration=iteration,
                 short=short)
        return ShortenProposal(short, iteration)
