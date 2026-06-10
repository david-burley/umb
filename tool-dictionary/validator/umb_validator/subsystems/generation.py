"""Auto-research subsystem — Generation phase. SPEC v2 §3.2.

Synthesizes the benchmark prompt set from grounding material using a strong
local model (default `qwen3.5-35b-a3b`). Over-generates: 45 positive + 30
negative candidates (self-validation culls down to the trusted oracle).

- Positive prompts: questions that SHOULD select this tool, grounded in real
  documented usage.
- Negative prompts: near-misses — questions that sound like the tool but
  actually want a different tool (the generator is shown sibling + distractor
  schemas so it can ground the 'correct other tool').
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from umb_validator.integration.llm import LLMClient, parse_json_block
from umb_validator.integration.gateway import ModelInfo
from umb_validator.logging_setup import get_logger

log = get_logger("generation")


@dataclass
class GeneratedPrompt:
    """One generated candidate prompt."""

    text: str
    expected: str  # 'positive' | 'negative'
    intended_other: str | None = None  # for negatives: correct other tool name


_POSITIVE_SYS = (
    "You generate evaluation prompts for a tool-routing benchmark. "
    "Output ONLY a JSON array of strings, no prose."
)
_NEGATIVE_SYS = (
    "You generate NEAR-MISS evaluation prompts for a tool-routing benchmark. "
    "Output ONLY a JSON array of objects {\"prompt\": str, \"correct_tool\": str}, "
    "no prose."
)


def _positive_prompt(
    tool_name: str, grounding: str, count: int
) -> str:
    """Build the positive-prompt generation request."""
    return (
        f"Tool under test: `{tool_name}`.\n\n"
        f"Grounding material (canonical schema + upstream docs):\n"
        f"{grounding}\n\n"
        f"Write {count} distinct single-line natural-language user questions "
        f"that SHOULD be routed to `{tool_name}`. Each question must reflect a "
        f"REAL, documented use of the tool — do not invent behaviour the "
        f"grounding material does not support. Vary phrasing, specificity, and "
        f"user intent. Output a JSON array of {count} strings."
    )


def _negative_prompt(
    tool_name: str, grounding: str, sibling_schemas: str, count: int
) -> str:
    """Build the negative (near-miss) generation request."""
    return (
        f"Tool that must NOT be selected: `{tool_name}`.\n\n"
        f"Grounding for `{tool_name}`:\n{grounding}\n\n"
        f"Other available tools (the correct answer must be one of these):\n"
        f"{sibling_schemas}\n\n"
        f"Write {count} distinct single-line user questions that SOUND like "
        f"they want `{tool_name}` but whose correct answer is actually one of "
        f"the OTHER tools listed (near-misses, not random unrelated prompts). "
        f"Output a JSON array of {count} objects, each "
        f'{{"prompt": "<question>", "correct_tool": "<other tool name>"}}.'
    )


def _schemas_block(tools: list[dict[str, Any]]) -> str:
    """Render a compact schema block of sibling/distractor tools."""
    lines: list[str] = []
    for t in tools:
        desc = t.get("description", "")
        lines.append(f"- {t['name']}: {desc[:200]}")
    return "\n".join(lines)


class PromptGenerator:
    """Generates the candidate benchmark prompt set (SPEC §3.2)."""

    def __init__(self, llm: LLMClient, generator: ModelInfo) -> None:
        self.llm = llm
        self.generator = generator

    async def generate_positive(
        self, tool_name: str, grounding: str, count: int,
    ) -> list[GeneratedPrompt]:
        """Generate positive candidate prompts. Robust to under-delivery."""
        comp = await self.llm.complete(
            self.generator,
            _positive_prompt(tool_name, grounding, count),
            system_prompt=_POSITIVE_SYS,
            max_tokens=3000,
            temperature=0.7,
        )
        if comp.error:
            log.warning("generation.positive_failed", tool=tool_name,
                        error=comp.error)
            return []
        try:
            arr = parse_json_block(comp.text)
        except ValueError:
            log.warning("generation.positive_unparseable", tool=tool_name)
            return []
        out: list[GeneratedPrompt] = []
        for item in arr if isinstance(arr, list) else []:
            text = item if isinstance(item, str) else str(item.get("prompt", ""))
            text = text.strip()
            if text:
                out.append(GeneratedPrompt(text=text, expected="positive"))
        log.info("generation.positive", tool=tool_name, n=len(out))
        return out

    async def generate_negative(
        self, tool_name: str, grounding: str,
        sibling_tools: list[dict[str, Any]], count: int,
    ) -> list[GeneratedPrompt]:
        """Generate negative (near-miss) candidate prompts."""
        comp = await self.llm.complete(
            self.generator,
            _negative_prompt(tool_name, grounding,
                             _schemas_block(sibling_tools), count),
            system_prompt=_NEGATIVE_SYS,
            max_tokens=3000,
            temperature=0.7,
        )
        if comp.error:
            log.warning("generation.negative_failed", tool=tool_name,
                        error=comp.error)
            return []
        try:
            arr = parse_json_block(comp.text)
        except ValueError:
            log.warning("generation.negative_unparseable", tool=tool_name)
            return []
        out: list[GeneratedPrompt] = []
        for item in arr if isinstance(arr, list) else []:
            if isinstance(item, dict):
                text = str(item.get("prompt", "")).strip()
                other = item.get("correct_tool")
                other = str(other).strip() if other else None
            else:
                text, other = str(item).strip(), None
            if text:
                out.append(GeneratedPrompt(text=text, expected="negative",
                                           intended_other=other))
        log.info("generation.negative", tool=tool_name, n=len(out))
        return out
