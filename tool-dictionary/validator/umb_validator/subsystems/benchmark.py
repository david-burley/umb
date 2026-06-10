"""Benchmark runner. SPEC v2 §4 / §5.

Runs the admitted oracle against a set of models with a given description
(canonical for baseline iteration 0, or a proposed short_description for
iteration >= 1). Each prompt is presented in its seeded 8-tool window with the
candidate tool's description swapped to the description under test.

The judge is exact tool-name match (SPEC §4). Produces per-model accuracy +
latency percentiles.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Any

from umb_validator.gates import composite_accuracy
from umb_validator.integration.gateway import ModelInfo
from umb_validator.integration.llm import LLMClient
from umb_validator.logging_setup import get_logger
from umb_validator.prompt_structure import ToolUniverse, build_window

log = get_logger("benchmark")


class ModelUngateable(Exception):
    """A model rejected forced tool-use (§2) — it cannot be benchmarked.

    Raised by `run_model` when any oracle prompt comes back
    `forced_unavailable`. `run_all_models` catches it and EXCLUDES the model
    from N, exactly as it does for a model that raised a transient error — so
    an un-runnable gate can never contribute a fabricated accuracy.
    """

    def __init__(self, model: str) -> None:
        super().__init__(f"forced tool-use unavailable on model {model!r}")
        self.model = model


@dataclass
class OraclePrompt:
    """One admitted oracle prompt for the benchmark."""

    prompt_idx: int
    text: str
    expected: str  # 'positive' | 'negative'
    intended_tool: str  # the candidate tool (positives) — also used for negatives
    intended_other: str | None = None  # negative: the correct other tool


@dataclass
class ModelBenchmark:
    """Per-model benchmark result for one (tool, iteration)."""

    model: ModelInfo
    n_positive: int
    n_negative: int
    n_correct_pos: int
    n_correct_neg: int
    accuracy: float
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def p50_latency_ms(self) -> float | None:
        return statistics.median(self.latencies_ms) if self.latencies_ms else None

    @property
    def p95_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        s = sorted(self.latencies_ms)
        idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
        return s[idx]


def _swap_description(
    universe: ToolUniverse, server: str, tool: str, description: str,
) -> ToolUniverse:
    """Return a shallow-copied universe with the candidate tool's description
    replaced by `description` (so baseline vs shortened only differ there)."""
    new_tools = dict(universe.tools)
    candidate = dict(new_tools[(server, tool)])
    candidate["description"] = description
    new_tools[(server, tool)] = candidate
    return ToolUniverse(new_tools)


class BenchmarkRunner:
    """Scores the admitted oracle across a model set (SPEC §4)."""

    def __init__(
        self, llm: LLMClient,
        positive_weight: float = 0.6, negative_weight: float = 0.4,
    ) -> None:
        self.llm = llm
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    async def run_model(
        self,
        model: ModelInfo,
        server: str,
        tool: str,
        description: str,
        oracle: list[OraclePrompt],
        universe: ToolUniverse,
        window: int = 8,
    ) -> ModelBenchmark:
        """Run the full oracle against ONE model with `description` in place.

        Returns per-model accuracy + latency percentiles. The prompt windows
        are seeded -> identical across baseline and shortened iterations.
        """
        swapped = _swap_description(universe, server, tool, description)

        async def _one(p: OraclePrompt) -> tuple[str, bool, float]:
            must_include: tuple[str, str] | None = None
            if p.expected == "negative" and p.intended_other:
                # Ensure the correct other-tool is in the window.
                for (s, t) in universe.keys():
                    if t == p.intended_other:
                        must_include = (s, t)
                        break
            win = build_window(swapped, server, tool, p.prompt_idx,
                               window=window, must_include=must_include)
            pick = await self.llm.select_tool(model, win, p.text)
            if pick.forced_unavailable:
                # §2: forced tool-use rejected — this is NOT a no-pick. The
                # gate cannot run on this model; abort it entirely so it is
                # excluded from N rather than scored as a fabricated ~0.4.
                raise ModelUngateable(model.gateway_id)
            if p.expected == "positive":
                correct = pick.picked == tool
            else:
                correct = pick.picked != tool
            return (p.expected, correct, pick.latency_ms)

        results = await asyncio.gather(*[_one(p) for p in oracle])
        n_pos = sum(1 for e, _, _ in results if e == "positive")
        n_neg = sum(1 for e, _, _ in results if e == "negative")
        cpos = sum(1 for e, c, _ in results if e == "positive" and c)
        cneg = sum(1 for e, c, _ in results if e == "negative" and c)
        lat = [ms for _, _, ms in results]
        acc = composite_accuracy(cpos, n_pos, cneg, n_neg,
                                 self.positive_weight, self.negative_weight)
        log.info("benchmark.model_done", server=server, tool=tool,
                 model=model.gateway_id, accuracy=round(acc, 4),
                 n_correct_pos=cpos, n_correct_neg=cneg)
        return ModelBenchmark(model, n_pos, n_neg, cpos, cneg, acc, lat)

    async def run_all_models(
        self,
        models: list[ModelInfo],
        server: str,
        tool: str,
        description: str,
        oracle: list[OraclePrompt],
        universe: ToolUniverse,
        window: int = 8,
    ) -> list[ModelBenchmark]:
        """Run the oracle against every model. Models that raise are skipped
        (excluded from N, SPEC §5).

        §2 / §13 K-of-N: a model that rejects forced tool-use
        (`ModelUngateable`) is likewise excluded from N — the gate genuinely
        ran only on the models that support it. If that leaves too few
        gateable models, the caller (pipeline) detects the under-strength
        jury and routes the tool to NEEDS_MANUAL rather than fabricating a
        pass from an incomplete benchmark.
        """
        async def _safe(m: ModelInfo) -> ModelBenchmark | None:
            try:
                return await self.run_model(m, server, tool, description,
                                            oracle, universe, window)
            except ModelUngateable:
                log.warning("benchmark.model_ungateable", model=m.gateway_id,
                            reason="forced_tool_use_unavailable")
                return None
            except Exception as exc:  # noqa: BLE001
                log.warning("benchmark.model_failed", model=m.gateway_id,
                            error=str(exc))
                return None

        results = await asyncio.gather(*[_safe(m) for m in models])
        return [r for r in results if r is not None]
