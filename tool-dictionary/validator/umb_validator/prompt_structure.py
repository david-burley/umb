"""Benchmark + jury prompt structure. SPEC v2 §3.4 / DECISION #4.

Each prompt is presented with the candidate Tool + up to 7 distractor tools
from DIFFERENT servers (an 8-tool window). The distractor set is randomized
but SEEDED by (server, tool, prompt_idx) so the SAME distractors are used in
self-validation, baseline, and shortened runs — the only thing that varies is
the description under test.

For NEGATIVE prompts, the intended *correct* tool MUST be present in the
window, or a juror cannot 'correctly not-pick' the candidate (DECISION #4).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

# DECISION #4: candidate + 7 distractors.
DEFAULT_WINDOW = 8


@dataclass
class ToolUniverse:
    """All canonical tools across all configured servers, for distractor
    sampling. Keyed by (server, tool_name)."""

    tools: dict[tuple[str, str], dict[str, Any]]

    @classmethod
    def from_canonical(
        cls, by_server: dict[str, list[Any]]
    ) -> "ToolUniverse":
        """Build from {server: [CanonicalTool, ...]}."""
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for server, tools in by_server.items():
            for t in tools:
                obj = t.as_tool_object() if hasattr(t, "as_tool_object") else t
                name = obj["name"]
                out[(server, name)] = {**obj, "_server": server}
        return cls(out)

    def keys(self) -> list[tuple[str, str]]:
        """All (server, tool) keys."""
        return list(self.tools.keys())

    def get(self, server: str, tool: str) -> dict[str, Any] | None:
        """Fetch a tool object by (server, tool)."""
        return self.tools.get((server, tool))


def _seed(server: str, tool: str, prompt_idx: int) -> int:
    """Deterministic per-prompt seed.

    Uses a SHA-256 digest, NOT the builtin `hash()` — the builtin is salted
    per-process (PYTHONHASHSEED) so it is NOT stable across runs or restarts.
    SPEC §3.4 requires the SAME distractor set in self-validation, baseline,
    and shortened runs, which means the seed must be process-independent.
    """
    key = f"{server}\x00{tool}\x00{prompt_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & 0x7FFFFFFF


def build_window(
    universe: ToolUniverse,
    server: str,
    tool: str,
    prompt_idx: int,
    *,
    window: int = DEFAULT_WINDOW,
    must_include: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build the 8-tool presentation window for one prompt (SPEC §3.4).

    - The candidate tool `(server, tool)` is always included.
    - Distractors are sampled from OTHER servers, seeded for reproducibility.
    - `must_include` (the intended other-tool for a negative prompt) is
      forced into the window (DECISION #4).
    - The window order is shuffled (seeded) so position carries no signal.

    The candidate's tool object uses whatever description the caller has
    already set on the universe entry — callers swap canonical vs shortened
    upstream of this function.
    """
    rng = random.Random(_seed(server, tool, prompt_idx))
    candidate = universe.get(server, tool)
    if candidate is None:
        raise KeyError(f"candidate tool not in universe: {server}/{tool}")

    chosen: dict[tuple[str, str], dict[str, Any]] = {(server, tool): candidate}

    # Force the intended other-tool for negatives.
    if must_include is not None and must_include != (server, tool):
        forced = universe.get(*must_include)
        if forced is not None:
            chosen[must_include] = forced

    # Candidate distractors: tools from DIFFERENT servers, not already chosen.
    pool = [
        k for k in universe.keys()
        if k[0] != server and k not in chosen
    ]
    rng.shuffle(pool)
    for key in pool:
        if len(chosen) >= window:
            break
        chosen[key] = universe.tools[key]

    items = list(chosen.values())
    rng.shuffle(items)
    # Strip internal bookkeeping key before presentation.
    return [{k: v for k, v in t.items() if not k.startswith("_")}
            for t in items]
