"""Run reporting. SPEC v2 §6.

Writes per-run JSON + Markdown dumps to `~/.umb-validator/runs/`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umb_validator.states import State
from umb_validator.store import StateStore


@dataclass
class RunSummary:
    """Per-run aggregate (SPEC §6)."""

    run_id: str
    with_cloud: bool
    tools_processed: int = 0
    tools_passed: int = 0
    tools_failed: int = 0
    tools_needs_manual: int = 0
    tools_low_confidence: int = 0
    mean_reduction_pct: float = 0.0
    total_cloud_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    local_models_in_roster: list[str] = field(default_factory=list)
    per_tool: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "with_cloud": self.with_cloud,
            "tools_processed": self.tools_processed,
            "tools_passed": self.tools_passed,
            "tools_failed": self.tools_failed,
            "tools_needs_manual": self.tools_needs_manual,
            "tools_low_confidence": self.tools_low_confidence,
            "mean_reduction_pct": round(self.mean_reduction_pct, 2),
            "total_cloud_cost_usd": round(self.total_cloud_cost_usd, 4),
            "wall_time_seconds": round(self.wall_time_seconds, 1),
            "local_models_in_roster": self.local_models_in_roster,
            "per_tool": self.per_tool,
        }


def build_run_summary(
    store: StateStore, run_id: str, final_states: dict[str, State],
    roster: list[str], with_cloud: bool, wall_seconds: float,
) -> RunSummary:
    """Assemble the per-run summary from the state store + final states."""
    summary = RunSummary(run_id=run_id, with_cloud=with_cloud,
                         local_models_in_roster=roster,
                         wall_time_seconds=wall_seconds)
    reductions: list[float] = []
    from umb_validator.gates import count_tokens
    for key, state in final_states.items():
        server, _, tool = key.partition("/")
        summary.tools_processed += 1
        if state == State.REVIEW_READY:
            summary.tools_passed += 1
        elif state == State.NEEDS_MANUAL:
            summary.tools_needs_manual += 1
        else:
            summary.tools_failed += 1
        events = store.events_for(server, tool)
        low_conf = False
        for ev in events:
            if ev["metadata_json"]:
                meta = json.loads(ev["metadata_json"])
                if meta.get("low_self_validation_confidence"):
                    low_conf = True
        if low_conf:
            summary.tools_low_confidence += 1
        runs = store.runs_for(server, tool)
        ctok = next((r["token_count"] for r in runs if r["iteration"] == 0),
                    None)
        stok = next((r["token_count"] for r in runs
                     if 1 <= r["iteration"] < 90), None)
        red = 0.0
        if ctok and stok and ctok > 0:
            red = (ctok - stok) / ctok * 100.0
            reductions.append(red)
        summary.total_cloud_cost_usd += sum(
            r["cost_usd"] for r in runs if r["model_class"] == "cloud")
        summary.per_tool.append({
            "tool": key, "state": str(state), "reduction_pct": round(red, 1),
            "low_confidence": low_conf})
    if reductions:
        summary.mean_reduction_pct = sum(reductions) / len(reductions)
    return summary


def render_markdown(summary: RunSummary) -> str:
    """Render a human-readable Markdown run report (SPEC §6)."""
    lines = [
        f"# umb-validator run {summary.run_id}",
        "",
        f"- Tools processed: {summary.tools_processed}",
        f"- Passed (REVIEW_READY): {summary.tools_passed}",
        f"- Needs manual: {summary.tools_needs_manual}",
        f"- Failed: {summary.tools_failed}",
        f"- Low-confidence: {summary.tools_low_confidence}",
        f"- Mean token reduction: {summary.mean_reduction_pct:.1f}%",
        f"- Cloud cost: ${summary.total_cloud_cost_usd:.2f}",
        f"- Wall time: {summary.wall_time_seconds:.0f}s",
        f"- Roster: {', '.join(summary.local_models_in_roster) or '(none)'}",
        "",
        "## Per-tool",
        "",
        "| Tool | State | Reduction | Confidence |",
        "|---|---|---|---|",
    ]
    for t in summary.per_tool:
        conf = "LOW" if t["low_confidence"] else "high"
        lines.append(
            f"| {t['tool']} | {t['state']} | {t['reduction_pct']}% | {conf} |")
    return "\n".join(lines) + "\n"


def write_run_reports(summary: RunSummary, runs_dir: str | Path) -> tuple[Path, Path]:
    """Write the JSON + Markdown run reports. Returns (json_path, md_path)."""
    runs_dir = Path(os.path.expanduser(str(runs_dir)))
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / f"run-{summary.run_id}.json"
    md_path = runs_dir / f"RUN-{summary.run_id}.md"
    json_path.write_text(json.dumps(summary.to_dict(), indent=2),
                         encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path
