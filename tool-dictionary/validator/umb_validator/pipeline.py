"""Pipeline orchestrator. SPEC v2 §1 (state machine + scheduler) / §3-§6.

Drives each tool through the per-tool state machine:

  PENDING -> CANONICAL_FETCHED -> HASH_COMPUTED -> RESEARCHED ->
  PROMPTS_GENERATED -> PROMPTS_SELF_VALIDATED -> BASELINE_RUN ->
  SHORTENED_PROPOSED -> LOCAL_GATE_RUN -> LOCAL_GATE_{PASS,FAIL} ->
  [CLOUD_GATE_* if --with-cloud] -> REVIEW_READY

Every long-running step records a `state_event` (idempotency stamp) BEFORE
dispatching inference, so a SIGTERM/restart resumes with no double-work.
The scheduler runs up to `max_concurrent_tools` tools as asyncio tasks.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from umb_validator.config import Config
from umb_validator.gates import (
    PerModelGate, count_tokens, evaluate_cloud_gate, evaluate_local_gate,
    mean_juror_agreement, per_model_cloud_pass, per_model_local_pass,
    token_reduction_pct, token_gate_pass, wilson_ci,
)
from umb_validator.hashing import description_hash
from umb_validator.integration.gateway import GatewayClient, ModelInfo, SessionPool
from umb_validator.integration.llm import LLMClient
from umb_validator.integration.umb_dev import CanonicalTool
from umb_validator.logging_setup import get_logger
from umb_validator.prompt_structure import ToolUniverse
from umb_validator.states import State
from umb_validator.store import StateStore
from umb_validator.subsystems.benchmark import BenchmarkRunner, OraclePrompt
from umb_validator.subsystems.generation import PromptGenerator
from umb_validator.subsystems.research import ResearchClient, grounding_summary
from umb_validator.subsystems.self_validation import SelfValidator, select_jury
from umb_validator.subsystems.shortener import Shortener

log = get_logger("pipeline")


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ToolContext:
    """Per-tool working context threaded through the pipeline."""

    server: str
    tool: str
    canonical: CanonicalTool
    upstream_source: str | None = None
    upstream_pinned: str | None = None
    grounding: str = "schema_only"
    oracle: list[OraclePrompt] = field(default_factory=list)
    baseline_acc: dict[str, float] = field(default_factory=dict)
    oracle_mean_agreement: float = 0.0
    low_confidence: bool = False
    shorten_attempts: list[str] = field(default_factory=list)
    proposed_short: str | None = None

    @property
    def key(self) -> str:
        return f"{self.server}/{self.tool}"


@dataclass
class PipelineDeps:
    """All injected collaborators — lets the CLI wire real clients and tests
    wire mocks."""

    config: Config
    store: StateStore
    gateway: GatewayClient
    pool: SessionPool
    llm: LLMClient
    research: ResearchClient
    universe: ToolUniverse
    roster: list[ModelInfo]
    jury_models: list[ModelInfo]
    generator: ModelInfo
    cloud_models: list[ModelInfo] = field(default_factory=list)


class ToolPipeline:
    """Drives ONE tool through the full state machine (SPEC §1)."""

    def __init__(self, deps: PipelineDeps, run_id: str, with_cloud: bool):
        self.d = deps
        self.run_id = run_id
        self.with_cloud = with_cloud
        self.cfg = deps.config

    # --------- state helpers ---------

    def _event(self, ctx: ToolContext, state: State,
               meta: dict[str, Any] | None = None) -> None:
        """Record a state transition (idempotency stamp + WAL flush)."""
        self.d.store.record_event(ctx.server, ctx.tool, state, meta)
        log.info("pipeline.state", server=ctx.server, tool=ctx.tool,
                 state=str(state))

    # --------- step 1: hash ---------

    async def step_hash(self, ctx: ToolContext) -> None:
        """CANONICAL_FETCHED -> HASH_COMPUTED. SPEC §5 hash policy.

        Hash of the live canonical description, byte-for-byte. Stored on the
        tool row. This step alone activates Auto-mode drift protection.
        """
        h = description_hash(ctx.canonical.description)
        self.d.store.upsert_tool(ctx.server, ctx.tool, current_hash=h,
                                 upstream_pinned=ctx.upstream_pinned)
        self._event(ctx, State.HASH_COMPUTED, {"hash": h})

    # --------- step 2: research ---------

    async def step_research(self, ctx: ToolContext) -> None:
        """HASH_COMPUTED -> RESEARCHED. SPEC §3.1."""
        artifacts, grounding = await self.d.research.gather(
            ctx.canonical, ctx.upstream_source)
        ctx.grounding = grounding
        for a in artifacts:
            self.d.store.add_research_artifact(
                ctx.server, ctx.tool, a.kind, a.content,
                a.source_url, a.source_pinned)
        self._event(ctx, State.RESEARCHED, {"grounding": grounding,
                                            "n_artifacts": len(artifacts)})

    # --------- step 3: generate ---------

    def _grounding_text(self, ctx: ToolContext) -> str:
        """Reconstruct the grounding block from stored artifacts."""
        from umb_validator.subsystems.research import Artifact
        rows = self.d.store.research_artifacts(ctx.server, ctx.tool)
        arts = [Artifact(kind=r["kind"], content=r["content"],
                         source_url=r["source_url"],
                         source_pinned=r["source_pinned"]) for r in rows]
        return grounding_summary(arts)

    def _sibling_tools(self, ctx: ToolContext) -> list[dict[str, Any]]:
        """Sibling + sample distractor tool objects for negative generation.

        Same-server siblings are listed FIRST (a near-miss negative most often
        belongs to a sibling), then cross-server distractors, capped at 15.
        """
        siblings: list[dict[str, Any]] = []
        distractors: list[dict[str, Any]] = []
        for (s, t), obj in self.d.universe.tools.items():
            if (s, t) == (ctx.server, ctx.tool):
                continue
            entry = {"name": t, "description": obj.get("description", "")}
            if s == ctx.server:
                siblings.append(entry)
            else:
                distractors.append(entry)
        return (siblings + distractors)[:15]

    async def step_generate(self, ctx: ToolContext) -> None:
        """RESEARCHED -> PROMPTS_GENERATED. SPEC §3.2."""
        gen = PromptGenerator(self.d.llm, self.d.generator)
        grounding = self._grounding_text(ctx)
        siblings = self._sibling_tools(ctx)
        idx = 0
        rounds = 0
        oc = self.cfg.oracle
        while rounds < oc.max_gen_rounds:
            pos = await gen.generate_positive(
                ctx.tool, grounding, oc.gen_positive)
            neg = await gen.generate_negative(
                ctx.tool, grounding, siblings, oc.gen_negative)
            for gp in pos + neg:
                self.d.store.add_prompt(
                    ctx.server, ctx.tool, idx, gp.text, gp.expected,
                    "auto_research", self.d.generator.gateway_id)
                idx += 1
            rounds += 1
            # Enough raw candidates to plausibly fill the oracle after culling?
            if len(pos) >= oc.oracle_min_positive and \
               len(neg) >= oc.oracle_min_negative:
                break
        self._event(ctx, State.PROMPTS_GENERATED, {"n_candidates": idx,
                                                   "gen_rounds": rounds})

    # --------- step 4: self-validate ---------

    async def step_self_validate(self, ctx: ToolContext) -> bool:
        """PROMPTS_GENERATED -> PROMPTS_SELF_VALIDATED. SPEC §3.3.

        Returns True if the oracle met ORACLE_MIN; False -> NEEDS_MANUAL.
        """
        from umb_validator.prompt_structure import build_window
        jury = select_jury(self.d.jury_models, self.d.generator.gateway_id,
                            self.d.roster)
        validator = SelfValidator(self.d.llm, self.cfg.jury.quorum_q)
        prompts = self.d.store.prompts_for(ctx.server, ctx.tool)
        admissions = []
        for p in prompts:
            intended_other = None
            # Negative prompts: derive intended-other from window must_include.
            window = build_window(
                self.d.universe, ctx.server, ctx.tool, p["prompt_idx"],
                window=self.cfg.research.distractor_window)
            adm = await validator.validate_prompt(
                p["prompt_idx"], p["prompt_text"], p["expected"],
                ctx.tool, window, jury)
            admissions.append((p, adm))
            for v in adm.verdicts:
                self.d.store.add_self_validation(
                    p["id"], v.juror_model, v.picked, v.agrees)
            self.d.store.set_prompt_admitted(p["id"], adm.admitted)
        admitted_pos = sum(1 for p, a in admissions
                           if a.admitted and p["expected"] == "positive")
        admitted_neg = sum(1 for p, a in admissions
                           if a.admitted and p["expected"] == "negative")
        oc = self.cfg.oracle
        quorums = [a.quorum for _, a in admissions if a.admitted]
        ctx.oracle_mean_agreement = mean_juror_agreement(quorums)
        if (admitted_pos < oc.oracle_min_positive or
                admitted_neg < oc.oracle_min_negative):
            self._event(ctx, State.NEEDS_MANUAL, {
                "reason": "oracle_too_small",
                "admitted_positive": admitted_pos,
                "admitted_negative": admitted_neg,
            })
            return False
        ctx.low_confidence = (
            ctx.grounding == "schema_only"
            or ctx.oracle_mean_agreement < oc.low_conf_agreement
        )
        # Build the in-memory oracle for the benchmark.
        for p, a in admissions:
            if not a.admitted:
                continue
            ctx.oracle.append(OraclePrompt(
                prompt_idx=p["prompt_idx"], text=p["prompt_text"],
                expected=p["expected"], intended_tool=ctx.tool))
        self._event(ctx, State.PROMPTS_SELF_VALIDATED, {
            "oracle_size": len(ctx.oracle),
            "oracle_mean_agreement": round(ctx.oracle_mean_agreement, 4),
            "low_self_validation_confidence": ctx.low_confidence,
        })
        return True

    # --------- step 5: baseline ---------

    async def step_baseline(self, ctx: ToolContext) -> bool:
        """PROMPTS_SELF_VALIDATED -> BASELINE_RUN. SPEC §4.

        Run the admitted oracle with the FULL canonical description on every
        local model -> per-model baseline accuracy.

        Returns False (-> NEEDS_MANUAL, reason `gateway_ungateable`) if too
        few local models support forced tool-use (SPEC §2 / §13). A model
        that 400s on forced tool-use is excluded from N by `run_all_models`;
        if the surviving gateable count is below the K-of-roster floor, a
        valid gate is impossible and the tool MUST NOT proceed to a gate that
        would fabricate a pass.
        """
        runner = BenchmarkRunner(self.d.llm, self.cfg.gates.positive_weight,
                                 self.cfg.gates.negative_weight)
        results = await runner.run_all_models(
            self.d.roster, ctx.server, ctx.tool,
            ctx.canonical.description, ctx.oracle, self.d.universe,
            self.cfg.research.distractor_window)
        min_gateable = self.cfg.gates.min_gateable_local(len(self.d.roster))
        if len(results) < min_gateable:
            self._event(ctx, State.NEEDS_MANUAL, {
                "reason": "gateway_ungateable",
                "n_gateable_models": len(results),
                "n_roster_models": len(self.d.roster),
                "min_gateable_required": min_gateable,
            })
            return False
        chash = description_hash(ctx.canonical.description)
        ctok = count_tokens(ctx.canonical.description)
        for r in results:
            ctx.baseline_acc[r.model.config_name] = r.accuracy
            self.d.store.add_validation_run(
                run_id=self.run_id, server_name=ctx.server,
                tool_name=ctx.tool, iteration=0,
                description=ctx.canonical.description,
                description_hash=chash, model=r.model.gateway_id,
                model_class="local", backend=r.model.backend,
                n_prompts=len(ctx.oracle), n_positive=r.n_positive,
                n_negative=r.n_negative, n_correct_pos=r.n_correct_pos,
                n_correct_neg=r.n_correct_neg, accuracy=r.accuracy,
                token_count=ctok, p50_latency_ms=r.p50_latency_ms,
                p95_latency_ms=r.p95_latency_ms, cost_usd=0.0)
        self._event(ctx, State.BASELINE_RUN, {
            "baseline_accuracy": {k: round(v, 4)
                                  for k, v in ctx.baseline_acc.items()}})
        return True

    # --------- step 6+: shorten + local gate loop ---------

    async def step_shorten_and_gate(self, ctx: ToolContext,
                                    feedback: str | None = None) -> bool:
        """BASELINE_RUN -> SHORTENED_PROPOSED -> LOCAL_GATE_RUN -> PASS/FAIL.

        Retries the shortener up to `shortener_max_retries`. Returns True iff
        a LOCAL_GATE_PASS was reached.
        """
        shortener = Shortener(self.d.llm, self.d.generator)
        runner = BenchmarkRunner(self.d.llm, self.cfg.gates.positive_weight,
                                 self.cfg.gates.negative_weight)
        ctok = count_tokens(ctx.canonical.description)
        max_iter = self.cfg.gates.shortener_max_retries
        for iteration in range(1, max_iter + 1):
            proposal = await shortener.propose(
                ctx.tool, ctx.canonical.description, iteration,
                feedback=feedback, prior_attempts=ctx.shorten_attempts)
            if proposal.error or not proposal.short_description:
                feedback = f"shortener error: {proposal.error}"
                continue
            short = proposal.short_description
            ctx.shorten_attempts.append(short)
            self._event(ctx, State.SHORTENED_PROPOSED,
                        {"iteration": iteration, "short": short})

            stok = count_tokens(short)
            reduction = token_reduction_pct(ctok, stok)
            if not token_gate_pass(ctok, stok,
                                   self.cfg.gates.token_reduction_min):
                feedback = (f"token reduction {reduction:.0f}% is below the "
                            f"{self.cfg.gates.token_reduction_min:.0f}% "
                            f"minimum — be terser")
                self._event(ctx, State.LOCAL_GATE_FAIL,
                            {"iteration": iteration, "reason": "token_gate",
                             "reduction_pct": round(reduction, 2)})
                continue

            self._event(ctx, State.LOCAL_GATE_RUN, {"iteration": iteration})
            results = await runner.run_all_models(
                self.d.roster, ctx.server, ctx.tool, short,
                ctx.oracle, self.d.universe,
                self.cfg.research.distractor_window)
            # §2 / §13: if forced tool-use is unavailable on too many local
            # models, the gate cannot legitimately run. Terminate to
            # NEEDS_MANUAL rather than passing on an under-strength jury —
            # never fabricate a LOCAL_GATE_PASS off un-runnable gates.
            min_gateable = self.cfg.gates.min_gateable_local(
                len(self.d.roster))
            if len(results) < min_gateable:
                self._event(ctx, State.NEEDS_MANUAL, {
                    "reason": "gateway_ungateable",
                    "iteration": iteration,
                    "n_gateable_models": len(results),
                    "n_roster_models": len(self.d.roster),
                    "min_gateable_required": min_gateable,
                })
                return False
            shash = description_hash(short)
            per_model: list[PerModelGate] = []
            for r in results:
                base = ctx.baseline_acc.get(r.model.config_name, 0.0)
                passed = per_model_local_pass(
                    base, r.accuracy, self.cfg.gates.local_pp_tolerance)
                per_model.append(PerModelGate(
                    r.model.config_name, base, r.accuracy, passed))
                self.d.store.add_validation_run(
                    run_id=self.run_id, server_name=ctx.server,
                    tool_name=ctx.tool, iteration=iteration,
                    description=short, description_hash=shash,
                    model=r.model.gateway_id, model_class="local",
                    backend=r.model.backend, n_prompts=len(ctx.oracle),
                    n_positive=r.n_positive, n_negative=r.n_negative,
                    n_correct_pos=r.n_correct_pos,
                    n_correct_neg=r.n_correct_neg, accuracy=r.accuracy,
                    token_count=stok, p50_latency_ms=r.p50_latency_ms,
                    p95_latency_ms=r.p95_latency_ms, cost_usd=0.0)
            gate = evaluate_local_gate(per_model,
                                       self.cfg.gates.local_k_fraction)
            if gate.passed:
                ctx.proposed_short = short
                self._event(ctx, State.LOCAL_GATE_PASS, {
                    "iteration": iteration, "gate": gate.summary(),
                    "reduction_pct": round(reduction, 2)})
                return True
            failing = [pm.model for pm in per_model if not pm.passed]
            feedback = (f"accuracy regressed on {failing} "
                        f"(gate {gate.summary()}); preserve discriminating "
                        f"detail those models rely on")
            self._event(ctx, State.LOCAL_GATE_FAIL, {
                "iteration": iteration, "gate": gate.summary(),
                "failing_models": failing})
        # Exhausted retries.
        self._event(ctx, State.NEEDS_MANUAL,
                    {"reason": "local_gate_exhausted",
                     "attempts": ctx.shorten_attempts})
        return False

    # --------- step 7: cloud gate ---------

    async def step_cloud_gate(self, ctx: ToolContext) -> bool:
        """LOCAL_GATE_PASS -> CLOUD_GATE_RUN -> PASS/FAIL. SPEC §5.

        Only runs when `--with-cloud`. Returns True on CLOUD_GATE_PASS.
        """
        if not self.with_cloud or not self.d.cloud_models:
            return True
        assert ctx.proposed_short is not None
        runner = BenchmarkRunner(self.d.llm, self.cfg.gates.positive_weight,
                                 self.cfg.gates.negative_weight)
        self._event(ctx, State.CLOUD_GATE_RUN, {})
        # Cloud baseline.
        base_results = await runner.run_all_models(
            self.d.cloud_models, ctx.server, ctx.tool,
            ctx.canonical.description, ctx.oracle, self.d.universe,
            self.cfg.research.distractor_window)
        cloud_base = {r.model.config_name: r.accuracy for r in base_results}
        # Cloud shortened.
        short_results = await runner.run_all_models(
            self.d.cloud_models, ctx.server, ctx.tool, ctx.proposed_short,
            ctx.oracle, self.d.universe, self.cfg.research.distractor_window)
        shash = description_hash(ctx.proposed_short)
        stok = count_tokens(ctx.proposed_short)
        per_model: list[PerModelGate] = []
        cost = 0.0
        for r in short_results:
            base = cloud_base.get(r.model.config_name, 0.0)
            passed = per_model_cloud_pass(base, r.accuracy,
                                          self.cfg.gates.cloud_rel_floor)
            per_model.append(PerModelGate(r.model.config_name, base,
                                          r.accuracy, passed))
            self.d.store.add_validation_run(
                run_id=self.run_id, server_name=ctx.server,
                tool_name=ctx.tool, iteration=99, description=ctx.proposed_short,
                description_hash=shash, model=r.model.gateway_id,
                model_class="cloud", backend="cloud",
                n_prompts=len(ctx.oracle), n_positive=r.n_positive,
                n_negative=r.n_negative, n_correct_pos=r.n_correct_pos,
                n_correct_neg=r.n_correct_neg, accuracy=r.accuracy,
                token_count=stok, p50_latency_ms=r.p50_latency_ms,
                p95_latency_ms=r.p95_latency_ms, cost_usd=0.0)
        if evaluate_cloud_gate(per_model):
            self._event(ctx, State.CLOUD_GATE_PASS, {})
            return True
        self._event(ctx, State.CLOUD_GATE_FAIL,
                    {"failing": [pm.model for pm in per_model
                                 if not pm.passed]})
        return False

    # --------- step 8: review ready ---------

    async def step_review_ready(self, ctx: ToolContext,
                                writer: "PendingWriter") -> None:
        """LOCAL/CLOUD_GATE_PASS -> REVIEW_READY. SPEC §6 / §9.

        Writes the proposal to `_pending/<server>.toml`.
        """
        assert ctx.proposed_short is not None
        shash = description_hash(ctx.canonical.description)
        path = writer.write_proposal(ctx)
        self.d.store.add_pending_diff(
            ctx.server, ctx.tool, ctx.proposed_short, shash, str(path))
        self._event(ctx, State.REVIEW_READY, {
            "pending_path": str(path),
            "low_self_validation_confidence": ctx.low_confidence})


class PendingWriter:
    """Writes REVIEW_READY proposals to `_pending/<server>.toml` (SPEC §9).

    NEVER commits, NEVER branches. Accumulates entries per server file.
    """

    def __init__(self, config: Config, store: StateStore):
        self.cfg = config
        self.store = store
        self.pending_dir = config.resolve_pending_dir()

    def write_proposal(self, ctx: ToolContext) -> Any:
        """Write/update the `_pending/<server>.toml` proposal for one tool."""
        from umb_validator.toml_io import (
            compute_and_format_provenance, load_document, new_pending_document,
            upsert_tool_in_document, add_provenance_comment, write_document,
        )
        assert ctx.proposed_short is not None
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        path = self.pending_dir / f"{ctx.server}.toml"
        if path.exists():
            doc = load_document(path)
        else:
            doc = new_pending_document(
                ctx.server, ctx.upstream_source, curator="umb-validator",
                reviewed_at=_utcnow()[:10])
        shash = description_hash(ctx.canonical.description)
        upsert_tool_in_document(doc, ctx.tool, ctx.proposed_short, shash)
        # Provenance block.
        runs0 = self.store.runs_for(ctx.server, ctx.tool, 0)
        baseline = {r["model"]: r["accuracy"] for r in runs0}
        last_iter = max((r["iteration"] for r in
                         self.store.runs_for(ctx.server, ctx.tool)
                         if r["iteration"] < 90), default=1)
        runsN = self.store.runs_for(ctx.server, ctx.tool, last_iter)
        shortened = {r["model"]: r["accuracy"] for r in runsN}
        ctok = count_tokens(ctx.canonical.description)
        stok = count_tokens(ctx.proposed_short)
        prov = compute_and_format_provenance(
            reviewed_at=_utcnow(), run_id="latest",
            local_gate="PASS", reduction_pct=token_reduction_pct(ctok, stok),
            baseline_acc=baseline, shortened_acc=shortened,
            oracle_size=len(ctx.oracle),
            mean_agreement=ctx.oracle_mean_agreement,
            low_confidence=ctx.low_confidence)
        add_provenance_comment(doc, ctx.tool, prov)
        write_document(path, doc)
        log.info("pipeline.pending_written", path=str(path), tool=ctx.tool)
        return path


async def run_tool_pipeline(
    pipeline: ToolPipeline, ctx: ToolContext, writer: PendingWriter,
    start_state: State,
) -> State:
    """Drive ONE tool from `start_state` to a terminal/review state.

    Resumable: `start_state` is the tool's current state from the store, so a
    restart picks up exactly where it left off (no re-running passed steps).
    """
    state = start_state
    try:
        if state in (State.PENDING, State.CANONICAL_FETCHED):
            await pipeline.step_hash(ctx)
            state = State.HASH_COMPUTED
        if state == State.HASH_COMPUTED:
            await pipeline.step_research(ctx)
            state = State.RESEARCHED
        if state == State.RESEARCHED:
            await pipeline.step_generate(ctx)
            state = State.PROMPTS_GENERATED
        if state == State.PROMPTS_GENERATED:
            ok = await pipeline.step_self_validate(ctx)
            if not ok:
                return State.NEEDS_MANUAL
            state = State.PROMPTS_SELF_VALIDATED
        if state == State.PROMPTS_SELF_VALIDATED:
            ok = await pipeline.step_baseline(ctx)
            if not ok:
                # §2: forced tool-use unavailable on too many local models.
                return State.NEEDS_MANUAL
            state = State.BASELINE_RUN
        if state in (State.BASELINE_RUN, State.LOCAL_GATE_FAIL,
                     State.SHORTENED_PROPOSED, State.REJECTED):
            ok = await pipeline.step_shorten_and_gate(ctx)
            if not ok:
                return State.NEEDS_MANUAL
            state = State.LOCAL_GATE_PASS
        if state == State.LOCAL_GATE_PASS and pipeline.with_cloud:
            ok = await pipeline.step_cloud_gate(ctx)
            if not ok:
                return State.NEEDS_MANUAL
            state = State.CLOUD_GATE_PASS
        if state in (State.LOCAL_GATE_PASS, State.CLOUD_GATE_PASS):
            await pipeline.step_review_ready(ctx, writer)
            state = State.REVIEW_READY
        return state
    except Exception as exc:  # noqa: BLE001 — pipeline-wide failure guard
        log.error("pipeline.tool_failed", server=ctx.server, tool=ctx.tool,
                  error=str(exc), exc_info=True)
        pipeline.d.store.record_event(
            ctx.server, ctx.tool, State.NEEDS_MANUAL,
            {"reason": "pipeline_exception", "error": str(exc)})
        return State.NEEDS_MANUAL


class Scheduler:
    """Runs up to `max_concurrent_tools` tool pipelines concurrently, with a
    wall-clock soft cap (SPEC §1 / §7)."""

    def __init__(self, deps: PipelineDeps, run_id: str, with_cloud: bool):
        self.d = deps
        self.run_id = run_id
        self.with_cloud = with_cloud
        self._sem = asyncio.Semaphore(deps.config.session_pool.max_concurrent_tools)

    async def run(
        self, contexts: list[tuple[ToolContext, State]],
    ) -> dict[str, State]:
        """Schedule all tool contexts. Returns {tool_key: final_state}.

        Honors `max_run_wall_seconds`: once the cap is hit, NEW tools are not
        started; in-flight tools finish. The run is fully resumable.
        """
        writer = PendingWriter(self.d.config, self.d.store)
        deadline = time.monotonic() + self.d.config.budget.max_run_wall_seconds
        results: dict[str, State] = {}

        async def _one(ctx: ToolContext, start: State) -> None:
            async with self._sem:
                if time.monotonic() > deadline:
                    log.warning("scheduler.wall_cap_skip", tool=ctx.key)
                    results[ctx.key] = start
                    return
                pipeline = ToolPipeline(self.d, self.run_id, self.with_cloud)
                final = await run_tool_pipeline(pipeline, ctx, writer, start)
                results[ctx.key] = final

        await asyncio.gather(*[_one(c, s) for c, s in contexts])
        return results
