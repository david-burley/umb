"""Run wiring + the `run` orchestration. SPEC v2 §1 / §9.

Wires real clients (gateway discovery, umb-dev stdio, openai SDK) into the
PipelineDeps, fetches canonical tool defs, and drives the Scheduler.

Network/binary-dependent: when the gateway or umb-dev is unreachable the
caller (CLI) handles the failure gracefully.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from umb_validator.config import SEED_SERVERS_SHIPPED, Config
from umb_validator.integration.gateway import GatewayClient, ModelInfo, SessionPool
from umb_validator.integration.llm import LLMClient
from umb_validator.integration.umb_dev import (
    CanonicalTool, UmbDevError, UmbDevSession, default_seed_registry,
    write_servers_json,
)
from umb_validator.logging_setup import get_logger
from umb_validator.pipeline import PipelineDeps, Scheduler, ToolContext
from umb_validator.prompt_structure import ToolUniverse
from umb_validator.reporting import build_run_summary, write_run_reports
from umb_validator.research_wiring import build_research_client
from umb_validator.states import State
from umb_validator.store import StateStore

log = get_logger("runner")

# Upstream canonical sources for the shipped seed servers (SPEC §3.1).
_SEED_UPSTREAM = {
    s: f"https://github.com/modelcontextprotocol/servers/tree/main/src/{s}"
    for s in SEED_SERVERS_SHIPPED
}


def new_run_id() -> str:
    """An ISO8601-ish run id safe for filenames."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def make_chat_fn(base_url: str, api_key: str = "x") -> Any:
    """Build an async chat-completions callable bound to an OpenAI-compatible
    endpoint. Imported lazily so the package imports without `openai`."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def _chat(**kwargs: Any) -> Any:
        return await client.chat.completions.create(**kwargs)

    return _chat


@dataclass
class RunArtifacts:
    """What a completed run produced."""

    run_id: str
    final_states: dict[str, State]
    json_report: Path
    md_report: Path


async def discover_and_build_roster(
    cfg: Config,
) -> tuple[GatewayClient, list[ModelInfo], list[ModelInfo], ModelInfo]:
    """Query the gateway, build the local roster + jury + generator.

    Raises on gateway-unreachable — the CLI decides whether to abort.
    """
    gw = GatewayClient(cfg)
    ids = await gw.discover()
    roster = gw.build_roster(ids)
    jury = gw.resolve_jury(cfg.jury.models, ids)
    generator = gw.resolve_generator(ids)
    if generator is None:
        raise RuntimeError("could not resolve a generator model from roster")
    log.info("runner.roster", roster=[m.gateway_id for m in roster],
             jury=[m.gateway_id for m in jury],
             generator=generator.gateway_id)
    return gw, roster, jury, generator


async def fetch_canonical_tools(
    cfg: Config, servers: list[str],
) -> dict[str, list[CanonicalTool]]:
    """Spawn umb-dev per server and capture canonical tool defs (SPEC §11).

    Servers that fail to spawn are logged + skipped (empty list).
    """
    state_dir = cfg.paths.state_path()
    home_dir = state_dir / "umb-home"
    home_dir.mkdir(parents=True, exist_ok=True)
    servers_json = home_dir / "servers.json"
    out: dict[str, list[CanonicalTool]] = {}
    for server in servers:
        write_servers_json(servers_json, default_seed_registry([server]))
        try:
            async with UmbDevSession(
                cfg.paths.umb_dev_bin, home_dir, servers_json,
            ) as sess:
                out[server] = await sess.list_tools(server)
        except UmbDevError as exc:
            log.error("runner.umb_dev_failed", server=server, error=str(exc))
            out[server] = []
    return out


def build_cloud_models(cfg: Config) -> list[ModelInfo]:
    """Build the cloud model roster for `--with-cloud` (SPEC §5)."""
    return [
        ModelInfo(cfg.cloud.anthropic_model, cfg.cloud.anthropic_model,
                  "cloud", model_class="cloud"),
        ModelInfo(cfg.cloud.openai_model, cfg.cloud.openai_model,
                  "cloud", model_class="cloud"),
    ]


async def execute_run(
    cfg: Config,
    store: StateStore,
    servers: list[str],
    with_cloud: bool = False,
    chat_fn: Any = None,
) -> RunArtifacts:
    """Execute a full `umb-validator run` (SPEC §9).

    Steps: discover roster -> fetch canonical defs -> build universe ->
    seed tool rows -> schedule pipelines -> write reports.

    `chat_fn` may be injected (tests); otherwise it is built from the gateway
    base_url. Resumable: tools already in a terminal state are skipped, and
    each tool resumes from its current state.
    """
    run_id = new_run_id()
    log.info("runner.run_start", run_id=run_id, servers=servers,
             with_cloud=with_cloud)
    gw, roster, jury, generator = await discover_and_build_roster(cfg)
    if chat_fn is None:
        chat_fn = make_chat_fn(cfg.gateway.base_url)

    canonical = await fetch_canonical_tools(cfg, servers)
    universe = ToolUniverse.from_canonical(canonical)

    pool = SessionPool(cfg)
    llm = LLMClient(pool, chat_fn, max_retries=5,
                    chat_template_kwargs=cfg.gateway.chat_template_kwargs)
    research = build_research_client(cfg)
    cloud_models = build_cloud_models(cfg) if with_cloud else []

    deps = PipelineDeps(
        config=cfg, store=store, gateway=gw, pool=pool, llm=llm,
        research=research, universe=universe, roster=roster,
        jury_models=jury, generator=generator, cloud_models=cloud_models)

    # Seed tool rows + build contexts.
    contexts: list[tuple[ToolContext, State]] = []
    for server, tools in canonical.items():
        for ct in tools:
            store.upsert_tool(server, ct.name, status=State.PENDING)
            cur = store.current_state(server, ct.name) or State.PENDING
            from umb_validator.states import is_terminal
            if is_terminal(cur) and cur != State.REJECTED:
                continue  # MERGED / HASH_COMPUTED / NEEDS_MANUAL — skip.
            ctx = ToolContext(
                server=server, tool=ct.name, canonical=ct,
                upstream_source=_SEED_UPSTREAM.get(server))
            contexts.append((ctx, cur))

    store.start_run(run_id, with_cloud, len(contexts))
    import time
    t0 = time.monotonic()
    scheduler = Scheduler(deps, run_id, with_cloud)
    final_states = await scheduler.run(contexts)
    wall = time.monotonic() - t0

    summary = build_run_summary(
        store, run_id, final_states, [m.gateway_id for m in roster],
        with_cloud, wall)
    runs_dir = cfg.paths.state_path() / "runs"
    json_path, md_path = write_run_reports(summary, runs_dir)
    store.finish_run(run_id, "complete", summary.to_dict())
    log.info("runner.run_complete", run_id=run_id,
             passed=summary.tools_passed, needs_manual=summary.tools_needs_manual)
    return RunArtifacts(run_id, final_states, json_path, md_path)
