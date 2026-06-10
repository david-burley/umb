"""umb-validator CLI. SPEC v2 §9.

Subcommands: setup, run, daemon, status, pending-review, show, merge, reject,
bootstrap-existing-15, add-server.

The CLI is a thin shell over the operations / runner / bootstrap modules.
Heavy work is async; the CLI bridges with `asyncio.run`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from umb_validator import __version__
from umb_validator.config import SEED_SERVERS, Config
from umb_validator.integration.umb_dev import (
    default_seed_registry, write_servers_json,
)
from umb_validator.logging_setup import configure_logging
from umb_validator.observability import build_status
from umb_validator.states import State
from umb_validator.store import StateStore

app = typer.Typer(
    name="umb-validator",
    help="UMB tool-dictionary auto-research validation harness (SPEC v2).",
    add_completion=False,
)
console = Console()


def _load(config_path: Optional[str]) -> Config:
    """Load config (default-resolved) + configure logging."""
    cfg = Config.load(config_path)
    return cfg


def _store(cfg: Config) -> StateStore:
    """Open the SQLite state store at `~/.umb-validator/state.sqlite`."""
    return StateStore(cfg.paths.state_path() / "state.sqlite")


@app.command()
def version() -> None:
    """Print the harness version."""
    console.print(f"umb-validator {__version__}")


@app.command()
def setup(
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
    json_logs: bool = typer.Option(False, help="JSON log output"),
) -> None:
    """Initialize working dirs, server registry, ntfy topics (SPEC §11)."""
    configure_logging(json_output=json_logs)
    from umb_validator.operations import setup as do_setup
    cfg = _load(config)
    result = do_setup(cfg)
    console.print(result.config_card)


@app.command()
def run(
    tools: str = typer.Option("all", help="'all' or comma-separated servers"),
    with_cloud: bool = typer.Option(False, "--with-cloud",
                                    help="add the cloud corroboration gate"),
    strict_ci: bool = typer.Option(False, "--strict-ci",
                                   help="require lower 95%% CI to pass"),
    bootstrap_existing_15: bool = typer.Option(
        False, "--bootstrap-existing-15",
        help="pure hash-stamping subcommand (no LLM, no research)"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="plan only; print what would happen"),
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
    json_logs: bool = typer.Option(True, help="JSON log output"),
) -> None:
    """Run the validation pipeline (SPEC §9).

    `--bootstrap-existing-15` short-circuits to the pure hash-stamping path.
    """
    configure_logging(json_output=json_logs)
    cfg = _load(config)
    if bootstrap_existing_15:
        _do_bootstrap(cfg)
        return

    server_list = (list(SEED_SERVERS) if tools == "all"
                   else [s.strip() for s in tools.split(",") if s.strip()])
    if dry_run:
        console.print(f"[bold]DRY RUN[/bold] — would validate "
                      f"{len(server_list)} servers: "
                      f"{', '.join(server_list)}")
        console.print(f"  with_cloud={with_cloud}  strict_ci={strict_ci}")
        console.print(f"  jury quorum: {cfg.jury.quorum_q}/{cfg.jury.n}  "
                      f"local-K fraction: {cfg.gates.local_k_fraction}")
        console.print("  Estimated wall-clock: ~2-4h local-only "
                      "(SPEC §10).")
        return

    from umb_validator.runner import execute_run
    store = _store(cfg)
    try:
        artifacts = asyncio.run(
            execute_run(cfg, store, server_list, with_cloud=with_cloud))
        console.print(f"[green]Run {artifacts.run_id} complete.[/green]")
        console.print(f"  JSON report: {artifacts.json_report}")
        console.print(f"  MD report:   {artifacts.md_report}")
        ready = sum(1 for s in artifacts.final_states.values()
                    if s == State.REVIEW_READY)
        console.print(f"  {ready} tools REVIEW_READY "
                      f"(`umb-validator pending-review`).")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Run failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        store.close()


@app.command()
def daemon(
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
) -> None:
    """Run as a systemd-managed daemon (SPEC §7).

    Drains PENDING + drift-detected work, then idles. SIGTERM checkpoints
    cleanly. This is the `ExecStart` target of umb-validator.service.
    """
    configure_logging(json_output=True)
    cfg = _load(config)
    from umb_validator.daemon import run_daemon
    asyncio.run(run_daemon(cfg))


@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
) -> None:
    """Show harness state — roster, queue, REVIEW_READY, NEEDS_MANUAL (§8)."""
    cfg = _load(config)
    store = _store(cfg)
    try:
        report = build_status(store)
        if json_output:
            console.print_json(json.dumps(report.to_dict()))
            return
        console.print("[bold]umb-validator status[/bold]")
        run = store.latest_run()
        console.print(f"  Active run: {report.active_run_id or '(none)'}")
        console.print(f"  Cloud cost today: ${report.cloud_cost_today:.2f}")
        console.print("  State counts:")
        for st, n in sorted(report.counts_by_state.items()):
            console.print(f"    {st}: {n}")
        console.print(f"  REVIEW_READY: {len(report.review_ready)} "
                      f"-> {', '.join(report.review_ready) or '-'}")
        console.print(f"  LOW-CONF: {len(report.low_confidence)} "
                      f"-> {', '.join(report.low_confidence) or '-'}")
        console.print(f"  NEEDS_MANUAL: {len(report.needs_manual)} "
                      f"-> {', '.join(report.needs_manual) or '-'}")
    finally:
        store.close()


@app.command(name="pending-review")
def pending_review(
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
) -> None:
    """List REVIEW_READY proposals awaiting owner sign-off (SPEC §9)."""
    cfg = _load(config)
    store = _store(cfg)
    try:
        diffs = store.pending_diffs(unreviewed_only=True)
        if not diffs:
            console.print("No pending proposals.")
            return
        table = Table(title="Pending proposals")
        table.add_column("Tool")
        table.add_column("Proposed short_description")
        table.add_column("Created")
        for d in diffs:
            table.add_row(f"{d['server_name']}.{d['tool_name']}",
                          d["proposed_short"], d["created_at"])
        console.print(table)
    finally:
        store.close()


@app.command()
def show(
    target: str = typer.Argument(..., help="<server>.<tool>"),
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
) -> None:
    """Show a tool's proposal, oracle stats + per-model outcomes (SPEC §9)."""
    cfg = _load(config)
    store = _store(cfg)
    try:
        server, _, tool = target.partition(".")
        if not tool:
            console.print("[red]target must be '<server>.<tool>'[/red]")
            raise typer.Exit(1)
        cur = store.current_state(server, tool)
        console.print(f"[bold]{target}[/bold]  state={cur}")
        diff = store.latest_pending_diff(server, tool)
        if diff is not None:
            console.print(f"  proposed: {diff['proposed_short']}")
            console.print(f"  hash:     {diff['proposed_hash']}")
            console.print(f"  pending:  {diff['pending_path']}")
        runs = store.runs_for(server, tool)
        if runs:
            table = Table(title="Validation runs")
            table.add_column("iter")
            table.add_column("model")
            table.add_column("class")
            table.add_column("accuracy")
            table.add_column("tokens")
            for r in runs:
                table.add_row(str(r["iteration"]), r["model"],
                              r["model_class"], f"{r['accuracy']:.3f}",
                              str(r["token_count"]))
            console.print(table)
        oracle = store.prompts_for(server, tool, only_admitted=True)
        console.print(f"  oracle: {len(oracle)} admitted prompts")
        events = store.events_for(server, tool)
        for ev in events:
            console.print(f"  [dim]{ev['event_at']}[/dim] {ev['new_status']}")
    finally:
        store.close()


@app.command()
def merge(
    target: str = typer.Argument(..., help="<server> or <server>.<tool>"),
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
) -> None:
    """Promote a _pending proposal into the live dict TOML (SPEC §9).

    Working-tree-dirty: the merge writes the live TOML but does NOT commit.
    """
    cfg = _load(config)
    store = _store(cfg)
    try:
        from umb_validator.operations import merge as do_merge
        result = do_merge(cfg, store, target)
        if result.error:
            console.print(f"[red]merge failed: {result.error}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Merged {len(result.merged_tools)} entries "
                      f"into {result.live_path}[/green]")
        console.print(f"  tools: {', '.join(result.merged_tools)}")
        console.print("  Working tree is dirty — review with `git diff` "
                      "and commit when satisfied.")
    finally:
        store.close()


@app.command()
def reject(
    target: str = typer.Argument(..., help="<server>.<tool>"),
    reason: str = typer.Option(..., "--reason", help="why it was rejected"),
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
) -> None:
    """Reject a REVIEW_READY proposal with a reason (SPEC §9).

    The reason is fed into the shortener on the next re-run retry.
    """
    cfg = _load(config)
    store = _store(cfg)
    try:
        from umb_validator.operations import reject as do_reject
        result = do_reject(cfg, store, target, reason)
        if result.error:
            console.print(f"[red]reject failed: {result.error}[/red]")
            raise typer.Exit(1)
        console.print(f"[yellow]Rejected {target}[/yellow] — {reason}")
    finally:
        store.close()


@app.command(name="bootstrap-existing-15")
def bootstrap_existing_15_cmd(
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
    json_logs: bool = typer.Option(True, help="JSON log output"),
) -> None:
    """Hash-stamp the 15 shipped dict TOMLs (SPEC §5 / §9).

    Pure mechanical: spawns each upstream MCP server, hashes live
    descriptions, writes `schema_hash_sha256` in-place. Working-tree-dirty.
    """
    configure_logging(json_output=json_logs)
    cfg = _load(config)
    _do_bootstrap(cfg)


def _do_bootstrap(cfg: Config) -> None:
    """Shared bootstrap execution for `run --bootstrap-existing-15` and the
    standalone subcommand."""
    from umb_validator.bootstrap import bootstrap_existing_15
    result = asyncio.run(bootstrap_existing_15(cfg))
    console.print(f"[green]bootstrap-existing-15:[/green] {result.summary()}")
    if result.tools_missing:
        console.print(f"  [yellow]unresolved tools:[/yellow] "
                      f"{', '.join(result.tools_missing)}")
    if result.servers_failed:
        console.print(f"  [red]servers failed to spawn:[/red] "
                      f"{', '.join(result.servers_failed)}")
    console.print("  Working tree is dirty — review with `git diff` "
                  "and commit when satisfied.")


@app.command(name="add-server")
def add_server(
    name: str = typer.Argument(..., help="server name (e.g. notion)"),
    command: str = typer.Option("npx", help="spawn command"),
    args: str = typer.Option("", help="comma-separated spawn args"),
    config: Optional[str] = typer.Option(None, help="Path to config.toml"),
) -> None:
    """Add a server to the umb-dev registry (SPEC §2 / §11).

    Defaults to `npx -y @modelcontextprotocol/server-<name>` when no args
    are given.
    """
    cfg = _load(config)
    home_dir = cfg.paths.state_path() / "umb-home"
    home_dir.mkdir(parents=True, exist_ok=True)
    servers_json = home_dir / "servers.json"
    # Load existing registry, append.
    registry: dict[str, dict[str, Any]] = {}
    if servers_json.exists():
        existing = json.loads(servers_json.read_text(encoding="utf-8"))
        for s in existing.get("servers", []):
            registry[s["name"]] = {"command": s.get("command", "npx"),
                                   "args": s.get("args", [])}
    arg_list = ([a.strip() for a in args.split(",") if a.strip()]
                if args else ["-y", f"@modelcontextprotocol/server-{name}"])
    registry[name] = {"command": command, "args": arg_list}
    write_servers_json(servers_json, registry)
    console.print(f"[green]Added '{name}' to {servers_json}[/green]")
    console.print(f"  command: {command} {' '.join(arg_list)}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
