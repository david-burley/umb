"""Owner-workflow operations: setup, merge, reject. SPEC v2 §9 / §11.

These back the CLI subcommands. The merge/reject ops mutate the working tree
ONLY via the explicit `merge` path; they NEVER commit, NEVER branch.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

from umb_validator.config import SEED_SERVERS, Config
from umb_validator.integration.umb_dev import (
    default_seed_registry, write_servers_json,
)
from umb_validator.logging_setup import get_logger
from umb_validator.states import State
from umb_validator.store import StateStore
from umb_validator.toml_io import merge_pending_into_live

log = get_logger("operations")


@dataclass
class SetupResult:
    """Outcome of `umb-validator setup`."""

    state_dir: Path
    servers_json: Path
    log_topic: str
    alert_topic: str
    config_card: str


def setup(cfg: Config) -> SetupResult:
    """Initialize the harness's working dirs + ntfy topics (SPEC §11 bootstrap).

    Idempotent: re-running `setup` reuses existing dirs and only generates
    ntfy topics if they are not already configured. Does NOT create the
    `umbvalidator` user or install systemd — those are documented manual
    steps in the printed config card (they need root).
    """
    state_dir = cfg.paths.state_path()
    home_dir = state_dir / "umb-home"
    runs_dir = state_dir / "runs"
    for d in (state_dir, home_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    servers_json = home_dir / "servers.json"
    if not servers_json.exists():
        write_servers_json(servers_json, default_seed_registry(SEED_SERVERS))

    log_topic = cfg.ntfy.log_topic or f"umb-validator-log-{secrets.token_hex(4)}"
    alert_topic = (cfg.ntfy.alert_topic
                   or f"umb-validator-alert-{secrets.token_hex(4)}")

    card = (
        "=== umb-validator setup ===\n"
        f"State dir:     {state_dir}\n"
        f"Server registry: {servers_json} ({len(SEED_SERVERS)} seed servers)\n"
        f"ntfy log topic:   {log_topic}\n"
        f"ntfy alert topic: {alert_topic}\n\n"
        "NEXT STEPS (manual, need root / one-time):\n"
        "  1. Add the two ntfy topics above to the phone app.\n"
        "  2. Put the topics into /etc/umb-validator/config.toml [ntfy].\n"
        "  3. Provision /etc/umb-validator/license.key (umb-dev license).\n"
        "  4. Install systemd unit: validator/systemd/umb-validator.service\n"
        "  5. `systemctl enable --now umb-validator`\n"
    )
    log.info("operations.setup_done", state_dir=str(state_dir))
    return SetupResult(state_dir, servers_json, log_topic, alert_topic, card)


@dataclass
class MergeResult:
    """Outcome of `umb-validator merge`."""

    merged_tools: list[str]
    live_path: Path
    pending_removed: bool
    error: str | None = None


def merge(
    cfg: Config, store: StateStore, target: str,
) -> MergeResult:
    """Promote `_pending/<server>.toml` -> `tool-dictionary/<server>.toml`.

    `target` is either `<server>` (merge all that server's pending entries) or
    `<server>.<tool>` (merge one entry). Comment-preserving; leaves the
    working tree UNCOMMITTED (SPEC §9). When a whole server file is fully
    consumed, the `_pending` file is removed.
    """
    server, _, tool = target.partition(".")
    pending_dir = cfg.resolve_pending_dir()
    dict_dir = cfg.resolve_dict_dir()
    pending_path = pending_dir / f"{server}.toml"
    live_path = dict_dir / f"{server}.toml"
    if not pending_path.is_file():
        return MergeResult([], live_path, False,
                           error=f"no _pending file for server '{server}'")

    only_tool = tool or None
    try:
        merged = merge_pending_into_live(pending_path, live_path, only_tool)
    except Exception as exc:  # noqa: BLE001
        return MergeResult([], live_path, False, error=str(exc))

    # Mark merged tools + their pending diffs.
    for t in merged:
        store.upsert_tool(server, t, status=State.MERGED)
        store.record_event(server, t, State.MERGED, {"merged_into": str(live_path)})
        diff = store.latest_pending_diff(server, t)
        if diff is not None and diff["reviewed_at"] is None:
            store.review_pending_diff(diff["id"], "merged")

    # If a single-tool merge, leave _pending intact for remaining tools.
    pending_removed = False
    if only_tool is None:
        try:
            pending_path.unlink()
            pending_removed = True
        except OSError:
            pass
    else:
        # Whole-file consumed iff every pending entry is now merged.
        from umb_validator.toml_io import parse_dict_file
        remaining = parse_dict_file(pending_path).tools
        all_merged = all(
            store.current_state(server, e.name) == State.MERGED
            for e in remaining
        )
        if all_merged and remaining:
            try:
                pending_path.unlink()
                pending_removed = True
            except OSError:
                pass

    log.info("operations.merged", server=server, tools=merged,
             pending_removed=pending_removed)
    return MergeResult(merged, live_path, pending_removed)


@dataclass
class RejectResult:
    """Outcome of `umb-validator reject`."""

    server: str
    tool: str
    reason: str
    error: str | None = None


def reject(
    cfg: Config, store: StateStore, target: str, reason: str,
) -> RejectResult:
    """Mark a REVIEW_READY proposal REJECTED with an owner reason (SPEC §9).

    `target` must be `<server>.<tool>`. The reason is recorded on the pending
    diff + the state event, and is fed into the shortener on a re-run retry.
    """
    server, _, tool = target.partition(".")
    if not tool:
        return RejectResult(server, "", reason,
                            error="reject target must be '<server>.<tool>'")
    cur = store.current_state(server, tool)
    if cur != State.REVIEW_READY:
        return RejectResult(server, tool, reason,
                            error=f"tool is in state {cur}, not REVIEW_READY")
    store.record_event(server, tool, State.REJECTED, {"reason": reason})
    store.upsert_tool(server, tool, status=State.REJECTED)
    diff = store.latest_pending_diff(server, tool)
    if diff is not None and diff["reviewed_at"] is None:
        store.review_pending_diff(diff["id"], "rejected", reason=reason)
    log.info("operations.rejected", server=server, tool=tool, reason=reason)
    return RejectResult(server, tool, reason)
