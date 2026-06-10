"""systemd-managed daemon. SPEC v2 §7.

The daemon is the `ExecStart` target of `umb-validator.service`. It:

1. Performs the per-run drift check (SPEC §2): re-fetches canonical defs for
   every MERGED tool, re-hashes, and resets drifted tools to PENDING.
2. Drains all PENDING + drift-detected + resumable work via a full
   `execute_run` (local-only by default — `--with-cloud` is a manual `run`).
3. Idles, re-checking for work on a fixed interval.

SIGTERM (sent by systemd at `RuntimeMaxSec`, or on `systemctl stop`) triggers
a clean checkpoint: the in-flight run finishes its current tools, the store is
flushed + closed, and the daemon exits 0. Because every pipeline step stamps a
`state_event` BEFORE inference dispatch, a restart resumes with no double-work.
"""

from __future__ import annotations

import asyncio
import signal

from umb_validator.config import SEED_SERVERS, Config
from umb_validator.hashing import description_hash
from umb_validator.integration.umb_dev import UmbDevError
from umb_validator.logging_setup import get_logger
from umb_validator.observability import NtfyPublisher
from umb_validator.runner import execute_run, fetch_canonical_tools
from umb_validator.states import State, is_terminal
from umb_validator.store import StateStore

log = get_logger("daemon")

# How long to sleep between work-drain passes when idle (seconds).
IDLE_POLL_SECONDS = 300


async def drift_check(cfg: Config, store: StateStore) -> list[str]:
    """Re-fetch canonical defs for MERGED tools, reset drifted ones to PENDING.

    SPEC §2 per-run drift check: if a MERGED tool's live canonical description
    hash differs from `tools.current_hash`, the tool re-enters the full
    pipeline. Returns the list of `server/tool` keys that drifted.

    A server that fails to spawn is logged + skipped (its tools are left
    untouched — the next pass retries).
    """
    merged = [t for t in store.all_tools()
              if store.current_state(t["server_name"], t["tool_name"])
              == State.MERGED]
    if not merged:
        return []
    servers = sorted({t["server_name"] for t in merged})
    drifted: list[str] = []
    try:
        canonical = await fetch_canonical_tools(cfg, servers)
    except UmbDevError as exc:
        log.error("daemon.drift_fetch_failed", error=str(exc))
        return []
    live_hash: dict[tuple[str, str], str] = {}
    for server, tools in canonical.items():
        for ct in tools:
            live_hash[(server, ct.name)] = description_hash(ct.description)
    for t in merged:
        key = (t["server_name"], t["tool_name"])
        new_hash = live_hash.get(key)
        if new_hash is None:
            continue  # server failed to spawn / tool renamed — skip.
        if t["current_hash"] and new_hash != t["current_hash"]:
            store.record_event(
                key[0], key[1], State.PENDING,
                {"reason": "upstream_drift", "old_hash": t["current_hash"],
                 "new_hash": new_hash})
            store.upsert_tool(key[0], key[1], status=State.PENDING,
                              current_hash=new_hash)
            drifted.append(f"{key[0]}/{key[1]}")
            log.info("daemon.drift_detected", server=key[0], tool=key[1],
                     old_hash=t["current_hash"], new_hash=new_hash)
    return drifted


def _has_pending_work(store: StateStore) -> bool:
    """True iff any tool is in a non-terminal, resumable state.

    A REJECTED tool is resumable (owner-feedback retry edge), so it counts.
    """
    for t in store.all_tools():
        st = store.current_state(t["server_name"], t["tool_name"])
        if st is None:
            continue
        if not is_terminal(st) or st == State.REJECTED:
            return True
    return False


async def run_daemon(cfg: Config, max_passes: int | None = None) -> None:
    """Daemon main loop (SPEC §7).

    `max_passes` bounds the number of drain passes — `None` means run forever
    (the production mode); a finite value is used by tests so the loop
    terminates. SIGTERM sets a stop flag honored between passes; the in-flight
    `execute_run` is allowed to finish (it self-checkpoints).
    """
    stop = asyncio.Event()
    ntfy = NtfyPublisher(cfg)

    def _on_sigterm() -> None:
        log.info("daemon.sigterm_received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_sigterm)
        except (NotImplementedError, ValueError):
            # add_signal_handler is unavailable on some platforms / threads.
            pass

    log.info("daemon.started", idle_poll_seconds=IDLE_POLL_SECONDS)
    passes = 0
    while not stop.is_set():
        store = StateStore(cfg.paths.state_path() / "state.sqlite")
        try:
            drifted = await drift_check(cfg, store)
            if drifted:
                ntfy.log_event(
                    f"drift detected on {len(drifted)} tools: "
                    f"{', '.join(drifted)}")
            if _has_pending_work(store) or drifted:
                log.info("daemon.draining_work")
                try:
                    artifacts = await execute_run(
                        cfg, store, list(SEED_SERVERS), with_cloud=False)
                    log.info("daemon.run_done", run_id=artifacts.run_id)
                except Exception as exc:  # noqa: BLE001 — keep the daemon alive
                    log.error("daemon.run_failed", error=str(exc),
                              exc_info=True)
                    ntfy.alert(f"validation run crashed: {exc}")
            else:
                log.info("daemon.idle", reason="no_pending_work")
        finally:
            store.close()

        passes += 1
        if max_passes is not None and passes >= max_passes:
            log.info("daemon.max_passes_reached", passes=passes)
            break
        if stop.is_set():
            break
        # Idle wait — interruptible by SIGTERM.
        try:
            await asyncio.wait_for(stop.wait(), timeout=IDLE_POLL_SECONDS)
        except asyncio.TimeoutError:
            pass

    log.info("daemon.stopped", passes=passes)
