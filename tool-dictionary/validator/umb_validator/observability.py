"""Observability + alerting. SPEC v2 §8.

- ntfy push to two topics (log + alert) on the existing ntfy host.
- `status` / `status --json` rendering from the state store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from umb_validator.config import Config
from umb_validator.logging_setup import get_logger
from umb_validator.states import State
from umb_validator.store import StateStore

log = get_logger("observability")


class NtfyPublisher:
    """Pushes notifications to ntfy topics (SPEC §8).

    Two topics: a `log` topic (severity=low, silent on phone) and an `alert`
    topic (severity=high, breaks DND). Missing topic config -> push is a
    logged no-op (so the harness runs fine before `setup`).
    """

    def __init__(self, cfg: Config) -> None:
        self.host = cfg.ntfy.host.rstrip("/")
        self.log_topic = cfg.ntfy.log_topic
        self.alert_topic = cfg.ntfy.alert_topic

    def _post(self, topic: str, message: str, title: str,
              priority: str) -> bool:
        """POST a message to a topic. Returns True on success."""
        if not topic:
            log.info("ntfy.skipped", reason="topic_unconfigured", title=title)
            return False
        try:
            resp = httpx.post(
                f"{self.host}/{topic}",
                content=message.encode("utf-8"),
                headers={"Title": title, "Priority": priority},
                timeout=10.0,
            )
            resp.raise_for_status()
            log.info("ntfy.pushed", topic=topic, title=title)
            return True
        except httpx.HTTPError as exc:
            log.warning("ntfy.failed", topic=topic, error=str(exc))
            return False

    def log_event(self, message: str, title: str = "umb-validator") -> bool:
        """Push to the log topic (silent on phone)."""
        return self._post(self.log_topic, message, title, "low")

    def alert(self, message: str, title: str = "umb-validator ALERT") -> bool:
        """Push to the alert topic (breaks DND)."""
        return self._post(self.alert_topic, message, title, "high")


@dataclass
class StatusReport:
    """A snapshot of harness state for `umb-validator status`."""

    daemon_running: bool
    active_run_id: str | None
    roster: list[str]
    with_cloud: bool
    session_utilization: dict[str, str]
    cloud_cost_today: float
    counts_by_state: dict[str, int]
    review_ready: list[str]
    low_confidence: list[str]
    needs_manual: list[str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for `status --json`."""
        return {
            "daemon_running": self.daemon_running,
            "active_run_id": self.active_run_id,
            "roster": self.roster,
            "with_cloud": self.with_cloud,
            "session_utilization": self.session_utilization,
            "cloud_cost_today": self.cloud_cost_today,
            "counts_by_state": self.counts_by_state,
            "review_ready": self.review_ready,
            "low_confidence": self.low_confidence,
            "needs_manual": self.needs_manual,
        }


def build_status(
    store: StateStore,
    roster: list[str] | None = None,
    session_utilization: dict[str, str] | None = None,
    daemon_running: bool = False,
) -> StatusReport:
    """Assemble a StatusReport from the state store (SPEC §8)."""
    counts: dict[str, int] = {}
    review_ready: list[str] = []
    low_conf: list[str] = []
    needs_manual: list[str] = []
    for t in store.all_tools():
        server, tool = t["server_name"], t["tool_name"]
        st = store.current_state(server, tool)
        if st is None:
            continue
        counts[str(st)] = counts.get(str(st), 0) + 1
        key = f"{server}.{tool}"
        if st == State.REVIEW_READY:
            review_ready.append(key)
        elif st == State.NEEDS_MANUAL:
            needs_manual.append(key)
        # low-confidence flag lives in the latest state_event metadata.
        events = store.events_for(server, tool)
        if events:
            import json as _json
            meta_raw = events[-1]["metadata_json"]
            if meta_raw:
                meta = _json.loads(meta_raw)
                if meta.get("low_self_validation_confidence"):
                    low_conf.append(key)
    run = store.active_run() or store.latest_run()
    return StatusReport(
        daemon_running=daemon_running,
        active_run_id=run["run_id"] if run else None,
        roster=roster or [],
        with_cloud=bool(run["with_cloud"]) if run else False,
        session_utilization=session_utilization or {},
        cloud_cost_today=store.cloud_cost_today(),
        counts_by_state=counts,
        review_ready=review_ready,
        low_confidence=low_conf,
        needs_manual=needs_manual,
    )
