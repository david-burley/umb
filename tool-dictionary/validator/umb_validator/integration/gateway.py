"""AI Gateway client + session pool. SPEC v2 §1 (concurrency) / §5 / §12.

The harness reaches local models through an OpenAI-compatible gateway
(e.g. `http://localhost:8000/v1` — vLLM or an AI gateway; see
`config.toml [gateway].base_url`). At startup it queries `/v1/models` and
builds a roster of whatever local models are currently served — no hardcoded
list.

Config model names (e.g. `qwen3.5-35b-a3b`) are FUZZY-resolved against the
gateway's actual ids (e.g. `lmstudio-community/Qwen3.5-35B-A3B-GGUF`,
`Qwen/Qwen3.5-4B`, `glm-4.7-flash`), because the gateway exposes vendor-
qualified ids that vary by backend.

A per-backend session semaphore enforces each inference host's concurrency
ceiling even when many models share a box.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from umb_validator.config import Config
from umb_validator.logging_setup import get_logger

log = get_logger("gateway")


def _normalize(name: str) -> str:
    """Lowercase, drop vendor prefix + common suffixes, strip non-alphanum.

    `lmstudio-community/Qwen3.5-35B-A3B-GGUF` -> `qwen3535ba3b`
    `qwen3.5-35b-a3b`                          -> `qwen3535ba3b`
    """
    base = name.split("/")[-1].lower()
    for suffix in ("-gguf", "-mlx", "-fp8", "-4bit", "-8bit", "-instruct"):
        base = base.replace(suffix, "")
    return re.sub(r"[^a-z0-9]", "", base)


def resolve_model_id(wanted: str, available: list[str]) -> str | None:
    """Resolve a config model name against the gateway's actual ids.

    Exact match wins; else normalized exact; else the shortest available id
    whose normalized form contains, or is contained by, the wanted one.
    Returns None if nothing plausibly matches.
    """
    if wanted in available:
        return wanted
    w = _normalize(wanted)
    norm = {a: _normalize(a) for a in available}
    exact = [a for a, n in norm.items() if n == w]
    if exact:
        return min(exact, key=len)
    partial = [a for a, n in norm.items() if w and (w in n or n in w)]
    if partial:
        return min(partial, key=len)
    return None


@dataclass
class ModelInfo:
    """A resolved roster model."""

    config_name: str  # the name as written in config
    gateway_id: str  # the id the gateway actually accepts
    backend: str  # 'backend-a' | 'backend-b' | 'backend-c' | 'cloud'
    model_class: str = "local"  # 'local' | 'cloud'


# Substring -> backend heuristics for per-backend semaphore keying.
# The gateway does not expose backend in /v1/models, so we infer from the id.
_BACKEND_HINTS: list[tuple[str, str]] = [
    ("35b-a3b", "backend-a"),
    ("35ba3b", "backend-a"),
    ("122b", "backend-a"),
    ("4b", "backend-b"),
    ("0.8b", "backend-b"),
    ("2b", "backend-b"),
    ("glm-4.7", "backend-c"),
    ("glm47", "backend-c"),
]


def infer_backend(gateway_id: str) -> str:
    """Best-effort backend inference from a gateway model id (SPEC §1 table).

    Unknown ids default to `backend-a` (the largest pool) — conservative:
    worst case the harness under-utilizes a backend, never over-saturates a
    small one, because the aggregate cap also applies.
    """
    norm = _normalize(gateway_id)
    for hint, backend in _BACKEND_HINTS:
        if _normalize(hint) in norm or hint in gateway_id.lower():
            return backend
    return "backend-a"


class SessionPool:
    """Per-backend + aggregate session semaphores (SPEC v2 §1).

    `acquire(backend)` blocks until both the backend-specific semaphore and
    the aggregate local cap have a free slot, so the harness never exceeds a
    backend host's session ceiling.
    """

    def __init__(self, cfg: Config) -> None:
        sp = cfg.session_pool
        self._aggregate = asyncio.Semaphore(sp.max_inflight_local)
        self._per_backend: dict[str, asyncio.Semaphore] = {
            "backend-a": asyncio.Semaphore(sp.backend_a),
            "backend-b": asyncio.Semaphore(sp.backend_b),
            "backend-c": asyncio.Semaphore(sp.backend_c),
        }
        self._caps: dict[str, int] = {
            "backend-a": sp.backend_a,
            "backend-b": sp.backend_b,
            "backend-c": sp.backend_c,
        }
        self._cloud = asyncio.Semaphore(6)  # MAX_INFLIGHT_CLOUD

    def _sem_for(self, backend: str) -> asyncio.Semaphore:
        return self._per_backend.get(backend, self._per_backend["backend-a"])

    class _Slot:
        """Async context manager for one held inference slot."""

        def __init__(self, pool: "SessionPool", backend: str, cloud: bool):
            self._pool = pool
            self._backend = backend
            self._cloud = cloud

        async def __aenter__(self) -> "SessionPool._Slot":
            if self._cloud:
                await self._pool._cloud.acquire()
            else:
                await self._pool._aggregate.acquire()
                await self._pool._sem_for(self._backend).acquire()
            return self

        async def __aexit__(self, *exc: object) -> None:
            if self._cloud:
                self._pool._cloud.release()
            else:
                self._pool._sem_for(self._backend).release()
                self._pool._aggregate.release()

    def slot(self, backend: str, cloud: bool = False) -> "SessionPool._Slot":
        """Acquire one inference slot for `backend` (or a cloud slot)."""
        return SessionPool._Slot(self, backend, cloud)

    def utilization(self) -> dict[str, str]:
        """Human-readable per-backend slot utilization for `status`."""
        out: dict[str, str] = {}
        for backend, sem in self._per_backend.items():
            cap = self._caps[backend]
            free = sem._value  # noqa: SLF001 — read-only introspection
            out[backend] = f"{cap - free}/{cap}"
        return out


@dataclass
class GatewayClient:
    """OpenAI-compatible gateway client + roster discovery.

    The actual chat completion calls go through the `openai` SDK pointed at
    `base_url`; this class owns discovery + roster resolution. The LLM-call
    layer (`llm.py`) consumes the resolved roster.
    """

    cfg: Config
    roster: list[ModelInfo] = field(default_factory=list)
    available_ids: list[str] = field(default_factory=list)

    async def discover(self, timeout: float = 10.0) -> list[str]:
        """Query `GET /v1/models`; returns the raw id list. Network failure
        raises — the caller decides whether to proceed with config pins."""
        url = self.cfg.gateway.base_url.rstrip("/") + "/models"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        ids = [m["id"] for m in data.get("data", []) if "id" in m]
        self.available_ids = ids
        log.info("gateway.discovered", n=len(ids))
        return ids

    def build_roster(self, available: list[str] | None = None) -> list[ModelInfo]:
        """Resolve the local roster from discovered ids + config.

        - `model_pin` non-empty -> use ONLY those (resolved).
        - else use all discovered ids, minus `model_exclude`.
        The roster is intersected with config intent and each model is keyed
        to a backend for the session pool.
        """
        ids = available if available is not None else self.available_ids
        gw = self.cfg.gateway
        roster: list[ModelInfo] = []
        if gw.model_pin:
            for pinned in gw.model_pin:
                resolved = resolve_model_id(pinned, ids)
                if resolved is None:
                    log.warning("gateway.pin_unresolved", wanted=pinned)
                    continue
                roster.append(ModelInfo(pinned, resolved,
                                        infer_backend(resolved)))
        else:
            excludes = {_normalize(e) for e in gw.model_exclude}
            for gid in ids:
                if _normalize(gid) in excludes:
                    continue
                roster.append(ModelInfo(gid, gid, infer_backend(gid)))
        self.roster = roster
        return roster

    def resolve_jury(
        self, jury_models: list[str], available: list[str] | None = None
    ) -> list[ModelInfo]:
        """Resolve the §3.3 jury model names to ModelInfo against the gateway.

        Unresolvable jury members are dropped (logged) — the quorum N adapts
        to whatever is actually serveable, matching SPEC §5's 'a model that
        fails to respond is excluded from N'."""
        ids = available if available is not None else self.available_ids
        out: list[ModelInfo] = []
        for name in jury_models:
            resolved = resolve_model_id(name, ids)
            if resolved is None:
                log.warning("gateway.jury_unresolved", wanted=name)
                continue
            out.append(ModelInfo(name, resolved, infer_backend(resolved)))
        return out

    def resolve_generator(
        self, available: list[str] | None = None
    ) -> ModelInfo | None:
        """Resolve the auto-research generator model (SPEC §3.2)."""
        ids = available if available is not None else self.available_ids
        resolved = resolve_model_id(self.cfg.gateway.generator_model, ids)
        if resolved is None:
            # Fall back to the first roster model.
            if self.roster:
                return self.roster[0]
            return None
        return ModelInfo(self.cfg.gateway.generator_model, resolved,
                         infer_backend(resolved))
