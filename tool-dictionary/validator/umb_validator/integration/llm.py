"""LLM call layer. SPEC v2 §3 / §4 / §7.

Two call shapes:
- `select_tool()` — forced tool-use: the model is given a tool list and must
  pick one (or none). Used by self-validation jury, baseline, gate runs.
  Exact-string-match on the returned tool name is the judge (SPEC §4).
- `complete()` — free-form chat: used by research synthesis, prompt
  generation, and the definition-shortener.

Backoff + retry per §7. Idempotency via the store's `stamp_llm_call`.
Local calls cost $0; cloud calls accrue to the cost ledger.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any

from umb_validator.integration.gateway import ModelInfo, SessionPool
from umb_validator.logging_setup import get_logger

log = get_logger("llm")

# Rough cloud pricing (USD per 1M tokens) for the ledger. Local = 0.
_CLOUD_PRICING: dict[str, tuple[float, float]] = {
    "sonnet": (3.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate a cloud call's USD cost. Local models return 0."""
    key = next((k for k in _CLOUD_PRICING if k in model.lower()), None)
    if key is None:
        return 0.0
    pin, pout = _CLOUD_PRICING[key]
    return (tokens_in / 1_000_000) * pin + (tokens_out / 1_000_000) * pout


@dataclass
class ToolPick:
    """Result of a forced tool-selection call.

    `forced_unavailable` is the §2 ungateable signal: it is True iff the
    backend REJECTED forced tool-use (HTTP 400 / tool-choice-unsupported) for
    this call. When True, `picked` is meaningless and MUST NOT be scored — the
    gate could not run on this model. This is explicitly DISTINCT from a
    legitimate model no-pick (`picked is None` with `forced_unavailable False`),
    which is a real, gradeable verdict.
    """

    picked: str | None  # tool name, or None if the model picked nothing
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str | None = None
    forced_unavailable: bool = False

    @property
    def gradeable(self) -> bool:
        """True iff this pick may be used for scoring / quorum.

        A forced-tool-use rejection or a hard error is NOT gradeable — it must
        not contribute a (fabricated) pass anywhere downstream.
        """
        return self.error is None and not self.forced_unavailable


def _is_forced_tool_use_rejection(exc: BaseException) -> bool:
    """True iff `exc` is a backend rejection of forced tool-use.

    The OpenAI-compatible gateway / vLLM / Ollama backends answer a
    `tool_choice="required"` request with HTTP 400 when they were not started
    with `--enable-auto-tool-choice` + a `--tool-call-parser`. The `openai`
    SDK surfaces this as `BadRequestError` (an `APIStatusError` with
    `status_code == 400`). We duck-type on `status_code` so a test mock can
    simulate a 400 without importing the SDK.

    A 400 is treated as a CAPABILITY signal (this model cannot be gated),
    NOT a transient error — so it is never retried and never silently masked.
    """
    status = getattr(exc, "status_code", None)
    if status == 400:
        return True
    # Some SDK paths expose the status on a nested `.response`.
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) == 400


@dataclass
class Completion:
    """Result of a free-form completion call."""

    text: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str | None = None


def build_tool_schema(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP Tool objects -> OpenAI function-calling tool schema."""
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema") or {
                    "type": "object", "properties": {}
                },
            },
        })
    return out


class LLMClient:
    """Async LLM client over the OpenAI-compatible gateway (+ optional cloud).

    `chat_fn` is injected so tests can supply a mock without network. In
    production it is bound to an `openai.AsyncOpenAI` client's
    `chat.completions.create`.

    `chat_template_kwargs` is an extra OpenAI-compatible request-body param
    (passed via the SDK's `extra_body`) carried on EVERY chat completion —
    both forced tool-use (`select_tool`) and free-form (`complete`). The
    default `{"enable_thinking": False}` keeps thinking-capable backends
    (e.g. a qwen3.5-122b) usable for forced tool-use: in thinking
    mode such a model reasons past `tool_choice="required"` and returns an
    empty `tool_calls=[]`. An empty dict omits the param entirely.
    """

    def __init__(
        self,
        pool: SessionPool,
        chat_fn: Any,
        max_retries: int = 5,
        base_backoff: float = 1.0,
        max_backoff: float = 32.0,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.pool = pool
        self._chat_fn = chat_fn
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.chat_template_kwargs: dict[str, Any] = dict(
            chat_template_kwargs or {})

    def _extra_body(self) -> dict[str, Any]:
        """Build the `extra_body` for an OpenAI-compatible chat request.

        `chat_template_kwargs` is not a first-class OpenAI param; the SDK
        forwards anything under `extra_body` verbatim into the JSON body, and
        vLLM/Ollama read `chat_template_kwargs` from there. Returns an empty
        dict (no `extra_body` sent) when no kwargs are configured.
        """
        if not self.chat_template_kwargs:
            return {}
        return {"extra_body": {
            "chat_template_kwargs": dict(self.chat_template_kwargs)}}

    async def _with_retry(self, coro_factory: Any, label: str) -> Any:
        """Run an awaitable factory with exponential backoff (1s..32s, ≤5).

        Transient errors retry; the last error is raised after exhaustion.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except Exception as exc:  # noqa: BLE001 — gateway/SDK error surface
                last_exc = exc
                # A forced-tool-use rejection (HTTP 400) is a capability
                # signal, not a transient fault — retrying cannot fix it.
                # Surface it immediately so select_tool can mark ungateable.
                if _is_forced_tool_use_rejection(exc):
                    raise
                if attempt >= self.max_retries:
                    break
                delay = min(
                    self.max_backoff,
                    self.base_backoff * (2 ** attempt),
                ) + random.uniform(0, 0.5)
                log.warning("llm.retry", label=label, attempt=attempt + 1,
                            delay=round(delay, 2), error=str(exc))
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def select_tool(
        self,
        model: ModelInfo,
        tools: list[dict[str, Any]],
        user_prompt: str,
    ) -> ToolPick:
        """Forced tool-use: the model must pick a tool from `tools`.

        Returns the picked tool name (exact string from the function call) or
        None. This is the §4 judge — no LLM-as-judge for selection.

        SPEC §4 mandates FORCED tool-use (`tool_choice` "required"/"any").
        `"auto"` is NOT forced. The harness sends `"required"` (the
        OpenAI-compatible forced value).

        §2 FAIL-SAFE: if a backend REJECTS forced tool-use (HTTP 400 —
        `--enable-auto-tool-choice` / `--tool-call-parser` not configured), the
        gate CANNOT run on that model. We do NOT fall back to an unforced call:
        an unforced completion returns no tool call, which is byte-identical to
        a legitimate no-pick and would let un-runnable gates fabricate a pass.
        Instead the result carries `forced_unavailable=True` so every
        downstream consumer (self-validation, benchmark, gate, pipeline) can
        tell "the gate could not run" apart from "the model declined to pick"
        and route the tool to NEEDS_MANUAL rather than LOCAL_GATE_PASS.
        """
        loop = asyncio.get_event_loop()
        is_cloud = model.model_class == "cloud"
        tool_schema = build_tool_schema(tools)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a tool-routing assistant. Given a user request "
                    "and a set of tools, call EXACTLY ONE tool that best "
                    "satisfies the request. If no tool fits, call none."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        async def _call(tool_choice: str | None) -> Any:
            kwargs: dict[str, Any] = dict(
                model=model.gateway_id,
                messages=messages,
                tools=tool_schema,
                max_tokens=256,
                temperature=0.0,
            )
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            kwargs.update(self._extra_body())
            return await self._chat_fn(**kwargs)

        start = loop.time()
        try:
            async with self.pool.slot(model.backend, cloud=is_cloud):
                resp = await self._with_retry(
                    lambda: _call("required"),
                    f"select_tool:{model.gateway_id}")
        except Exception as exc:  # noqa: BLE001
            latency_ms = (loop.time() - start) * 1000
            if _is_forced_tool_use_rejection(exc):
                # §2: the backend rejected forced tool-use. The gate CANNOT
                # run on this model. Do NOT fall back to an unforced call —
                # mark ungateable so no fabricated pass is possible downstream.
                log.warning("select_tool.forced_unavailable",
                            model=model.gateway_id, error=str(exc))
                return ToolPick(None, latency_ms, 0, 0, 0.0,
                                error="tool_choice_unsupported",
                                forced_unavailable=True)
            return ToolPick(None, latency_ms, 0, 0, 0.0, error=str(exc))
        latency_ms = (loop.time() - start) * 1000
        picked, tin, tout = _extract_tool_call(resp)
        cost = estimate_cost(model.gateway_id, tin, tout) if is_cloud else 0.0
        return ToolPick(picked, latency_ms, tin, tout, cost)

    async def complete(
        self,
        model: ModelInfo,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> Completion:
        """Free-form completion (research synthesis / generation / shortening)."""
        loop = asyncio.get_event_loop()
        is_cloud = model.model_class == "cloud"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        async def _call() -> Any:
            return await self._chat_fn(
                model=model.gateway_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **self._extra_body(),
            )

        start = loop.time()
        try:
            async with self.pool.slot(model.backend, cloud=is_cloud):
                resp = await self._with_retry(_call, f"complete:{model.gateway_id}")
        except Exception as exc:  # noqa: BLE001
            return Completion("", (loop.time() - start) * 1000, 0, 0, 0.0,
                              error=str(exc))
        latency_ms = (loop.time() - start) * 1000
        text, tin, tout = _extract_text(resp)
        cost = estimate_cost(model.gateway_id, tin, tout) if is_cloud else 0.0
        return Completion(text, latency_ms, tin, tout, cost)


def _resp_get(resp: Any, *path: str) -> Any:
    """Navigate an OpenAI-SDK object OR a plain dict (tests use dicts)."""
    cur: Any = resp
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _extract_tool_call(resp: Any) -> tuple[str | None, int, int]:
    """Pull the picked tool name + token counts from a chat response."""
    choices = _resp_get(resp, "choices")
    picked: str | None = None
    if choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else getattr(
            first, "message", None)
        tool_calls = (message.get("tool_calls") if isinstance(message, dict)
                      else getattr(message, "tool_calls", None))
        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function") if isinstance(tc, dict) else getattr(
                tc, "function", None)
            picked = (fn.get("name") if isinstance(fn, dict)
                      else getattr(fn, "name", None))
    tin = _resp_get(resp, "usage", "prompt_tokens") or 0
    tout = _resp_get(resp, "usage", "completion_tokens") or 0
    return picked, int(tin), int(tout)


def _extract_text(resp: Any) -> tuple[str, int, int]:
    """Pull the completion text + token counts from a chat response."""
    choices = _resp_get(resp, "choices")
    text = ""
    if choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else getattr(
            first, "message", None)
        content = (message.get("content") if isinstance(message, dict)
                   else getattr(message, "content", None))
        text = content or ""
    tin = _resp_get(resp, "usage", "prompt_tokens") or 0
    tout = _resp_get(resp, "usage", "completion_tokens") or 0
    return text, int(tin), int(tout)


def parse_json_block(text: str) -> Any:
    """Extract the first JSON value from an LLM response.

    Models often wrap JSON in ```json fences or prose; this finds the first
    balanced `[...]` or `{...}` and parses it. Raises ValueError on failure.
    """
    stripped = text.strip()
    # Strip code fences.
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1] if stripped.count("```") >= 2 \
            else stripped.lstrip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Find the first balanced bracket span.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no parseable JSON found in LLM response")
