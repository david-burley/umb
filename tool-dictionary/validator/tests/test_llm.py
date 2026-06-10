"""LLM request-body wiring tests — `chat_template_kwargs` injection.

A thinking-capable backend (e.g. a qwen3.5-122b) reasons its way past
`tool_choice="required"` and returns an empty `tool_calls=[]`, making it
unusable as a gate/jury model. The empirically verified fix is to send
`chat_template_kwargs={"enable_thinking": false}` in the request body — which
the OpenAI SDK forwards verbatim under `extra_body`. These tests pin that the
harness injects it on every forced-tool-use (`select_tool`) and free-form
(`complete`) call, that it is config-driven, and that an empty config omits it.

GATEWAY-TOOLUSE-FINDINGS.md (2026-05-22) / config.gateway.chat_template_kwargs.
"""

from __future__ import annotations

from typing import Any

import pytest

from umb_validator.config import Config, GatewayConfig
from umb_validator.integration.gateway import ModelInfo, SessionPool
from umb_validator.integration.llm import LLMClient


class CapturingChat:
    """A mock chat_fn that records every request's kwargs and returns a
    valid OpenAI-shaped tool-call response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {
                "content": "ok",
                "tool_calls": [{"function": {"name": "read_file"}}],
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }


def _llm(chat: Any, chat_template_kwargs: dict[str, Any] | None) -> LLMClient:
    return LLMClient(SessionPool(Config()), chat, max_retries=0,
                     chat_template_kwargs=chat_template_kwargs)


_MODEL = ModelInfo("qwen3.5-122b", "qwen3.5-122b", "backend-a")
_TOOLS = [{"name": "read_file", "description": "Read a file"}]


# --- default: enable_thinking=false injected on the forced-tool-use path ----


@pytest.mark.asyncio
async def test_select_tool_injects_chat_template_kwargs_by_default() -> None:
    """With the config default in effect, the forced tool-selection request
    body carries `extra_body.chat_template_kwargs.enable_thinking == False`
    alongside `tool_choice="required"`."""
    chat = CapturingChat()
    # The shipped GatewayConfig default is {"enable_thinking": False}.
    llm = _llm(chat, GatewayConfig().chat_template_kwargs)

    await llm.select_tool(_MODEL, _TOOLS, "Read foo.txt")

    assert len(chat.calls) == 1
    req = chat.calls[0]
    assert req["tool_choice"] == "required"
    assert "extra_body" in req
    assert req["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False}


@pytest.mark.asyncio
async def test_complete_injects_chat_template_kwargs_by_default() -> None:
    """The free-form `complete()` path carries the same `extra_body`."""
    chat = CapturingChat()
    llm = _llm(chat, GatewayConfig().chat_template_kwargs)

    await llm.complete(_MODEL, "summarize this")

    assert len(chat.calls) == 1
    assert chat.calls[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False}


# --- config disabled: extra_body omitted entirely ---------------------------


@pytest.mark.asyncio
async def test_select_tool_omits_extra_body_when_config_disables_it() -> None:
    """An empty `chat_template_kwargs` config sends NO `extra_body` — the
    request body is unchanged from the pre-fix behaviour."""
    chat = CapturingChat()
    llm = _llm(chat, {})

    await llm.select_tool(_MODEL, _TOOLS, "Read foo.txt")

    assert len(chat.calls) == 1
    assert "extra_body" not in chat.calls[0]


@pytest.mark.asyncio
async def test_extra_body_overridable_per_deployment() -> None:
    """A deployment may override the kwargs; the override is forwarded
    verbatim (proves the knob is config-driven, not hardcoded)."""
    chat = CapturingChat()
    llm = _llm(chat, {"enable_thinking": True, "custom_flag": "x"})

    await llm.select_tool(_MODEL, _TOOLS, "Read foo.txt")

    assert chat.calls[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True, "custom_flag": "x"}


# --- config plumbing: the GatewayConfig default + TOML override -------------


def test_gateway_config_default_disables_thinking() -> None:
    """The shipped default is `enable_thinking = false` — so a from-defaults
    Config (no on-disk file) already makes thinking-capable models gateable."""
    assert Config().gateway.chat_template_kwargs == {"enable_thinking": False}


def test_gateway_config_chat_template_kwargs_overridable_via_toml(
    tmp_path: Any,
) -> None:
    """A partial config.toml can override `chat_template_kwargs`."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[gateway]\nchat_template_kwargs = { enable_thinking = true }\n")
    cfg = Config.load(cfg_file)
    assert cfg.gateway.chat_template_kwargs == {"enable_thinking": True}
