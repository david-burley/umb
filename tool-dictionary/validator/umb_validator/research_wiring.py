"""Research client wiring. SPEC v2 §3.1.

The research subsystem's web-doc phase needs WebSearch + WebFetch callables.
The harness runs headless on a server, so the web-doc phase uses plain
httpx fetches; web SEARCH has no headless equivalent and is left unwired by
default (research falls back to canonical schema + GitHub README, which is
the dominant grounding source). A future operator can inject a search
callable here.

The GitHub README/examples path (the primary grounding source) works with no
extra wiring — it is plain httpx in `ResearchClient`.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from umb_validator.config import Config
from umb_validator.subsystems.research import ResearchClient


async def _httpx_fetch(url: str) -> str:
    """Fetch a URL's body as text (best-effort)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as c:
        resp = await c.get(url)
        resp.raise_for_status()
        return resp.text


def build_research_client(
    cfg: Config,
    web_search_fn: Any = None,
    web_fetch_fn: Any = None,
) -> ResearchClient:
    """Build a ResearchClient.

    - GitHub README/examples grounding always works (plain httpx).
    - `web_fetch_fn` defaults to a plain httpx fetch.
    - `web_search_fn` defaults to None (no headless search) -> the web-doc
      breadth phase is skipped, which is a valid `schema_only`/README-only
      grounding outcome per SPEC §3.1.
    - `GITHUB_TOKEN` env var, if set, raises the GitHub API rate limit.
    """
    return ResearchClient(
        cfg.research,
        web_search_fn=web_search_fn,
        web_fetch_fn=web_fetch_fn or _httpx_fetch,
        github_token=os.environ.get("GITHUB_TOKEN"),
    )
