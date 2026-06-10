"""Auto-research subsystem — Research phase. SPEC v2 §3.1.

Gathers grounding material into `research_artifacts`:
  1. canonical_schema  — the umb-dev tools/list {name, description, inputSchema}
  2. upstream_readme   — fetched from the server's upstream_canonical_source
  3. upstream_example  — examples/ snippets or README fenced blocks
  4. web_doc           — bounded web search (top-3), capped per tool

Network fetches are pinned (commit SHA / retrieval timestamp). A tool with
only the canonical schema proceeds with `grounding=schema_only`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from umb_validator.config import ResearchConfig
from umb_validator.integration.umb_dev import CanonicalTool
from umb_validator.logging_setup import get_logger

log = get_logger("research")


@dataclass
class Artifact:
    """One grounding artifact destined for `research_artifacts`."""

    kind: str  # canonical_schema | upstream_readme | upstream_example | web_doc
    content: str
    source_url: str | None = None
    source_pinned: str | None = None


def _trim(text: str, max_chars: int) -> str:
    """Trim a string to `max_chars`, on a word boundary where possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > max_chars * 0.8 else cut) + " …[trimmed]"


def canonical_schema_artifact(tool: CanonicalTool) -> Artifact:
    """Build the always-present canonical_schema artifact (SPEC §3.1.1)."""
    payload = {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }
    return Artifact(
        kind="canonical_schema",
        content=json.dumps(payload, indent=2),
        source_url=None,
        source_pinned=None,
    )


def parse_github_source(url: str) -> tuple[str, str, str] | None:
    """Parse a github tree URL into (owner, repo, path).

    `https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem`
      -> ('modelcontextprotocol', 'servers', 'src/filesystem')
    Returns None if the URL is not a recognizable github source.
    """
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/[^/]+/(.+))?/?$", url
    )
    if not m:
        return None
    owner, repo, path = m.group(1), m.group(2), m.group(3) or ""
    return owner, repo, path


class ResearchClient:
    """HTTP research client — GitHub README/examples + bounded web docs.

    `web_search_fn` and `web_fetch_fn` are injected so the harness can use
    whatever web tooling is available (and tests can mock them). Both are
    optional: if absent, the web_doc phase is skipped (schema_only is still
    a valid grounding outcome).
    """

    def __init__(
        self,
        cfg: ResearchConfig,
        web_search_fn: Any = None,
        web_fetch_fn: Any = None,
        github_token: str | None = None,
    ) -> None:
        self.cfg = cfg
        self._web_search = web_search_fn
        self._web_fetch = web_fetch_fn
        self.github_token = github_token

    async def _resolve_commit(
        self, client: httpx.AsyncClient, owner: str, repo: str
    ) -> str:
        """Resolve the current `main` commit SHA for reproducible pinning."""
        url = f"{self.cfg.github_api_base}/repos/{owner}/{repo}/commits/main"
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        try:
            resp = await client.get(url, headers=headers, timeout=15.0)
            resp.raise_for_status()
            return str(resp.json().get("sha", "unknown"))
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            return "unknown"

    async def _fetch_readme(
        self, client: httpx.AsyncClient, owner: str, repo: str,
        path: str, pinned: str,
    ) -> Artifact | None:
        """Fetch the README at a repo subpath via raw.githubusercontent.com."""
        ref = pinned if pinned != "unknown" else "main"
        for fname in ("README.md", "readme.md", "Readme.md"):
            sub = f"{path}/{fname}" if path else fname
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{sub}"
            try:
                resp = await client.get(raw, timeout=15.0)
                if resp.status_code == 200 and resp.text.strip():
                    return Artifact(
                        kind="upstream_readme",
                        content=_trim(resp.text, self.cfg.max_artifact_chars),
                        source_url=raw,
                        source_pinned=pinned,
                    )
            except httpx.HTTPError:
                continue
        return None

    @staticmethod
    def _extract_examples(readme_text: str) -> str | None:
        """Pull fenced code blocks from a README as an examples artifact."""
        blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", readme_text, re.DOTALL)
        if not blocks:
            return None
        return "\n\n---\n\n".join(b.strip() for b in blocks[:6])

    async def gather(
        self, tool: CanonicalTool, upstream_source: str | None,
    ) -> tuple[list[Artifact], str]:
        """Gather all grounding artifacts for a tool.

        Returns (artifacts, grounding) where `grounding` is 'full' or
        'schema_only'. The canonical_schema artifact is always present.
        """
        artifacts: list[Artifact] = [canonical_schema_artifact(tool)]
        got_docs = False

        if upstream_source:
            parsed = parse_github_source(upstream_source)
            if parsed is not None:
                owner, repo, path = parsed
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    pinned = await self._resolve_commit(client, owner, repo)
                    readme = await self._fetch_readme(
                        client, owner, repo, path, pinned)
                    if readme is not None:
                        artifacts.append(readme)
                        got_docs = True
                        examples = self._extract_examples(readme.content)
                        if examples:
                            artifacts.append(Artifact(
                                kind="upstream_example",
                                content=_trim(examples,
                                              self.cfg.max_artifact_chars),
                                source_url=readme.source_url,
                                source_pinned=pinned,
                            ))

        # Bounded web docs (optional breadth source).
        if self._web_search is not None and self._web_fetch is not None:
            try:
                web = await self._gather_web(tool)
                artifacts.extend(web)
                got_docs = got_docs or bool(web)
            except Exception as exc:  # noqa: BLE001 — web tooling is best-effort
                log.warning("research.web_failed", tool=tool.name, error=str(exc))

        grounding = "full" if got_docs else "schema_only"
        log.info("research.gathered", server=tool.server, tool=tool.name,
                 n_artifacts=len(artifacts), grounding=grounding)
        return artifacts, grounding

    async def _gather_web(self, tool: CanonicalTool) -> list[Artifact]:
        """Bounded web-doc retrieval (SPEC §3.1.4)."""
        query = f"{tool.server} mcp {tool.name} example"
        results = await self._web_search(query)
        urls = [r["url"] for r in (results or [])
                if isinstance(r, dict) and "url" in r]
        urls = urls[: self.cfg.max_web_docs_per_tool]
        out: list[Artifact] = []
        retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for url in urls:
            try:
                html = await self._web_fetch(url)
                text = (BeautifulSoup(html, "html.parser").get_text(" ")
                        if html else "")
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    out.append(Artifact(
                        kind="web_doc",
                        content=_trim(text, self.cfg.max_artifact_chars),
                        source_url=url,
                        source_pinned=retrieved,
                    ))
            except Exception as exc:  # noqa: BLE001
                log.warning("research.web_doc_failed", url=url, error=str(exc))
        return out


def grounding_summary(artifacts: list[Artifact]) -> str:
    """Concatenate artifacts into one grounding block for the generator LLM."""
    parts: list[str] = []
    for a in artifacts:
        header = f"### {a.kind}"
        if a.source_url:
            header += f"  (source: {a.source_url}, pinned: {a.source_pinned})"
        parts.append(f"{header}\n{a.content}")
    return "\n\n".join(parts)
