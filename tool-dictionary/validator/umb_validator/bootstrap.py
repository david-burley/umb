"""bootstrap-existing-15 — pure hash-stamping. SPEC v2 §5 / §9.

NO LLM, NO research, NO shortening. For each shipped dict TOML:
  1. spawn the upstream MCP server via umb-dev,
  2. capture the live `description` for each `[[tools]]` entry,
  3. compute sha256_hex(description.as_bytes()),
  4. write `schema_hash_sha256` into the TOML in-place (comment-preserving),
  5. leave the working tree dirty — the owner commits.

This transitions the 15 shipped TOMLs from "Auto≡On forever" to "Auto fully
active." ~30 min wall-time, $0 cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from umb_validator.config import SEED_SERVERS_SHIPPED, Config
from umb_validator.hashing import description_hash
from umb_validator.integration.umb_dev import (
    UmbDevError, UmbDevSession, default_seed_registry, write_servers_json,
)
from umb_validator.logging_setup import get_logger
from umb_validator.toml_io import bootstrap_stamp_file, parse_dict_file

log = get_logger("bootstrap")


@dataclass
class BootstrapResult:
    """Outcome of a bootstrap-existing-15 run."""

    files_processed: int = 0
    tools_stamped: int = 0
    tools_missing: list[str] = field(default_factory=list)
    servers_failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human summary."""
        return (f"{self.files_processed} files, {self.tools_stamped} tools "
                f"stamped, {len(self.tools_missing)} tools unresolved, "
                f"{len(self.servers_failed)} servers failed to spawn")


async def bootstrap_existing_15(
    config: Config, servers: list[str] | None = None,
) -> BootstrapResult:
    """Run the hash-stamping bootstrap over the shipped seed TOMLs.

    `servers` defaults to the 15 shipped seed servers. Each server is spawned
    once via umb-dev; its tools' live descriptions are hashed and stamped into
    the corresponding `tool-dictionary/<server>.toml`.
    """
    servers = servers or list(SEED_SERVERS_SHIPPED)
    dict_dir = config.resolve_dict_dir()
    state_dir = config.paths.state_path()
    home_dir = state_dir / "umb-home"
    home_dir.mkdir(parents=True, exist_ok=True)
    servers_json = home_dir / "servers.json"
    result = BootstrapResult()
    header = (f"Hashes back-filled "
              f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')} by "
              f"umb-validator bootstrap-existing-15; review with git diff "
              f"and commit.")

    for server in servers:
        toml_path = dict_dir / f"{server}.toml"
        if not toml_path.is_file():
            log.warning("bootstrap.toml_missing", server=server,
                        path=str(toml_path))
            result.servers_failed.append(server)
            continue
        # Write a single-server registry for this spawn.
        write_servers_json(servers_json,
                           default_seed_registry([server]))
        try:
            async with UmbDevSession(
                config.paths.umb_dev_bin, home_dir, servers_json,
            ) as sess:
                tools = await sess.list_tools(server)
        except UmbDevError as exc:
            log.error("bootstrap.spawn_failed", server=server, error=str(exc))
            result.servers_failed.append(server)
            continue

        live_by_name = {t.name: t.description for t in tools}
        parsed = parse_dict_file(toml_path)
        hashes: dict[str, str] = {}
        for entry in parsed.tools:
            live_desc = live_by_name.get(entry.name)
            if live_desc is None:
                result.tools_missing.append(f"{server}.{entry.name}")
                log.warning("bootstrap.tool_unresolved", server=server,
                            tool=entry.name)
                continue
            hashes[entry.name] = description_hash(live_desc)

        stamped, missing = bootstrap_stamp_file(toml_path, hashes, header)
        result.files_processed += 1
        result.tools_stamped += stamped
        for m in missing:
            result.tools_missing.append(f"{server}.{m}")
        log.info("bootstrap.file_done", server=server, stamped=stamped)

    log.info("bootstrap.complete", summary=result.summary())
    return result


def bootstrap_stamp_from_descriptions(
    toml_path: str | Path, live_descriptions: dict[str, str],
    header: str | None = None,
) -> tuple[int, list[str]]:
    """Stamp a single TOML from a known {tool: live_description} map.

    Pure (no umb-dev) — used by the test suite and by callers that have
    already captured descriptions. Returns (n_stamped, [missing])."""
    hashes = {name: description_hash(desc)
              for name, desc in live_descriptions.items()}
    hdr = header or (
        f"Hashes back-filled "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')} by "
        f"umb-validator bootstrap-existing-15; review with git diff and commit."
    )
    return bootstrap_stamp_file(toml_path, hashes, hdr)
