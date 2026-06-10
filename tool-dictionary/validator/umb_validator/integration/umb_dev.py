"""umb-dev MCP stdio integration. SPEC v2 §11.

Spawns `umb-dev` against a per-server registry, speaks JSON-RPC 2.0 over
stdio, captures the canonical `tools/list` array, then SIGTERMs the child.
One umb-dev process per SERVER (not per tool).

The harness writes a transient `servers.json` so umb-dev knows which upstream
MCP server to register; umb-dev itself spawns the upstream `npx`/`pip`
package. The parse logic is also exercised against a recorded fixture in the
test suite (no live umb-dev needed for that test).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umb_validator.logging_setup import get_logger

log = get_logger("umb_dev")

_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class CanonicalTool:
    """One tool as returned by an upstream MCP server's `tools/list`."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server: str = ""

    def as_tool_object(self) -> dict[str, Any]:
        """Render as a Tool object for prompt presentation (SPEC §3.4)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def parse_tools_list_result(result: dict[str, Any], server: str) -> list[CanonicalTool]:
    """Parse a JSON-RPC `tools/list` result object into CanonicalTool[].

    Pure function — unit-tested against a recorded fixture. Accepts the
    `result` payload of the JSON-RPC response (the object holding `tools`).
    """
    out: list[CanonicalTool] = []
    for raw in result.get("tools", []):
        name = raw.get("name")
        if not name:
            continue
        out.append(
            CanonicalTool(
                name=str(name),
                description=str(raw.get("description", "")),
                input_schema=raw.get("inputSchema") or {},
                server=server,
            )
        )
    return out


class UmbDevError(RuntimeError):
    """Raised when umb-dev fails to spawn / initialize / list tools."""


class UmbDevSession:
    """A spawned umb-dev process speaking MCP JSON-RPC over stdio.

    Usage::

        async with UmbDevSession(bin_path, home_dir, servers_json) as sess:
            tools = await sess.list_tools("filesystem")
    """

    def __init__(
        self,
        bin_path: str | Path,
        home_dir: str | Path,
        servers_json_path: str | Path,
        startup_timeout: float = 30.0,
    ) -> None:
        self.bin_path = str(bin_path)
        self.home_dir = str(home_dir)
        self.servers_json_path = str(servers_json_path)
        self.startup_timeout = startup_timeout
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id = 0

    async def __aenter__(self) -> "UmbDevSession":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def start(self) -> None:
        """Spawn umb-dev with an isolated HOME (SPEC §11 step 1)."""
        env = dict(os.environ)
        env["HOME"] = self.home_dir
        env["UMB_SERVERS_PATH"] = self.servers_json_path
        try:
            self.proc = await asyncio.create_subprocess_exec(
                self.bin_path,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            raise UmbDevError(f"failed to spawn umb-dev: {exc}") from exc
        log.info("umb_dev.spawned", bin=self.bin_path, pid=self.proc.pid)
        await self._initialize()

    async def _send(self, obj: dict[str, Any]) -> None:
        """Write one JSON-RPC message + newline to umb-dev stdin."""
        if self.proc is None or self.proc.stdin is None:
            raise UmbDevError("umb-dev process not started")
        line = (json.dumps(obj) + "\n").encode("utf-8")
        self.proc.stdin.write(line)
        await self.proc.stdin.drain()

    async def _read_response(self, expect_id: int, timeout: float) -> dict[str, Any]:
        """Read JSON-RPC lines until one with `id == expect_id` arrives.

        Notification lines (no `id`) and unrelated responses are skipped.
        """
        if self.proc is None or self.proc.stdout is None:
            raise UmbDevError("umb-dev process not started")
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise UmbDevError(f"timeout waiting for JSON-RPC id={expect_id}")
            try:
                raw = await asyncio.wait_for(
                    self.proc.stdout.readline(), timeout=remaining
                )
            except asyncio.TimeoutError as exc:
                raise UmbDevError(
                    f"timeout waiting for JSON-RPC id={expect_id}"
                ) from exc
            if not raw:
                raise UmbDevError("umb-dev closed stdout unexpectedly")
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                # Non-JSON banner / log line on stdout — skip.
                continue
            if isinstance(msg, dict) and msg.get("id") == expect_id:
                if "error" in msg:
                    raise UmbDevError(f"JSON-RPC error: {msg['error']}")
                result: dict[str, Any] = msg.get("result") or {}
                return result

    async def _initialize(self) -> None:
        """JSON-RPC initialize handshake (SPEC §11 steps 3-5)."""
        init_id = self._new_id()
        await self._send({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "umb-validator", "version": "0.1"},
            },
        })
        await self._read_response(init_id, self.startup_timeout)
        # notifications/initialized has no id, expects no response.
        await self._send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        log.info("umb_dev.initialized")

    async def list_tools(self, server: str, timeout: float = 60.0) -> list[CanonicalTool]:
        """Send `tools/list`, parse the canonical Tool[] array (SPEC §11 6-7)."""
        list_id = self._new_id()
        await self._send({
            "jsonrpc": "2.0",
            "id": list_id,
            "method": "tools/list",
            "params": {},
        })
        result = await self._read_response(list_id, timeout)
        tools = parse_tools_list_result(result, server)
        log.info("umb_dev.tools_listed", server=server, n=len(tools))
        return tools

    async def stop(self) -> None:
        """SIGTERM umb-dev + wait for clean exit (SPEC §11 step 8)."""
        if self.proc is None:
            return
        if self.proc.returncode is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                log.warning("umb_dev.sigkill", pid=self.proc.pid)
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
                await self.proc.wait()
        log.info("umb_dev.stopped", returncode=self.proc.returncode)
        self.proc = None


def write_servers_json(
    path: str | Path, servers: dict[str, dict[str, Any]]
) -> None:
    """Write the umb-dev server registry. `servers` maps server-name ->
    {command, args}. Atomic write (tmp + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": [
            {"name": name, "command": spec.get("command", "npx"),
             "args": spec.get("args", [])}
            for name, spec in servers.items()
        ]
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def default_seed_registry(servers: list[str]) -> dict[str, dict[str, Any]]:
    """Build a starter registry covering the seed servers via the official
    `npx -y @modelcontextprotocol/server-*` packages (SPEC §11 bootstrap)."""
    return {
        name: {
            "command": "npx",
            "args": ["-y", f"@modelcontextprotocol/server-{name}"],
        }
        for name in servers
    }
