# Agent-Harness Integration Guide

How to wire **Universal MCP Bridge (UMB)** into your AI coding agent.

UMB is a single native Rust binary (`umb`) that runs as a **stdio MCP server**:
your agent launches the binary and speaks JSON-RPC 2.0 over stdin/stdout. UMB
presents a compact API — three meta-tools (`list_tools`, `list_mcps`,
`route_mcp_call`) plus `get_tool_info` and built-in file/shell tools — and
routes every call to the MCP servers you register in `~/.umb/servers.json`.

> **Transport note (read this first).** UMB **serves stdio only.** The `sse`,
> `http`, and `streamable-http` entry types in `servers.json` describe the
> *backing* servers UMB connects to as a client — they are **not** ways to
> reach UMB itself. Any harness below connects to UMB by spawning the `umb`
> binary as a local stdio process. If a harness can only talk to *remote*
> HTTP/SSE MCP servers and cannot spawn a local binary, you must run UMB
> behind a stdio→HTTP adapter (e.g. an `mcp-proxy`-style bridge); that path is
> noted honestly per-harness where it applies.

---

## 30-second quickstart

```bash
# 1. Build UMB (or grab a prebuilt binary from the releases page)
git clone https://github.com/david-burley/umb
cd umb && cargo build --release          # → target/release/umb

# 2. Register the MCP servers you want UMB to bridge
mkdir -p ~/.umb
cat > ~/.umb/servers.json <<'JSON'
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home"]
    }
  }
}
JSON

# 3. Point your agent at the umb binary as a stdio MCP server (see below)
```

That's it. Your agent now sees UMB's small meta-tool surface instead of every
tool from every server, and reaches all of them through UMB.

---

## Step 0 — Register your MCP servers (`~/.umb/servers.json`)

Every harness below does the **same** thing: it launches `umb`. What UMB then
exposes is whatever you list in `~/.umb/servers.json`. This file uses the
standard MCP server format — the same shape as `.mcp.json` /
`claude_desktop_config.json`. The top-level key is `servers` (the alias
`mcpServers` is also accepted).

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_..." }
    },

    "_comment": "=== remote servers UMB connects to as a client ===",
    "some-sse-server":  { "type": "sse",  "url": "https://example.com/sse" },
    "some-http-server": { "type": "http", "url": "https://example.com/mcp" }
  }
}
```

Supported entry types (verified against `src/registry/config.rs`):

| `type`              | Fields                          | Notes |
|---------------------|---------------------------------|-------|
| _(absent)_ / `stdio`| `command`, `args`, `env`        | Default; UMB spawns the server as a child process |
| `sse`               | `type`, `url`, `env`            | Backing server reached over Server-Sent Events |
| `http` / `streamable-http` | `type`, `url`, `env`     | Backing server reached over HTTP |
| bare string value   | —                               | Treated as a JSON comment, ignored |

Each entry also accepts `"enabled": false` to keep it in the file but inactive.
Edit `servers.json` while UMB is running — changes **hot-swap** automatically,
no restart required. A bad/typo'd entry is skipped with a warning; the rest of
your config still loads (the file is never silently wiped).

---

## Claude Code

*Source of truth: Claude Code's `claude mcp add` CLI + project `.mcp.json`.*

**CLI (user scope — available in every project):**

```bash
claude mcp add umb -- /path/to/umb
```

The `--` separates Claude Code's flags from the command it will launch. Use the
absolute path to the binary (or just `umb` if it is on your `PATH`).

**Project scope (`.mcp.json` committed to a repo):**

```json
{
  "mcpServers": {
    "umb": {
      "command": "/path/to/umb"
    }
  }
}
```

Claude Desktop uses the same block in `claude_desktop_config.json`.

---

## opencode

*Source of truth: <https://opencode.ai/docs/mcp-servers/> — confirmed against a
local opencode install's `~/.config/opencode/opencode.json`.*

opencode configures MCP servers under the top-level `mcp` key. A local server
has `type: "local"` and a **`command` array** (binary first, then args):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "umb": {
      "type": "local",
      "command": ["umb"],
      "enabled": true
    }
  }
}
```

Add `"environment": { "KEY": "value" }` if UMB itself needs env vars (usually
not — per-server env belongs in `servers.json`). opencode starts the process
and talks stdio. Its `type: "remote"` (with `url`/`headers`) is for remote MCP
servers and does **not** apply to UMB, which is a local stdio binary.

---

## Cursor / Windsurf / generic `mcpServers` clients

*Source of truth: the standard `mcpServers` JSON block (Cursor `~/.cursor/mcp.json`,
project `.cursor/mcp.json`); Windsurf and most other clients use the same shape.
Confirmed against a local Cursor install's `~/.cursor/mcp.json`.*

```json
{
  "mcpServers": {
    "umb": {
      "command": "umb",
      "args": []
    }
  }
}
```

This is the most portable form. Any MCP client that accepts an `mcpServers`
object with `command`/`args` (Cursor, Windsurf, Cline, Zed, and others) can
launch UMB this way. Use an absolute path for `command` if `umb` is not on the
client's `PATH`.

---

## Hermes Agent (Nous Research)

*Source of truth: <https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp>.*

> **Naming note, stated honestly:** the agent product commonly referred to in
> this context is **Nous Research's Hermes Agent**. If you are running a
> self-hosted Hermes instance, the MCP configuration below is what its docs
> specify; verify the exact keys against the version you have installed.

Hermes reads MCP config from `~/.hermes/config.yaml` under `mcp_servers`, and
supports **stdio** (local subprocess) and HTTP transports. Register UMB as a
stdio server:

```yaml
mcp_servers:
  umb:
    command: "umb"
    args: []
    enabled: true
```

The docs also describe a `hermes mcp` CLI interactive picker for catalog
management, but direct YAML editing of `~/.hermes/config.yaml` is the primary
method. Because Hermes supports stdio, UMB runs directly — no HTTP adapter
needed. If your Hermes deployment is configured to reach *remote* MCP servers
only, run UMB behind a stdio→HTTP adapter and point Hermes at that URL with the
HTTP transport form (`url:` / `headers:`).

---

## OpenClaw

*Source of truth: <https://docs.openclaw.ai/cli/mcp>.*

OpenClaw spawns stdio MCP servers as child processes and stores definitions
under `mcp.servers` in `~/.openclaw/openclaw.json`. Add UMB with the CLI:

```bash
openclaw mcp add umb --command umb
```

`--arg` (repeatable), `--env`, and `--cwd` are available if you need them.
Equivalent config-file form:

```json
{
  "mcp": {
    "servers": {
      "umb": {
        "command": "umb",
        "args": []
      }
    }
  }
}
```

Verify the connection with `openclaw mcp doctor umb --probe`.

---

## Any other MCP client

If your client isn't listed, it almost certainly accepts one of two shapes:

1. an `mcpServers` (or `mcp`) object with `command` + `args`, or
2. a CLI `... mcp add` command that takes a `--command`.

In both cases, point it at the `umb` binary with no arguments. UMB needs no
flags to run as an MCP server — it speaks JSON-RPC 2.0 over stdio by default
(silent mode: protocol JSON on stdout, logs on stderr).

If a client can only connect to **remote** HTTP/SSE MCP servers and cannot
launch a local binary, wrap UMB with an `mcp-proxy`-style stdio→HTTP adapter
and register that adapter's URL. UMB itself does not serve HTTP/SSE.

---

## Verifying the connection

Once UMB is wired in, your agent should see exactly these tools:
`list_tools`, `get_tool_info`, `list_mcps`, `route_mcp_call`, plus UMB's
built-in file/shell tools. The typical flow:

1. `list_tools({ "query": "what you need" })` — discover tool names + short
   descriptions across all backing servers.
2. `get_tool_info({ "tool": "<name>" })` — fetch the full schema for the one
   tool you intend to call.
3. `route_mcp_call({ "tool": "<name>", "args": { ... } })` — execute it. Pass
   `"server"` to disambiguate if two backing servers export the same name.

You can also sanity-check your registry from the shell, independent of any
agent:

```bash
umb --list-servers
```

See the [README](../README.md) for the full CLI reference, daemon/proxy mode,
and the `umb --doctor` orphan-process scanner.
