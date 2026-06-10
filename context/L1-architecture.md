# L1-architecture.md: Universal MCP Bridge — Architecture

## Overview

Universal MCP Bridge (UMB) is a Rust command-line application that consolidates
many MCP (Model Context Protocol) servers behind a single unified MCP server.
The architecture is a layered design with clear separation across the CLI, the
MCP protocol/router, the optional daemon/proxy transport, the server registry,
and the optional feature implementations (semantic search, hot-swap).

- **Binary**: `umb`
- **Version**: 0.1.0
- **License**: PolyForm Noncommercial 1.0.0 (see `LICENSE`)
- **Key dependencies**: `tokio` (async runtime), `serde_json` (JSON-RPC),
  `clap` (CLI), `reqwest` (HTTP transport to remote MCP servers), `ort` +
  `tokenizers` (semantic search), `notify` (hot-swap)

This is the open release: there is **no license validation, no machine
fingerprinting, no encryption/caching of entitlements, no auto-update, no
tier system, no server limit, and no feature gating**. Every capability is
always available.

---

## Architectural Layers

### Layer 1: CLI Entry Point (`src/main.rs`)

A minimal async entry point that:

1. Initializes the logger (stderr-only, so stdout stays clean for MCP).
2. Parses CLI arguments via clap (`src/cli/args.rs`). `--version`/`--help`
   exit early, before signal handlers are installed (prevents zombies).
3. Routes info commands and run modes:
   - `--doctor [--clean] [--json] [--yes]` → scan/clean orphaned daemons
   - `--list-servers` → print configured servers and exit
   - `--daemon [--daemon-port N]` → run the daemon backend
   - `--proxy` → run a lightweight proxy that connects to (or starts) a daemon
   - default → start the MCP server in-process
4. Installs signal handling (`src/cli/signal.rs`) via a `CancellationToken`
   for graceful shutdown (SIGTERM/SIGINT).

Supported flags (full list): `--list-servers`, `-v/--verbose`, `--doctor`,
`--clean`, `--json`, `--yes`, `--daemon`, `--daemon-port`, `--proxy`,
`--search-threshold`, `--search-limit`. Silent mode (JSON-only on stdout) is
the default for MCP compliance; `--verbose` adds a banner and info output.

### Layer 2: MCP Server & Protocol (`src/server/`)

- `mcp.rs` — JSON-RPC 2.0 transport over stdio. Handles `initialize`,
  `tools/list`, `tools/call`, and the built-in meta-tools `list_tools`
  (optional semantic/substring query), `list_mcps`, `route_mcp_call`. Also
  exposes built-in file/shell tools directly.
- `router.rs` — `ToolRouter`: holds server configs and the tool registry,
  discovers tools from backing servers, dispatches calls, and applies
  semantic-search filtering in `list_tools()`. All discovered tools are
  registered; there is no tier-based filtering.

### Layer 3: Startup & Discovery (`src/startup/`)

- `server.rs` — `start_server_silent()` runs the MCP loop and spawns
  background tasks for discovery and hot-swap.
- `discovery.rs` — orchestrates tool discovery across transports in order:
  stdio (fast) → SSE/Docker → HTTP. Discovery runs in the background so it
  does not block server startup.
- `hot_swap.rs` — watches `~/.umb/servers.json` with `notify` and applies
  add/remove/update of servers live, without a restart.

### Layer 4: Daemon / Proxy (`src/daemon/`, `src/client/`)

Optional multi-client mode. `--daemon` runs a backend listening on a Unix
socket (`~/.umb/umb.sock`) or a TCP fallback (default port 19384). `--proxy`
runs a lightweight forwarder that connects to an existing daemon (starting one
if needed) and forwards stdin/stdout MCP traffic. Daemon state holds the
router and config (no license/auth state).

### Layer 5: Registry & Config (`src/registry/`, `src/config.rs`)

- `registry/config.rs` — parses `~/.umb/servers.json` (standard MCP server
  format, same shape as `.mcp.json` / `claude_desktop_config.json`). Every
  enabled server is active.
- `config.rs` — parses optional `~/.umb/config.toml` for general settings and
  semantic-search configuration (embedding backend, dimension, similarity
  threshold, max results, custom model/tokenizer paths, cache directory).

### Layer 6: Features (`src/features/`)

- `semantic_search.rs` — EmbeddingGemma ONNX inference with Matryoshka
  dimension support (128/256/512/768) for semantic tool discovery.
- `hot_swap.rs` — file-watch reload notifications for live config changes.

These features are always available; there is no entitlement check.

---

## Data / Control Flow

```
agent ──stdio JSON-RPC──▶ umb (mcp.rs)
                              │
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
       list_tools         list_mcps        route_mcp_call
            │                                    │
            ▼                                    ▼
   semantic/substring                    ToolRouter.call_tool()
   filter (router.rs)                            │
                                                 ▼
                              backing MCP server (stdio | HTTP | SSE)
```

Background tasks (spawned via `tokio::spawn`): tool discovery across servers,
and the hot-swap watcher on `servers.json`. Shared state is held in
`Arc<RwLock<_>>`.

## Configuration Locations

```
~/.umb/config.toml     # general + semantic-search settings (optional)
~/.umb/servers.json    # MCP server definitions
~/.umb/cache/          # model/cache directory (semantic search)
~/.umb/umb.sock        # daemon Unix socket (daemon mode only)
```

## Error Handling & Logging

- `anyhow::Result<T>` is used throughout for ergonomic error propagation.
- `tracing` / `tracing-subscriber` (env-filter) logs to stderr only, with
  context tags such as `[Main]`, `[Background]`, `[HotSwap]`.
