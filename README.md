# Universal MCP Bridge (UMB)

**A [Feature Collective Investments](https://universalmcpbridge.app) company.**

[![CI](https://github.com/david-burley/umb/actions/workflows/ci.yml/badge.svg)](https://github.com/david-burley/umb/actions/workflows/ci.yml)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue.svg)](./LICENSE)

Universal MCP Bridge consolidates many Model Context Protocol (MCP) servers
behind a single unified MCP server. Instead of configuring dozens of MCP
servers directly in your AI coding agent, you configure them once in UMB, and
UMB presents a compact 3-tool API (plus built-in file/shell tools) to the
agent. This keeps the agent's tool list small and fast while still giving it
access to every backing server.

UMB is a single native Rust binary (`umb`) with no runtime dependencies.

## Features

- Native Rust implementation — single static binary, fast startup
- Unified MCP gateway over JSON-RPC 2.0 (stdio)
- Routes tool calls to any number of configured MCP servers (stdio, HTTP, SSE)
- Optional semantic tool search (EmbeddingGemma ONNX, Matryoshka dimensions)
- Live config hot-swap — edit `servers.json` while UMB is running
- Optional daemon/proxy mode for sharing one backend across multiple clients
- No server limits, no feature gating — every capability is always available

## The 3 Meta-Tools

UMB exposes a minimal API to the agent:

1. **`list_tools`** — enumerate available tools (optional semantic/substring query)
2. **`list_mcps`** — list connected MCP servers with their tool counts
3. **`route_mcp_call`** — execute any tool on any backing server by name

Built-in file/shell tools are also provided directly by UMB.

## Prerequisites

Install Rust via rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version && cargo --version
```

## Building

```bash
# Development build
cargo build

# Release build (optimized, LTO, stripped)
cargo build --release
```

The binary will be at `target/release/umb`. Typical binary size is 3–8 MB.

## Usage

UMB runs as an MCP server speaking JSON-RPC 2.0 over stdio. In silent mode
(the default) it emits only protocol JSON on stdout and logs to stderr, so it
is MCP-compliant out of the box.

### CLI

```
umb [OPTIONS]

Options:
      --list-servers                List configured MCP servers and exit
  -v, --verbose                     Show startup banner and verbose output
                                    (default: silent for MCP compliance)
      --doctor                      Scan for orphaned umb daemon processes
      --clean                       With --doctor: terminate orphaned daemons
                                    (default is a read-only scan)
      --json                        With --doctor: machine-readable JSON output
      --yes                         With --doctor --clean: skip confirmation
      --daemon                      Run as a daemon backend (multi-client mode);
                                    listens on ~/.umb/umb.sock or TCP fallback
      --daemon-port <PORT>          Port for the daemon TCP listener
                                    (default: 19384)
      --proxy                       Run as a lightweight proxy that connects to
                                    (or starts) a daemon and forwards MCP traffic
      --search-threshold <THRESHOLD>
                                    Minimum cosine similarity for semantic
                                    search, 0.0–1.0 (default: 0.7). Higher =
                                    fewer, more relevant results.
      --search-limit <LIMIT>        Max number of tools returned by list_tools
                                    (default: 10)
  -h, --help                        Print help
  -V, --version                     Print version
```

### Run as an MCP server (default)

```bash
./target/release/umb
```

Add UMB to your agent's MCP configuration the same way you would any stdio MCP
server (e.g. an entry in `.mcp.json` or `claude_desktop_config.json` pointing
at the `umb` binary).

### List configured servers

```bash
./target/release/umb --list-servers
```

### Daemon / proxy mode

Run one shared backend and connect lightweight proxies to it:

```bash
# Terminal 1: start the daemon backend
./target/release/umb --daemon

# Each agent: connect via a lightweight proxy
./target/release/umb --proxy
```

### Diagnose orphaned daemons

```bash
./target/release/umb --doctor            # read-only scan
./target/release/umb --doctor --clean    # terminate orphans (prompts first)
./target/release/umb --doctor --json     # machine-readable output
```

## Configuration

UMB reads configuration from `~/.umb/`:

| File | Purpose |
|------|---------|
| `~/.umb/config.toml`   | General + semantic-search settings (optional) |
| `~/.umb/servers.json`  | The MCP servers UMB connects to |
| `~/.umb/cache/`        | Model/cache directory (semantic search) |

`servers.json` uses the standard MCP server format (the same shape as
`.mcp.json` / `claude_desktop_config.json`):

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home"]
    }
  }
}
```

Edit `servers.json` while UMB is running — changes are picked up automatically
(hot-swap), no restart required.

## MCP Protocol

The server implements JSON-RPC 2.0 over stdio.

### `tools/list` (alias `list_tools`)

```json
{ "jsonrpc": "2.0", "id": 1, "method": "tools/list",
  "params": { "query": "optional search query" } }
```

### `list_mcps`

```json
{ "jsonrpc": "2.0", "id": 1, "method": "list_mcps" }
```

### `tools/call` (alias `route_mcp_call`)

```json
{ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
  "params": { "tool": "tool_name", "args": { "param1": "value1" } } }
```

## Development

```bash
cargo test                     # run the test suite
RUST_LOG=debug ./target/release/umb   # run with verbose logging
cargo clean                    # clean build artifacts
```

## Cross-Compilation

### macOS universal binary

```bash
rustup target add x86_64-apple-darwin aarch64-apple-darwin
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
lipo -create \
  target/x86_64-apple-darwin/release/umb \
  target/aarch64-apple-darwin/release/umb \
  -output target/umb-macos-universal
```

### Linux

```bash
rustup target add x86_64-unknown-linux-gnu
cargo build --release --target x86_64-unknown-linux-gnu
```

### Windows

```bash
rustup target add x86_64-pc-windows-msvc
cargo build --release --target x86_64-pc-windows-msvc
```

## License

Copyright 2026 Feature Collective Investments, LLC

Required Notice: Copyright 2026 Feature Collective Investments, LLC (https://universalmcpbridge.app)

Universal MCP Bridge is licensed under the **PolyForm Noncommercial License
1.0.0**. See the [`LICENSE`](./LICENSE) file for the full terms. Any
noncommercial purpose is permitted. Per the license Notices section, anyone
who receives a copy of any part of the software must also receive a copy of
the license terms and the `Required Notice` line above.

For commercial use, contact **david.burley@featurecollectiveinvestments.com**.
