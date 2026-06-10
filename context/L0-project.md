# L0-project.md: Universal MCP Bridge — Project Overview

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Project** | Universal MCP Bridge (UMB) — a Feature Collective Investments company |
| **Language** | Rust (2021 edition) |
| **Runtime** | Tokio async runtime, JSON-RPC 2.0 over stdio |
| **Build System** | Cargo (native Rust) |
| **Package Name** | `umb` |
| **Version** | 0.1.0 |
| **Entry Point** | `src/main.rs` (async, delegates to modules) |
| **Binary Output** | `target/release/umb` (3–8 MB stripped) |
| **License** | PolyForm Noncommercial 1.0.0 (see `LICENSE`) |

## What This Project Does

Universal MCP Bridge consolidates many MCP (Model Context Protocol) servers
behind a single unified MCP server. Instead of configuring dozens of servers
directly in an AI coding agent, the user configures them once in UMB, and UMB
presents a compact 3-tool API (plus built-in file/shell tools) to the agent.

UMB runs as a JSON-RPC 2.0 server over stdio. Core responsibilities:

- Manage MCP server connections across stdio, HTTP, and SSE transports
- Discover tools from every configured server
- Route tool invocations to the correct backing server
- Optional semantic tool search (EmbeddingGemma ONNX, Matryoshka dimensions)
- Live config hot-swap (watch `servers.json`, apply changes without restart)
- Optional daemon/proxy mode to share one backend across multiple clients

There is **no license validation, tier system, server limit, or feature
gating**. Every capability is always available. This is the open release.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/`            | Rust source code |
| `src/cli/`        | CLI argument parsing, command handlers, signal setup |
| `src/startup/`    | MCP server startup, tool discovery, hot-swap watcher |
| `src/server/`     | MCP protocol (JSON-RPC) and tool routing |
| `src/daemon/`     | Daemon backend (multi-client mode) |
| `src/client/`     | Proxy connector / stdio forwarder for daemon mode |
| `src/registry/`   | Server registry (`~/.umb/servers.json`) |
| `src/features/`   | Semantic search + hot-swap implementations |
| `src/config.rs`   | User configuration (`~/.umb/config.toml`) |
| `src/utils/`      | Logging helpers |
| `Cargo.toml`      | Dependency manifest |
| `build.rs`        | Build script (ONNX Runtime setup) |

## Common Tasks

```bash
# Build
cargo build               # dev build
cargo build --release     # optimized, stripped → target/release/umb

# Test / lint
cargo test
cargo fmt
cargo clippy
cargo clean

# Run (silent MCP mode is the default)
./target/release/umb
./target/release/umb --verbose
./target/release/umb --list-servers
./target/release/umb --daemon            # daemon backend
./target/release/umb --proxy             # proxy to daemon
./target/release/umb --doctor            # scan orphaned daemons
```

### Configuration Files

```
~/.umb/config.toml     # general + semantic-search settings (optional)
~/.umb/servers.json    # MCP server definitions (standard MCP format)
~/.umb/cache/          # model/cache directory (semantic search)
```

### Cross-Platform Compilation

```bash
# macOS universal
rustup target add x86_64-apple-darwin aarch64-apple-darwin
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
lipo -create target/x86_64-apple-darwin/release/umb \
  target/aarch64-apple-darwin/release/umb -output target/umb-macos-universal

# Linux x64
cargo build --release --target x86_64-unknown-linux-gnu

# Windows x64
cargo build --release --target x86_64-pc-windows-msvc
```

## Navigation Hints

**Startup sequence** (`src/main.rs`):
1. Initialize logger
2. Parse CLI arguments (`src/cli/args.rs`)
3. Handle info commands: `--doctor`, `--list-servers` (`src/cli/commands.rs`)
4. Branch: `--daemon` → daemon backend; `--proxy` → proxy forwarder;
   otherwise → start the MCP server (`src/startup/server.rs`)
5. Signal handling via `src/cli/signal.rs` (graceful shutdown)

**Core modules**:
- `src/server/mcp.rs` — JSON-RPC 2.0 protocol: `initialize`, `tools/list`,
  `tools/call`, plus meta-tools `list_tools`, `list_mcps`, `route_mcp_call`
- `src/server/router.rs` — `ToolRouter`: server configs, tool registry,
  discovery, dispatch, semantic-search filtering in `list_tools()`
- `src/startup/discovery.rs` — discovery pipeline across stdio → SSE → HTTP
- `src/features/semantic_search.rs` — EmbeddingGemma ONNX inference,
  Matryoshka dimensions (128/256/512/768)
- `src/features/hot_swap.rs` — `notify`-based watcher on `servers.json`
- `src/registry/config.rs` — parses `~/.umb/servers.json`; all enabled
  servers are active (no limits)
- `src/daemon/` + `src/client/` — daemon backend and lightweight proxy

## Architecture Patterns

- **Background discovery**: tool discovery runs in background tasks so it does
  not block MCP server startup
- **Multiple transports**: stdio (fast), SSE/Docker, HTTP
- **Async/await**: all I/O is non-blocking on the tokio runtime
- **Shared state**: router wrapped in `Arc<RwLock<_>>` for thread-safe access
- **Error handling**: `anyhow::Result<T>` throughout
- **Structured logging**: `tracing` with context tags (`[Main]`, `[Background]`,
  `[HotSwap]`)
