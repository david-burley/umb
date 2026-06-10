# Contributing to Universal MCP Bridge

Thanks for your interest in UMB. This document covers how to build, test, and
submit changes.

## License of contributions

By submitting a contribution you agree to license it under the
[PolyForm Noncommercial License 1.0.0](./LICENSE) — the same license the
project ships under (inbound = outbound). There is no separate CLA.

## Building and testing

UMB is a single Rust binary with no runtime dependencies. Install Rust via
[rustup](https://rustup.rs), then:

```bash
# Development build
cargo build

# Release build (optimized, LTO, stripped)
cargo build --release

# Run the test suite
cargo test
```

The binary lands at `target/release/umb`. Run with verbose logging via
`RUST_LOG=debug ./target/release/umb`.

Semantic search is an opt-in feature behind the `embed-onnx` flag. Most work
does not need it, but if you touch that path, build and test it too:

```bash
cargo build --features embed-onnx
cargo test --features embed-onnx
```

## What a PR is expected to do

- `cargo test` passes (default features, and `--features embed-onnx` if you
  touched semantic search).
- No new compiler warnings. The build currently carries 14 documented,
  intentional warnings — do not add more, and don't suppress the existing
  ones to make a number go down.
- Keep the diff focused. README/doc tone is terse and operator-focused;
  match it. No marketing copy.

## IMPORTANT: daemon / process-handling changes need real-Linux E2E

UMB manages a process subtree (backend MCP servers, their children, reparented
orphans) and tears it down on connection close, idle eviction, hot-swap, and
shutdown. **`cargo test` does not adequately exercise this.** Unit tests have
repeatedly reported green while a real defect remained in the daemon / stdio /
process-reap / idle-eviction surface.

If your change touches any of:

- daemon or proxy mode (`--daemon`, `--proxy`)
- stdio transport / framing
- process spawning, reaping, or the connection pool
- idle eviction or hot-swap teardown

then it requires real-Linux end-to-end validation beyond `cargo test`.
Maintainers will run that gate on such PRs before merge. Call out in your PR
description that your change is in this surface so it isn't merged on a green
unit run alone. See [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) for the scope of
what UMB does and does not own at this layer.

## Contributing to the tool dictionary

The tool dictionary (`tool-dictionary/`) is community-curated short
descriptions overlaid on live MCP-server tool defs. It has its own
contribution flow — see
[`tool-dictionary/README.md`](./tool-dictionary/README.md) for the TOML
schema, the per-entry hash-guard, and the step-by-step curation template.

## Commercial use

UMB is free for any noncommercial purpose under PolyForm NC 1.0.0. Commercial
licensing is handled separately — see <https://universalmcpbridge.app>. Please
don't open issues for licensing questions; use the contact there instead.
