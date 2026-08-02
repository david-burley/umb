# Changelog

All notable changes to Universal MCP Bridge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Agent-skills progressive disclosure: UMB now serves agent skills next to
  MCP tools. A skill is a subdirectory of the configured skills directory
  containing a `SKILL.md` with YAML-ish frontmatter (`name`, `description`;
  extra keys, folded/literal block scalars, and flow lists are tolerated).
  Two new builtin meta-tools: `skills_list` (compact index: name + short
  description + pinned flag) and `skills_read` (full body on demand,
  frontmatter stripped). Both are also reachable as direct JSON-RPC methods.
  New `[skills]` section in `config.toml` (`dir`, default `~/.umb/skills`;
  `pinned` list). Malformed skills are skipped with a warning, never fatal;
  the registry cache invalidates per file on (mtime, length) changes.

## [0.1.0] - 2026-05-22

Initial public release.

### Added

- 3 meta-tools architecture (`list_tools`, `list_mcps`, `route_mcp_call`)
  presenting a compact API to the agent regardless of how many backends are
  configured, plus built-in file/shell tools.
- Two-tier tool discovery: a slim `list_tools` envelope with on-demand
  full-detail lookup via `get_tool_info`, keeping the agent's static tool
  context small (~99% measured reduction versus exposing every backend tool
  directly).
- `(server, name)`-keyed tool registry with collision disambiguation: tools of
  the same name from different servers stay independently reachable; ambiguous
  calls return the candidate servers, and `route_mcp_call` accepts `server:`
  to disambiguate.
- Tool-dictionary overlay with a short-definitions mode (`off` / `auto` /
  `on`) and a per-entry SHA-256 hash-guard, plus a seed set of 15 well-known
  MCP servers.
- Live config hot-swap: edits to `servers.json` are picked up without a
  restart.
- Idle eviction: idle backend connections are torn down after a TTL to bound
  resource use.
- Transports: stdio, HTTP, SSE, and Streamable HTTP backends.
- Optional semantic tool search behind the `embed-onnx` feature
  (EmbeddingGemma ONNX, Matryoshka dimensions); default builds are
  keyword-only with zero ONNX dependencies.
- Daemon / proxy mode for sharing one backend across multiple clients.
- `umb --doctor` to scan for (and optionally clean) orphaned daemon processes.

[Unreleased]: https://github.com/david-burley/umb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/david-burley/umb/releases/tag/v0.1.0
