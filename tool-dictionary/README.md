# Tool-Definitions Dictionary

This directory contains community-curated short descriptions for well-known
MCP servers' tools. UMB overlays these on top of the live, server-supplied
tool descriptions at read time (`get_tool_info` / `list_tools`), giving
agents leaner per-tool text without modifying any underlying server.

## How it works (one paragraph)

The router stores live tool defs verbatim. At read time (in `get_tool_info`
and in the slim two-tier `list_tools` envelope), the dict is consulted via
`(server_name, tool_name)`. If a matching entry exists AND the safety mode
permits (see `short_definitions` config), the dict's `short_description`
replaces the live description in the JSON response. Provenance is exposed
as `_source: "dict" | "server"` so operators can audit which entries
actually applied. The underlying storage is never mutated — flipping the
flag at runtime returns to the live def immediately.

## Precedence (lowest → highest)

1. **Compile-time fallback** — every file in this directory is
   `include_str!`'d into the binary, so a freshly-installed `umb` has the
   dict even without any on-disk files.
2. **In-repo on-disk** — `<repo>/tool-dictionary/*.toml` (when running from
   a checkout); file-by-file overlay over compile-in entries.
3. **Config-specified dirs** — every path listed in
   `[general].tool_dictionary_paths` in `~/.umb/config.toml`.
4. **User overlay** — `~/.umb/tool-dictionary/*.toml` (or whatever
   `tool_dictionary_user_dir` resolves to) — HIGHEST precedence; users win.

Later overlays override earlier ones per `(server, tool)` key.

## Safety mode (`short_definitions` in `~/.umb/config.toml`)

```toml
[general]
short_definitions = "auto"   # off | auto | on
```

- **`off`** — dict is never consulted; always returns live server def.
- **`auto`** (default) — applies if a dict entry exists AND its recorded
  `schema_hash_sha256` matches the live description's hash. On mismatch
  the loader silently falls back to the live def (the upstream tool drifted
  and the override is stale). Entries WITHOUT a `schema_hash_sha256` are
  treated as if `On` (curator opted out of the hash-guard for that entry).
- **`on`** — applies if a dict entry exists, regardless of hash. Useful
  for audit / force-apply scenarios.

## Per-entry hash guard

`schema_hash_sha256` is the SHA-256 (lowercase hex) of the LIVE canonical
description (the upstream server's `tools/list` `description` for that
tool) recorded at curation time. The loader hashes the live description at
lookup and compares; on mismatch `Auto` falls back silently.

Curators populate this by spawning the upstream MCP server, calling
`tools/list`, and hashing the `description` field for each tool. See the
contribution template below.

## TOML schema

```toml
[metadata]
server_name = "filesystem"                # canonical server name (matches Tool.server)
upstream_canonical_source = "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem"
curator = "umb-core"
reviewed_at = "2026-05-20"

[[tools]]
name = "read_file"
short_description = "Read file contents"  # hand-curated, terse
# schema_hash_sha256 = "abc..."           # optional; enables Auto-mode hash guard
```

Field roles:

| Field | Required | Purpose |
|---|---|---|
| `metadata.server_name` | yes | Exact server name as the user registers it. |
| `metadata.upstream_canonical_source` | recommended | Audit / curator provenance. |
| `metadata.curator` | optional | Who curated / signs the file. |
| `metadata.reviewed_at` | optional | Date the entries were last reviewed. |
| `[[tools]].name` | yes | Exact tool name as the server exports it. |
| `[[tools]].short_description` | yes | Hand-curated replacement. |
| `[[tools]].schema_hash_sha256` | optional | sha256(live canonical description). When present, gates Auto-mode. |

## Inspecting which entries applied

Run `umb --doctor-tools` (with all your servers registered) — it dumps
JSON of every registered tool as `{name, server, description, source}`
where `source` is `"dict"` if the dictionary overrode the description and
`"server"` otherwise. Useful for catching stale entries and for measuring
how much of the dict actually fires against your particular set of
servers.

## Contribution template

To curate / contribute a new server's entries:

1. Spawn the upstream MCP server, call `tools/list`, capture the
   `description` for each tool. (For example:
   `npx -y @modelcontextprotocol/server-X | jq '.tools[] | {name, description}'`.)
2. For each tool, write a terse 4-12-word `short_description`. Action-verb
   first. No marketing fluff. The agent must still be able to pick the
   tool correctly from the short text alone.
3. Compute `schema_hash_sha256 = sha256(live_description_text)` — lowercase
   hex, no whitespace. Add it to each entry to enable the Auto-mode
   hash-guard. (Optional — entries without it always apply in Auto mode.)
4. Add a `[metadata]` block with `server_name`, `upstream_canonical_source`,
   `curator`, `reviewed_at`.
5. Save as `tool-dictionary/<server_name>.toml`.
6. Sanity-check: `cargo build --release && cargo test`.

Example minimal entry:

```toml
[metadata]
server_name = "myserver"
upstream_canonical_source = "https://example.com/myserver"
curator = "yourname"
reviewed_at = "2026-05-20"

[[tools]]
name = "ping"
short_description = "Send a ping request"
schema_hash_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
```

## Seed list (15 servers shipping today)

The initial set focuses on the highest-impact upstream MCP servers. The
top 5 (github, filesystem, playwright, sequential-thinking, memory) have
the most thorough hand-curation; the remaining 10 are conservative
short_descriptions with no hash-guard (so they always apply when the
server name + tool name match, regardless of upstream description drift).

| File | Server | Notes |
|---|---|---|
| `github.toml` | github | 5 high-impact tools (issues, PRs, files, repos) |
| `filesystem.toml` | filesystem | All 11 official filesystem tools |
| `playwright.toml` | playwright | Top browser-control tools |
| `sequential-thinking.toml` | sequential-thinking | The single `sequentialthinking` tool |
| `memory.toml` | memory | All 9 KG entity/relation tools |
| `fetch.toml` | fetch | Single `fetch` tool |
| `time.toml` | time | 2 tools |
| `github-actions.toml` | github-actions | Workflow management tools |
| `brave-search.toml` | brave-search | Web + local search |
| `sqlite.toml` | sqlite | 6 SQL tools |
| `postgres.toml` | postgres | `query` tool |
| `gitlab.toml` | gitlab | Parallel to github |
| `slack.toml` | slack | Channel + message tools |
| `puppeteer.toml` | puppeteer | Browser-control tools |
| `gdrive.toml` | gdrive | Search + read tools |

## Limits

- The dict can only OVERRIDE existing tools — it cannot invent new ones.
  If a server isn't registered, its dict entries simply never fire.
- The dict does NOT modify input schemas — only descriptions. (Schemas are
  emitted via `minify_schema()` from the live server def.)
- One-shot load at startup. No hot-reload yet — restart `umb` after
  editing files in `~/.umb/tool-dictionary/`.
