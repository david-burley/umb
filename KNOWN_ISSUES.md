# Known Issues

This document records known limitations of the Universal MCP Bridge (UMB)
and the rationale behind them. UMB is a **transport bridge**: it aggregates
multiple MCP servers behind one stdio interface. It is **not** a process
supervisor, sandbox, or security arbiter for the servers it bridges.

---

## 1. A daemonizing third-party MCP server can leak one helper process per session

**Summary.** If a third-party MCP server *itself* double-forks and calls
`setsid()` to spawn a fully detached background helper, that helper escapes
the process subtree a transport gateway can see, and one such helper may
remain after the session ends.

**Why this is a third-party-server issue, not a UMB defect.** UMB cleanly
manages the process subtree a gateway is responsible for: each backend MCP
server it launches, that server's normal child processes, and any orphan
that the kernel reparents back to UMB. It terminates that subtree
deterministically on connection close, idle eviction, hot-swap, and
shutdown. A process that has deliberately double-forked and `setsid()`'d
into its own session has, by design, severed every link a parent or
gateway could use to track or reap it — that is the explicit purpose of
daemonizing.

**A conformant MCP server cannot trigger this.** The MCP stdio transport
*is* the parent/child pipe between UMB and the server. A server that
daemonizes detaches from that pipe and can no longer speak MCP, so a
correctly implemented stdio MCP server never daemonizes a detached helper
in the first place. This only arises with atypical servers that spawn an
unrelated, fully detached background process as a side effect.

**Bounded, not a cascade.** The leak is at most one detached helper per
session, per such atypical server — it does not accumulate within a
session or cascade across servers.

**Future hardening (not a committed deliverable).** Kernel-enforced
subtree containment (Linux cgroup v2 `cgroup.kill`) is one possible future
option to bound even deliberately-detached descendants. It is noted here as
a *possibility only* and is explicitly **not** a committed roadmap item.

---

## 2. Same-named tools from different servers — usage note

Two external MCP servers may both export a tool with the **same name**.
Both remain reachable; if your agent calls by name when the name is
ambiguous, UMB returns an error listing the candidate servers — pass
`server: <name>` in `route_mcp_call` to disambiguate. `get_tool_info`
on an ambiguous name returns all matches so you can read each server's
distinct description before choosing.

---

## Scope note

UMB is a transport bridge. It does not police, sandbox, or repair the
behaviour of the MCP servers it connects — it faithfully forwards their
protocol traffic and manages the process subtree a gateway owns. Servers
that misbehave (daemonize detached helpers, emit non-protocol output,
export colliding tool names) are handled as gracefully as a transport
layer can, but correcting server behaviour is the responsibility of those
servers' authors.
