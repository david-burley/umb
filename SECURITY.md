# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately. Do **not** open a public issue for
a security report.

Email **david.burley@featurecollectiveinvestments.com** with:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept),
- the UMB version (`umb --version`) and your OS.

We aim to acknowledge a report within a few business days and to work toward a
fix or coordinated disclosure within **90 days** of the initial report.

## Scope

UMB is a transport bridge: it aggregates multiple MCP servers behind one stdio
interface and forwards their protocol traffic. That shapes what is and isn't a
UMB vulnerability.

**In scope** — defects in UMB's own code, including:

- parsing of MCP/JSON-RPC traffic and `servers.json`,
- process spawning, reaping, and the connection pool,
- the stdio / HTTP / SSE transport layer,
- the tool-dictionary loader and hash-guard,
- daemon / proxy mode and the local socket / TCP listener.

**Out of scope** — the behaviour of the MCP servers you connect UMB to. You
choose which servers to configure, and a configured MCP server runs inside
your trust boundary. A malicious or buggy backend server is not a UMB
vulnerability; UMB forwards its traffic and manages the process subtree, but it
does not sandbox, police, or repair servers. See
[`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) for accepted limitations at this layer
(notably issue A2: a third-party server that deliberately double-forks and
`setsid()`s a detached helper can leak one helper process per session).

## A note on the email address

`david.burley@featurecollectiveinvestments.com` is the intended contact. If you cannot reach
it, the maintainer contact on <https://universalmcpbridge.app> is an
alternative.
