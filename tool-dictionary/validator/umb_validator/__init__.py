"""umb-validator — tool-dictionary auto-research validation harness.

Implements Validation Harness SPEC v2
(test-campaign/19-validation-harness/SPEC.md).

This package is ADDITIVE tooling. It reads canonical tool definitions via the
`umb-dev` MCP binary, auto-researches + self-validates a benchmark, gates
shortened descriptions against the homelab local-model fleet, and writes
proposed TOML changes to `tool-dictionary/_pending/`. It NEVER modifies the
Rust loader or any other umb/ core source.
"""

__version__ = "0.1.0"
