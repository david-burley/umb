"""Hash computation. SPEC v2 §5 hash policy.

The harness MUST produce hashes the Rust loader accepts. The loader computes
`sha256_hex(description.as_bytes())` (tool_dictionary.rs:254/365): SHA-256 over
the RAW UTF-8 bytes of the canonical `description` string, formatted as
lowercase hex. No normalization, no trim, no JSON re-encoding.
"""

from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    """Lowercase-hex SHA-256 of raw bytes — byte-for-byte parity with the
    Rust loader's `sha256_hex` in tool_dictionary.rs."""
    return hashlib.sha256(data).hexdigest()


def description_hash(description: str) -> str:
    """Hash of a canonical tool description.

    The description is hashed as its UTF-8 bytes verbatim — exactly what the
    loader does (`live_description.as_bytes()`). Callers MUST pass the
    description string exactly as returned by the upstream MCP server's
    `tools/list`, with no whitespace stripping or normalization.
    """
    return sha256_hex(description.encode("utf-8"))
