"""Hash computation == sha256_hex(description.as_bytes()). SPEC §5.

The harness MUST produce hashes the Rust loader accepts. The loader computes
`sha256_hex(live_description.as_bytes())` (tool_dictionary.rs:254/365):
lowercase-hex SHA-256 over the raw UTF-8 bytes.
"""

from __future__ import annotations

import hashlib

from umb_validator.hashing import description_hash, sha256_hex


def test_sha256_hex_matches_hashlib() -> None:
    data = b"hello world"
    assert sha256_hex(data) == hashlib.sha256(data).hexdigest()


def test_description_hash_is_lowercase_hex() -> None:
    h = description_hash("Read the contents of a file.")
    assert len(h) == 64
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)


def test_description_hash_known_vector() -> None:
    """Pin a known vector — this is the exact byte-for-byte contract with the
    Rust loader's sha256_hex(description.as_bytes())."""
    desc = "Get the current time in a timezone"
    expected = hashlib.sha256(desc.encode("utf-8")).hexdigest()
    assert description_hash(desc) == expected


def test_description_hash_no_normalization() -> None:
    """Whitespace / trailing newlines are part of the hashed bytes — the
    loader does NOT trim, so neither may the harness."""
    assert description_hash("abc") != description_hash("abc\n")
    assert description_hash(" abc") != description_hash("abc")


def test_description_hash_unicode_bytes() -> None:
    """Non-ASCII is hashed as its UTF-8 bytes, matching `.as_bytes()`."""
    desc = "Convert café ☕ time"
    assert description_hash(desc) == sha256_hex(desc.encode("utf-8"))


def test_empty_description() -> None:
    assert description_hash("") == hashlib.sha256(b"").hexdigest()
