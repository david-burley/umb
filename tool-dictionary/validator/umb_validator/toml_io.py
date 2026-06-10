"""Comment-preserving TOML round-trip. SPEC v2 §5 / §6 / §9.

The harness reads/writes `tool-dictionary/*.toml` files. It MUST preserve
curator comments and formatting (the loader is comment-agnostic, but OSS
contributors are not). `tomlkit` is the comment-preserving round-trip lib.

This module handles:
- parsing a dict TOML into a typed view,
- writing a `_pending/<server>.toml` proposal,
- the `merge` operation (promote `_pending` entry/entries into the live TOML),
- hash-stamping for `bootstrap-existing-15`.

All writes are atomic (tmp + fsync + os.replace) per §7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit import TOMLDocument

from umb_validator.hashing import description_hash


@dataclass
class DictToolEntry:
    """One `[[tools]]` entry in a dict TOML."""

    name: str
    short_description: str
    schema_hash_sha256: str | None = None


@dataclass
class DictToolFile:
    """A parsed `tool-dictionary/<server>.toml`."""

    server_name: str
    tools: list[DictToolEntry] = field(default_factory=list)
    upstream_canonical_source: str | None = None
    curator: str | None = None
    reviewed_at: str | None = None

    def tool(self, name: str) -> DictToolEntry | None:
        """Find an entry by tool name."""
        return next((t for t in self.tools if t.name == name), None)


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically: tmp -> fsync -> os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def parse_dict_file(path: str | Path) -> DictToolFile:
    """Parse a dict TOML into a typed view (loses comments — read-only view)."""
    doc = tomlkit.parse(Path(path).read_text(encoding="utf-8"))
    meta = doc.get("metadata", {})
    tools: list[DictToolEntry] = []
    for raw in doc.get("tools", []):
        tools.append(DictToolEntry(
            name=str(raw.get("name", "")),
            short_description=str(raw.get("short_description", "")),
            schema_hash_sha256=(str(raw["schema_hash_sha256"])
                                if raw.get("schema_hash_sha256") else None),
        ))
    return DictToolFile(
        server_name=str(meta.get("server_name", "")),
        tools=tools,
        upstream_canonical_source=(str(meta["upstream_canonical_source"])
                                   if meta.get("upstream_canonical_source")
                                   else None),
        curator=str(meta["curator"]) if meta.get("curator") else None,
        reviewed_at=str(meta["reviewed_at"]) if meta.get("reviewed_at") else None,
    )


def load_document(path: str | Path) -> TOMLDocument:
    """Load a TOML file as an editable tomlkit document (preserves comments)."""
    return tomlkit.parse(Path(path).read_text(encoding="utf-8"))


def stamp_hash_in_document(
    doc: TOMLDocument, tool_name: str, schema_hash: str
) -> bool:
    """Set `schema_hash_sha256` on a `[[tools]]` entry in-place.

    Preserves all surrounding comments + formatting. Returns True if the
    entry was found and stamped, False otherwise.
    """
    for entry in doc.get("tools", []):
        if str(entry.get("name", "")) == tool_name:
            entry["schema_hash_sha256"] = schema_hash
            return True
    return False


def set_short_description_in_document(
    doc: TOMLDocument, tool_name: str, short_description: str,
    schema_hash: str | None = None,
) -> bool:
    """Update an entry's `short_description` (+ optional hash) in-place.

    Returns True if the entry existed, False otherwise."""
    for entry in doc.get("tools", []):
        if str(entry.get("name", "")) == tool_name:
            entry["short_description"] = short_description
            if schema_hash is not None:
                entry["schema_hash_sha256"] = schema_hash
            return True
    return False


def upsert_tool_in_document(
    doc: TOMLDocument, tool_name: str, short_description: str,
    schema_hash: str | None = None,
) -> None:
    """Update an existing entry, or append a new `[[tools]]` entry."""
    if set_short_description_in_document(doc, tool_name, short_description,
                                         schema_hash):
        return
    tools = doc.get("tools")
    if tools is None:
        tools = tomlkit.aot()
        doc["tools"] = tools
    table = tomlkit.table()
    table["name"] = tool_name
    table["short_description"] = short_description
    if schema_hash is not None:
        table["schema_hash_sha256"] = schema_hash
    tools.append(table)


def new_pending_document(
    server_name: str,
    upstream_canonical_source: str | None = None,
    curator: str = "umb-validator",
    reviewed_at: str | None = None,
) -> TOMLDocument:
    """Build a fresh dict TOML document for a `_pending/<server>.toml`."""
    doc = tomlkit.document()
    doc.add(tomlkit.comment(
        f" {server_name} — proposed by umb-validator auto-research harness."))
    doc.add(tomlkit.comment(
        " Review with `git diff` and promote with `umb-validator merge`."))
    meta = tomlkit.table()
    meta["server_name"] = server_name
    if upstream_canonical_source:
        meta["upstream_canonical_source"] = upstream_canonical_source
    meta["curator"] = curator
    if reviewed_at:
        meta["reviewed_at"] = reviewed_at
    doc["metadata"] = meta
    doc["tools"] = tomlkit.aot()
    return doc


def add_provenance_comment(
    doc: TOMLDocument, tool_name: str, provenance_lines: list[str]
) -> None:
    """Prepend a validation-provenance comment block above a `[[tools]]`
    entry (SPEC §6 curator-comment block)."""
    for entry in doc.get("tools", []):
        if str(entry.get("name", "")) == tool_name:
            # tomlkit comments on an entry: add as leading comments.
            for line in reversed(provenance_lines):
                entry.value.body.insert(0, (None, tomlkit.comment(f" {line}")))
            return


def write_document(path: str | Path, doc: TOMLDocument) -> None:
    """Atomically write a tomlkit document to disk."""
    _atomic_write(Path(path), tomlkit.dumps(doc))


def bootstrap_stamp_file(
    toml_path: str | Path, hashes: dict[str, str], header_comment: str,
) -> tuple[int, list[str]]:
    """Hash-stamp a shipped dict TOML in-place (SPEC §5 / §9 bootstrap).

    `hashes` maps tool name -> sha256 of its live canonical description.
    Stamps every matching `[[tools]]` entry, preserving comments. Adds a
    header comment. Returns (n_stamped, [missing_tool_names]).
    """
    path = Path(toml_path)
    doc = load_document(path)
    stamped = 0
    missing: list[str] = []
    present = {str(e.get("name", "")) for e in doc.get("tools", [])}
    for tool_name, schema_hash in hashes.items():
        if tool_name not in present:
            missing.append(tool_name)
            continue
        if stamp_hash_in_document(doc, tool_name, schema_hash):
            stamped += 1
    # Add header comment once (idempotent — skip if already present).
    body = doc.body
    already = any(
        item is not None and hasattr(item, "as_string")
        and header_comment.strip()[:24] in item.as_string()
        for _, item in body
    )
    if not already and stamped > 0:
        doc.body.insert(0, (None, tomlkit.comment(f" {header_comment}")))
    if stamped > 0:
        write_document(path, doc)
    return stamped, missing


def merge_pending_into_live(
    pending_path: str | Path,
    live_path: str | Path,
    only_tool: str | None = None,
    provenance: dict[str, list[str]] | None = None,
) -> list[str]:
    """Promote `_pending/<server>.toml` entries into the live dict TOML.

    SPEC §9: comment-preserving `tomlkit` merge; adds provenance comment
    blocks; leaves everything UNCOMMITTED. If `only_tool` is set, only that
    one entry is merged. Returns the list of merged tool names.

    If the live file does not exist yet, the pending file becomes the live
    file (full promotion).
    """
    pending = parse_dict_file(pending_path)
    live = Path(live_path)
    if live.exists():
        doc = load_document(live)
    else:
        doc = new_pending_document(
            pending.server_name,
            pending.upstream_canonical_source,
            curator="umb-validator",
        )
    merged: list[str] = []
    for entry in pending.tools:
        if only_tool is not None and entry.name != only_tool:
            continue
        upsert_tool_in_document(
            doc, entry.name, entry.short_description, entry.schema_hash_sha256)
        if provenance and entry.name in provenance:
            add_provenance_comment(doc, entry.name, provenance[entry.name])
        merged.append(entry.name)
    write_document(live, doc)
    return merged


def compute_and_format_provenance(
    *,
    reviewed_at: str,
    run_id: str,
    local_gate: str,
    reduction_pct: float,
    baseline_acc: dict[str, float],
    shortened_acc: dict[str, float],
    oracle_size: int,
    mean_agreement: float,
    low_confidence: bool,
) -> list[str]:
    """Build the §6 validation-provenance comment block lines."""
    base = "  ".join(f"{m}={a:.2f}" for m, a in baseline_acc.items())
    short = "  ".join(f"{m}={a:.2f}" for m, a in shortened_acc.items())
    conf = "LOW confidence" if low_confidence else "high confidence"
    return [
        f"Last validated: {reviewed_at}  (run-{run_id})",
        f"Local gate: {local_gate}   Token reduction: {reduction_pct:.0f}%",
        f"Baseline acc:  {base}",
        f"Shortened acc: {short}",
        f"Oracle: {oracle_size} prompts, mean juror agreement "
        f"{mean_agreement:.2f}   self-validation: {conf}",
    ]


def verify_hash_roundtrip(description: str) -> str:
    """Convenience: hash a description (used by tests + bootstrap)."""
    return description_hash(description)
