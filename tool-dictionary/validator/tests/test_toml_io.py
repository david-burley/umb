"""TOML round-trip + merge + hash-stamp tests. SPEC §5 / §6 / §9."""

from __future__ import annotations

from pathlib import Path

import tomlkit

from umb_validator.bootstrap import bootstrap_stamp_from_descriptions
from umb_validator.hashing import description_hash
from umb_validator.toml_io import (
    add_provenance_comment, compute_and_format_provenance, load_document,
    merge_pending_into_live, new_pending_document, parse_dict_file,
    stamp_hash_in_document, upsert_tool_in_document, write_document,
)

# A realistic shipped dict TOML, comments + formatting included.
_SAMPLE = """\
# Time MCP Server — @modelcontextprotocol/server-time
# Canonical source: https://github.com/modelcontextprotocol/servers
[metadata]
server_name = "time"
upstream_canonical_source = "https://github.com/modelcontextprotocol/servers/tree/main/src/time"
curator = "umb-core"
reviewed_at = "2026-05-20"

[[tools]]
name = "get_current_time"
short_description = "Get the current time in a timezone"

[[tools]]
name = "convert_time"
short_description = "Convert a time between timezones"
"""


def test_parse_dict_file(tmp_path: Path) -> None:
    p = tmp_path / "time.toml"
    p.write_text(_SAMPLE, encoding="utf-8")
    parsed = parse_dict_file(p)
    assert parsed.server_name == "time"
    assert parsed.curator == "umb-core"
    assert {t.name for t in parsed.tools} == {"get_current_time",
                                              "convert_time"}
    assert parsed.tool("convert_time").short_description == \
        "Convert a time between timezones"


def test_roundtrip_preserves_comments(tmp_path: Path) -> None:
    """tomlkit round-trip MUST preserve curator comments + formatting."""
    p = tmp_path / "time.toml"
    p.write_text(_SAMPLE, encoding="utf-8")
    doc = load_document(p)
    stamp_hash_in_document(doc, "get_current_time", "deadbeef")
    write_document(p, doc)
    out = p.read_text(encoding="utf-8")
    # Comments survive.
    assert "# Time MCP Server" in out
    assert "# Canonical source:" in out
    # The hash was added.
    assert 'schema_hash_sha256 = "deadbeef"' in out
    # Untouched entry preserved.
    assert "Convert a time between timezones" in out


def test_stamp_hash_in_document_missing_entry(tmp_path: Path) -> None:
    doc = tomlkit.parse(_SAMPLE)
    assert stamp_hash_in_document(doc, "nonexistent", "h") is False
    assert stamp_hash_in_document(doc, "get_current_time", "h") is True


def test_upsert_tool_updates_existing(tmp_path: Path) -> None:
    doc = tomlkit.parse(_SAMPLE)
    upsert_tool_in_document(doc, "get_current_time", "New short desc", "hh")
    entry = next(e for e in doc["tools"] if e["name"] == "get_current_time")
    assert entry["short_description"] == "New short desc"
    assert entry["schema_hash_sha256"] == "hh"


def test_upsert_tool_appends_new(tmp_path: Path) -> None:
    doc = tomlkit.parse(_SAMPLE)
    upsert_tool_in_document(doc, "brand_new_tool", "Does a thing", "h2")
    names = {e["name"] for e in doc["tools"]}
    assert "brand_new_tool" in names and len(doc["tools"]) == 3


def test_new_pending_document_has_metadata() -> None:
    doc = new_pending_document("github", "https://github.com/x/y",
                               reviewed_at="2026-05-22")
    assert doc["metadata"]["server_name"] == "github"
    assert doc["metadata"]["curator"] == "umb-validator"
    rendered = tomlkit.dumps(doc)
    assert "umb-validator merge" in rendered  # the review comment


def test_provenance_comment_attaches(tmp_path: Path) -> None:
    doc = tomlkit.parse(_SAMPLE)
    prov = compute_and_format_provenance(
        reviewed_at="2026-05-22T00:00:00Z", run_id="latest",
        local_gate="PASS 3/3", reduction_pct=67.0,
        baseline_acc={"qwen": 0.88}, shortened_acc={"qwen": 0.87},
        oracle_size=52, mean_agreement=0.91, low_confidence=False)
    add_provenance_comment(doc, "get_current_time", prov)
    out = tomlkit.dumps(doc)
    assert "Last validated:" in out
    assert "Token reduction: 67%" in out
    # Document still parses after comment insertion.
    reparsed = tomlkit.parse(out)
    assert len(reparsed["tools"]) == 2


def test_merge_pending_into_live(tmp_path: Path) -> None:
    """`merge` promotes a _pending entry into the live TOML, comments kept."""
    live = tmp_path / "time.toml"
    live.write_text(_SAMPLE, encoding="utf-8")
    pending = tmp_path / "_pending_time.toml"
    pdoc = new_pending_document("time")
    upsert_tool_in_document(pdoc, "get_current_time",
                            "Get current time, short", "newhash")
    write_document(pending, pdoc)

    merged = merge_pending_into_live(pending, live, only_tool=None)
    assert "get_current_time" in merged
    out = live.read_text(encoding="utf-8")
    assert "Get current time, short" in out
    assert "# Time MCP Server" in out  # original comments survive the merge


def test_merge_single_tool_only(tmp_path: Path) -> None:
    live = tmp_path / "time.toml"
    live.write_text(_SAMPLE, encoding="utf-8")
    pending = tmp_path / "_pending_time.toml"
    pdoc = new_pending_document("time")
    upsert_tool_in_document(pdoc, "get_current_time", "A", "h1")
    upsert_tool_in_document(pdoc, "convert_time", "B", "h2")
    write_document(pending, pdoc)
    merged = merge_pending_into_live(pending, live, only_tool="convert_time")
    assert merged == ["convert_time"]


def test_bootstrap_stamp_from_descriptions(tmp_path: Path) -> None:
    """The bootstrap hash-stamp path: live descriptions -> stamped TOML."""
    p = tmp_path / "time.toml"
    p.write_text(_SAMPLE, encoding="utf-8")
    descs = {
        "get_current_time": "Get the current time in a timezone",
        "convert_time": "Convert a time between timezones",
    }
    n, missing = bootstrap_stamp_from_descriptions(p, descs)
    assert n == 2 and missing == []
    parsed = parse_dict_file(p)
    expected = description_hash(descs["get_current_time"])
    assert parsed.tool("get_current_time").schema_hash_sha256 == expected


def test_bootstrap_stamp_reports_missing(tmp_path: Path) -> None:
    p = tmp_path / "time.toml"
    p.write_text(_SAMPLE, encoding="utf-8")
    n, missing = bootstrap_stamp_from_descriptions(
        p, {"get_current_time": "x", "ghost_tool": "y"})
    assert n == 1
    assert missing == ["ghost_tool"]
