"""umb-dev stdio parse tests against a recorded tools/list fixture. SPEC §11."""

from __future__ import annotations

from umb_validator.integration.umb_dev import (
    CanonicalTool, default_seed_registry, parse_tools_list_result,
)


def test_parse_recorded_fixture(filesystem_tools_list: dict) -> None:
    """Parse the recorded filesystem `tools/list` JSON-RPC result."""
    result = filesystem_tools_list["result"]
    tools = parse_tools_list_result(result, "filesystem")
    names = {t.name for t in tools}
    # 3 real tools; the 4th fixture entry has no `name` -> skipped.
    assert names == {"read_file", "write_file", "list_directory"}
    assert all(t.server == "filesystem" for t in tools)


def test_parse_preserves_description_verbatim(
    filesystem_tools_list: dict,
) -> None:
    """The description must be captured byte-for-byte (the hash depends on it)."""
    result = filesystem_tools_list["result"]
    tools = parse_tools_list_result(result, "filesystem")
    read = next(t for t in tools if t.name == "read_file")
    assert read.description == result["tools"][0]["description"]
    assert read.input_schema == result["tools"][0]["inputSchema"]


def test_parse_skips_nameless_entries(filesystem_tools_list: dict) -> None:
    tools = parse_tools_list_result(filesystem_tools_list["result"],
                                    "filesystem")
    assert "missing_name_entry" not in {t.name for t in tools}
    assert len(tools) == 3


def test_parse_empty_result() -> None:
    assert parse_tools_list_result({}, "x") == []
    assert parse_tools_list_result({"tools": []}, "x") == []


def test_parse_missing_description_defaults_empty() -> None:
    tools = parse_tools_list_result(
        {"tools": [{"name": "t"}]}, "s")
    assert len(tools) == 1
    assert tools[0].description == ""


def test_canonical_tool_as_tool_object() -> None:
    ct = CanonicalTool("read_file", "Reads a file.",
                        {"type": "object"}, "filesystem")
    obj = ct.as_tool_object()
    assert obj == {"name": "read_file", "description": "Reads a file.",
                   "inputSchema": {"type": "object"}}


def test_default_seed_registry() -> None:
    reg = default_seed_registry(["filesystem", "github"])
    assert reg["filesystem"]["command"] == "npx"
    assert reg["filesystem"]["args"] == [
        "-y", "@modelcontextprotocol/server-filesystem"]
