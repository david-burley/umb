"""Distractor-window reproducibility tests. SPEC §3.4 / DECISION #4."""

from __future__ import annotations

from umb_validator.integration.umb_dev import CanonicalTool
from umb_validator.prompt_structure import ToolUniverse, build_window


def _universe() -> ToolUniverse:
    by_server = {
        "filesystem": [
            CanonicalTool("read_file", "Read a file", {}, "filesystem"),
            CanonicalTool("write_file", "Write a file", {}, "filesystem"),
        ],
        "github": [
            CanonicalTool("create_issue", "Create an issue", {}, "github"),
            CanonicalTool("search_repos", "Search repos", {}, "github"),
        ],
        "time": [
            CanonicalTool("get_time", "Get the time", {}, "time"),
            CanonicalTool("convert_time", "Convert time", {}, "time"),
        ],
        "slack": [
            CanonicalTool("post_message", "Post a message", {}, "slack"),
            CanonicalTool("list_channels", "List channels", {}, "slack"),
        ],
        "memory": [
            CanonicalTool("create_entities", "Create entities", {}, "memory"),
        ],
    }
    return ToolUniverse.from_canonical(by_server)


def test_window_includes_candidate() -> None:
    u = _universe()
    win = build_window(u, "filesystem", "read_file", 0, window=8)
    assert any(t["name"] == "read_file" for t in win)


def test_window_size_capped() -> None:
    u = _universe()
    win = build_window(u, "filesystem", "read_file", 0, window=8)
    assert len(win) <= 8


def test_window_is_seeded_reproducible() -> None:
    """SPEC §3.4: SAME (server, tool, prompt_idx) -> identical window, so
    self-validation, baseline, and shortened runs see the same distractors."""
    u = _universe()
    w1 = build_window(u, "github", "create_issue", 5, window=8)
    w2 = build_window(u, "github", "create_issue", 5, window=8)
    assert [t["name"] for t in w1] == [t["name"] for t in w2]


def test_window_reproducible_across_fresh_universe() -> None:
    """The seed must NOT depend on builtin hash() salting — a fresh process /
    fresh universe object must yield the identical window."""
    w1 = build_window(_universe(), "time", "get_time", 3, window=8)
    w2 = build_window(_universe(), "time", "get_time", 3, window=8)
    assert [t["name"] for t in w1] == [t["name"] for t in w2]


def test_different_prompt_idx_differs() -> None:
    u = _universe()
    w1 = [t["name"] for t in build_window(u, "time", "get_time", 1)]
    w2 = [t["name"] for t in build_window(u, "time", "get_time", 999)]
    assert w1 != w2  # overwhelmingly likely with a good seed


def test_distractors_from_other_servers() -> None:
    """Distractors are drawn from DIFFERENT servers than the candidate."""
    u = _universe()
    win = build_window(u, "filesystem", "read_file", 0, window=8)
    # The only filesystem tool allowed is the candidate itself.
    fs_tools = [t for t in win if t["name"] in ("read_file", "write_file")]
    assert [t["name"] for t in fs_tools] == ["read_file"]


def test_must_include_forces_negative_other_tool() -> None:
    """DECISION #4: the intended other-tool is guaranteed in the window."""
    u = _universe()
    win = build_window(u, "filesystem", "read_file", 7, window=8,
                       must_include=("slack", "post_message"))
    assert any(t["name"] == "post_message" for t in win)
    assert any(t["name"] == "read_file" for t in win)


def test_window_strips_internal_keys() -> None:
    u = _universe()
    win = build_window(u, "memory", "create_entities", 0, window=8)
    for t in win:
        assert not any(k.startswith("_") for k in t)
