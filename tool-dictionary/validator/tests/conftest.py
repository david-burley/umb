"""Shared pytest fixtures for the umb-validator suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from umb_validator.store import StateStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    """A fresh, isolated SQLite state store under a tmp dir."""
    s = StateStore(tmp_path / "state.sqlite")
    yield s
    s.close()


@pytest.fixture
def filesystem_tools_list() -> dict:
    """The recorded `tools/list` JSON-RPC response for the filesystem server."""
    return json.loads(
        (FIXTURE_DIR / "filesystem_tools_list.json").read_text(encoding="utf-8")
    )
