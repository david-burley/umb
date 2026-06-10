"""Per-tool state machine. SPEC v2 §1.

State transitions are append-only `state_events` rows; current state =
`MAX(event_at) WHERE server=… AND tool=…`. Durable + resumable.
"""

from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    """Per-tool pipeline state (SPEC §1 state machine diagram)."""

    PENDING = "PENDING"
    CANONICAL_FETCHED = "CANONICAL_FETCHED"
    HASH_COMPUTED = "HASH_COMPUTED"  # terminal for bootstrap-existing-15
    RESEARCHED = "RESEARCHED"
    PROMPTS_GENERATED = "PROMPTS_GENERATED"
    PROMPTS_SELF_VALIDATED = "PROMPTS_SELF_VALIDATED"
    BASELINE_RUN = "BASELINE_RUN"
    SHORTENED_PROPOSED = "SHORTENED_PROPOSED"
    LOCAL_GATE_RUN = "LOCAL_GATE_RUN"
    LOCAL_GATE_FAIL = "LOCAL_GATE_FAIL"
    LOCAL_GATE_PASS = "LOCAL_GATE_PASS"
    CLOUD_GATE_RUN = "CLOUD_GATE_RUN"
    CLOUD_GATE_FAIL = "CLOUD_GATE_FAIL"
    CLOUD_GATE_PASS = "CLOUD_GATE_PASS"
    REVIEW_READY = "REVIEW_READY"
    MERGED = "MERGED"  # terminal
    REJECTED = "REJECTED"  # terminal
    NEEDS_MANUAL = "NEEDS_MANUAL"  # terminal


# Terminal states — no further automatic processing.
TERMINAL_STATES: frozenset[State] = frozenset(
    {State.HASH_COMPUTED, State.MERGED, State.REJECTED, State.NEEDS_MANUAL}
)

# Valid forward transitions. The scheduler consults this to advance a tool.
# REJECTED -> SHORTENED_PROPOSED is the owner-feedback retry edge.
VALID_TRANSITIONS: dict[State, frozenset[State]] = {
    State.PENDING: frozenset({State.CANONICAL_FETCHED, State.NEEDS_MANUAL}),
    State.CANONICAL_FETCHED: frozenset(
        {State.HASH_COMPUTED, State.RESEARCHED, State.NEEDS_MANUAL}
    ),
    State.HASH_COMPUTED: frozenset({State.RESEARCHED, State.PENDING}),
    State.RESEARCHED: frozenset({State.PROMPTS_GENERATED, State.NEEDS_MANUAL}),
    State.PROMPTS_GENERATED: frozenset(
        {State.PROMPTS_SELF_VALIDATED, State.NEEDS_MANUAL}
    ),
    State.PROMPTS_SELF_VALIDATED: frozenset(
        {State.BASELINE_RUN, State.NEEDS_MANUAL}
    ),
    State.BASELINE_RUN: frozenset({State.SHORTENED_PROPOSED, State.NEEDS_MANUAL}),
    State.SHORTENED_PROPOSED: frozenset({State.LOCAL_GATE_RUN, State.NEEDS_MANUAL}),
    # NEEDS_MANUAL edge: forced tool-use unavailable on too many local
    # models (§2 `gateway_ungateable`) — the gate cannot legitimately run.
    State.LOCAL_GATE_RUN: frozenset(
        {State.LOCAL_GATE_PASS, State.LOCAL_GATE_FAIL, State.NEEDS_MANUAL}
    ),
    State.LOCAL_GATE_FAIL: frozenset(
        {State.SHORTENED_PROPOSED, State.NEEDS_MANUAL}
    ),
    State.LOCAL_GATE_PASS: frozenset({State.CLOUD_GATE_RUN, State.REVIEW_READY}),
    State.CLOUD_GATE_RUN: frozenset(
        {State.CLOUD_GATE_PASS, State.CLOUD_GATE_FAIL}
    ),
    State.CLOUD_GATE_FAIL: frozenset(
        {State.SHORTENED_PROPOSED, State.NEEDS_MANUAL}
    ),
    State.CLOUD_GATE_PASS: frozenset({State.REVIEW_READY}),
    State.REVIEW_READY: frozenset({State.MERGED, State.REJECTED}),
    State.MERGED: frozenset(),
    State.REJECTED: frozenset({State.SHORTENED_PROPOSED}),
    State.NEEDS_MANUAL: frozenset(),
}


def is_terminal(state: State) -> bool:
    """True iff `state` admits no further automatic processing."""
    return state in TERMINAL_STATES


def can_transition(src: State, dst: State) -> bool:
    """True iff `src -> dst` is a valid state-machine edge."""
    return dst in VALID_TRANSITIONS.get(src, frozenset())
