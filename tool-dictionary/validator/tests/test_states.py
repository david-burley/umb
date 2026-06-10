"""State-machine transition tests. SPEC §1."""

from __future__ import annotations

from umb_validator.states import (
    TERMINAL_STATES, State, VALID_TRANSITIONS, can_transition, is_terminal,
)


def test_terminal_states() -> None:
    assert is_terminal(State.MERGED)
    assert is_terminal(State.REJECTED)
    assert is_terminal(State.NEEDS_MANUAL)
    assert is_terminal(State.HASH_COMPUTED)
    assert not is_terminal(State.PENDING)
    assert not is_terminal(State.LOCAL_GATE_RUN)


def test_happy_path_transitions() -> None:
    """The canonical PENDING -> REVIEW_READY chain is all valid edges."""
    chain = [
        State.PENDING, State.CANONICAL_FETCHED, State.HASH_COMPUTED,
        State.RESEARCHED, State.PROMPTS_GENERATED, State.PROMPTS_SELF_VALIDATED,
        State.BASELINE_RUN, State.SHORTENED_PROPOSED, State.LOCAL_GATE_RUN,
        State.LOCAL_GATE_PASS, State.REVIEW_READY, State.MERGED,
    ]
    for src, dst in zip(chain, chain[1:]):
        assert can_transition(src, dst), f"{src} -> {dst} should be valid"


def test_invalid_transitions_rejected() -> None:
    assert not can_transition(State.PENDING, State.MERGED)
    assert not can_transition(State.MERGED, State.PENDING)
    assert not can_transition(State.REVIEW_READY, State.RESEARCHED)
    assert not can_transition(State.NEEDS_MANUAL, State.PENDING)


def test_local_gate_fail_retry_edge() -> None:
    """A failed local gate retries the shortener."""
    assert can_transition(State.LOCAL_GATE_FAIL, State.SHORTENED_PROPOSED)
    assert can_transition(State.LOCAL_GATE_FAIL, State.NEEDS_MANUAL)


def test_reject_feedback_retry_edge() -> None:
    """REJECTED re-enters SHORTENED_PROPOSED with owner feedback (SPEC §1)."""
    assert can_transition(State.REJECTED, State.SHORTENED_PROPOSED)


def test_cloud_gate_conditional_edges() -> None:
    assert can_transition(State.LOCAL_GATE_PASS, State.CLOUD_GATE_RUN)
    assert can_transition(State.LOCAL_GATE_PASS, State.REVIEW_READY)
    assert can_transition(State.CLOUD_GATE_PASS, State.REVIEW_READY)
    assert can_transition(State.CLOUD_GATE_FAIL, State.SHORTENED_PROPOSED)


def test_terminal_states_have_no_or_only_retry_exits() -> None:
    """MERGED / NEEDS_MANUAL are dead-ends; REJECTED only has the retry edge."""
    assert VALID_TRANSITIONS[State.MERGED] == frozenset()
    assert VALID_TRANSITIONS[State.NEEDS_MANUAL] == frozenset()
    assert VALID_TRANSITIONS[State.REJECTED] == frozenset(
        {State.SHORTENED_PROPOSED})


def test_every_state_has_a_transition_entry() -> None:
    """No state may be missing from the transition table."""
    for st in State:
        assert st in VALID_TRANSITIONS, f"{st} missing from VALID_TRANSITIONS"
    for st in TERMINAL_STATES:
        assert st in VALID_TRANSITIONS
