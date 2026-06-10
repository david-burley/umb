"""Accuracy + gate math tests. SPEC §4 / §5."""

from __future__ import annotations

import math

from umb_validator.gates import (
    PerModelGate, composite_accuracy, evaluate_cloud_gate, evaluate_local_gate,
    evaluate_quorum, local_k, mean_juror_agreement, per_model_cloud_pass,
    per_model_local_pass, token_gate_pass, token_reduction_pct, wilson_ci,
)


def test_composite_accuracy_weighting() -> None:
    """accuracy = (cp/np)*0.6 + (cn/nn)*0.4 (SPEC §4)."""
    acc = composite_accuracy(20, 25, 12, 15, 0.6, 0.4)
    assert abs(acc - ((20 / 25) * 0.6 + (12 / 15) * 0.4)) < 1e-9


def test_composite_accuracy_zero_denominator() -> None:
    """An empty side contributes 0, not a division error."""
    assert composite_accuracy(0, 0, 10, 10, 0.6, 0.4) == 0.4
    assert composite_accuracy(10, 10, 0, 0, 0.6, 0.4) == 0.6


def test_token_reduction_pct() -> None:
    assert token_reduction_pct(100, 40) == 60.0
    assert token_reduction_pct(0, 0) == 0.0  # no baseline


def test_token_gate() -> None:
    assert token_gate_pass(100, 50, 50.0)       # exactly 50% -> pass
    assert token_gate_pass(100, 30, 50.0)       # 70% -> pass
    assert not token_gate_pass(100, 60, 50.0)   # 40% -> fail


def test_local_k_ceil_075() -> None:
    """K = ceil(0.75 * N), clamped to [1, N] (DECISION #3)."""
    assert local_k(4, 0.75) == 3
    assert local_k(3, 0.75) == 3
    assert local_k(2, 0.75) == 2
    assert local_k(1, 0.75) == 1
    assert local_k(0, 0.75) == 0


def test_per_model_local_pass_tolerance() -> None:
    """shortened >= baseline - 3pp tolerance."""
    assert per_model_local_pass(0.80, 0.78, 3.0)   # -2pp -> pass
    assert per_model_local_pass(0.80, 0.77, 3.0)   # -3pp exactly -> pass
    assert not per_model_local_pass(0.80, 0.76, 3.0)  # -4pp -> fail


def test_evaluate_local_gate_k_of_n() -> None:
    """Aggregate gate: PASS iff n_passed >= K (SPEC §5)."""
    pm = [
        PerModelGate("a", 0.8, 0.79, True),
        PerModelGate("b", 0.8, 0.79, True),
        PerModelGate("c", 0.8, 0.79, True),
        PerModelGate("d", 0.8, 0.60, False),
    ]
    gate = evaluate_local_gate(pm, 0.75)
    assert gate.k_required == 3
    assert gate.n_passed == 3
    assert gate.passed
    assert gate.summary() == "PASS 3/4"


def test_evaluate_local_gate_below_k_fails() -> None:
    pm = [
        PerModelGate("a", 0.8, 0.79, True),
        PerModelGate("b", 0.8, 0.50, False),
        PerModelGate("c", 0.8, 0.50, False),
        PerModelGate("d", 0.8, 0.50, False),
    ]
    gate = evaluate_local_gate(pm, 0.75)
    assert not gate.passed
    assert gate.summary() == "FAIL 1/4"


def test_evaluate_local_gate_empty_fails() -> None:
    """Zero responsive models -> the gate cannot pass."""
    assert not evaluate_local_gate([], 0.75).passed


def test_local_gate_adapts_n_to_responsive_models() -> None:
    """A flaky model excluded from per_model lowers N -> K rescales (3-of-3)."""
    pm = [PerModelGate(m, 0.8, 0.79, True) for m in ("a", "b", "c")]
    gate = evaluate_local_gate(pm, 0.75)
    assert gate.n_models == 3 and gate.k_required == 3 and gate.passed


def test_cloud_gate_all_must_pass() -> None:
    assert evaluate_cloud_gate([
        PerModelGate("sonnet", 0.9, 0.88, True),
        PerModelGate("gpt", 0.9, 0.87, True),
    ])
    assert not evaluate_cloud_gate([
        PerModelGate("sonnet", 0.9, 0.88, True),
        PerModelGate("gpt", 0.9, 0.50, False),
    ])
    assert not evaluate_cloud_gate([])


def test_per_model_cloud_pass_relative_floor() -> None:
    """shortened >= 0.95 * baseline."""
    assert per_model_cloud_pass(0.90, 0.855, 0.95)   # exactly 0.95x
    assert not per_model_cloud_pass(0.90, 0.80, 0.95)


def test_evaluate_quorum_3_of_4() -> None:
    """Prompt admitted iff >= Q jurors agree (DECISION #1, Q=3 N=4)."""
    assert evaluate_quorum([True, True, True, False], 3).admitted
    assert evaluate_quorum([True, True, True, True], 3).admitted
    assert not evaluate_quorum([True, True, False, False], 3).admitted


def test_evaluate_quorum_adapts_to_juror_count() -> None:
    """A juror that errored out lowers N; 3 agree of 3 still meets Q=3."""
    q = evaluate_quorum([True, True, True], 3)
    assert q.n_jurors == 3 and q.admitted
    assert not evaluate_quorum([], 3).admitted


def test_quorum_agreement_fraction() -> None:
    q = evaluate_quorum([True, True, True, False], 3)
    assert abs(q.agreement - 0.75) < 1e-9


def test_mean_juror_agreement() -> None:
    qs = [evaluate_quorum([True, True, True, False], 3),
          evaluate_quorum([True, True, True, True], 3)]
    assert abs(mean_juror_agreement(qs) - ((0.75 + 1.0) / 2)) < 1e-9
    assert mean_juror_agreement([]) == 0.0


def test_wilson_ci_bounds() -> None:
    lo, hi = wilson_ci(0.85, 50)
    assert 0.0 <= lo < 0.85 < hi <= 1.0
    # n=0 -> maximally uncertain.
    assert wilson_ci(0.5, 0) == (0.0, 1.0)


def test_wilson_ci_width_shrinks_with_n() -> None:
    narrow = wilson_ci(0.85, 500)
    wide = wilson_ci(0.85, 20)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
