"""Accuracy measurement + gate math. SPEC v2 §4 / §5.

Pure functions — heavily unit-tested. No I/O, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count via tiktoken cl100k_base (SPEC §5 token-reduction gate)."""
    return len(_ENCODER.encode(text))


def composite_accuracy(
    n_correct_pos: int, n_positive: int,
    n_correct_neg: int, n_negative: int,
    positive_weight: float = 0.6, negative_weight: float = 0.4,
) -> float:
    """SPEC §4 composite accuracy.

    accuracy = (correct_pos/n_pos)*0.6 + (correct_neg/n_neg)*0.4

    A zero-denominator side contributes 0 (it cannot be 'perfect' on an empty
    set). Weights need not sum to 1 — the caller owns that policy.
    """
    pos = (n_correct_pos / n_positive) if n_positive > 0 else 0.0
    neg = (n_correct_neg / n_negative) if n_negative > 0 else 0.0
    return pos * positive_weight + neg * negative_weight


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a proportion (SPEC §4).

    Returns (lower, upper), clamped to [0, 1]. n=0 -> (0, 1).
    """
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
              / denom)
    return (max(0.0, center - margin), min(1.0, center + margin))


def token_reduction_pct(canonical_tokens: int, shortened_tokens: int) -> float:
    """SPEC §5 reduction_pct = (canonical - shortened) / canonical * 100.

    canonical_tokens <= 0 -> 0.0 (no baseline to reduce from).
    """
    if canonical_tokens <= 0:
        return 0.0
    return (canonical_tokens - shortened_tokens) / canonical_tokens * 100.0


def token_gate_pass(
    canonical_tokens: int, shortened_tokens: int, min_reduction: float = 50.0
) -> bool:
    """Token-reduction gate: PASS iff reduction >= min_reduction (SPEC §5)."""
    return token_reduction_pct(canonical_tokens, shortened_tokens) >= min_reduction


def local_k(n: int, fraction: float = 0.75) -> int:
    """K = ceil(fraction * N), clamped to [1, N] (SPEC §5 / DECISION #3)."""
    if n <= 0:
        return 0
    return max(1, min(n, math.ceil(fraction * n)))


@dataclass
class PerModelGate:
    """Per-model local-gate result."""

    model: str
    baseline_accuracy: float
    shortened_accuracy: float
    passed: bool

    @property
    def delta_pp(self) -> float:
        """Shortened minus baseline, in percentage points."""
        return (self.shortened_accuracy - self.baseline_accuracy) * 100.0


def per_model_local_pass(
    baseline_accuracy: float, shortened_accuracy: float,
    pp_tolerance: float = 3.0,
) -> bool:
    """Per-model local pass: shortened >= baseline - tolerance_pp (SPEC §5).

    `pp_tolerance` is in percentage points; accuracies are 0..1 fractions.
    """
    return shortened_accuracy >= baseline_accuracy - (pp_tolerance / 100.0)


@dataclass
class LocalGateResult:
    """Aggregate local-model gate result (SPEC §5)."""

    per_model: list[PerModelGate]
    n_models: int
    n_passed: int
    k_required: int
    passed: bool

    def summary(self) -> str:
        """Compact '3/4' style summary for reports."""
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict} {self.n_passed}/{self.n_models}"


def evaluate_local_gate(
    per_model: list[PerModelGate], k_fraction: float = 0.75,
) -> LocalGateResult:
    """Aggregate per-model results into the local gate (SPEC §5).

    A model that failed to respond should be EXCLUDED before calling this
    (it does not appear in `per_model`), so N adapts to responsive models.
    PASS iff n_passed >= K where K = ceil(k_fraction * N).
    """
    n = len(per_model)
    n_passed = sum(1 for pm in per_model if pm.passed)
    k = local_k(n, k_fraction)
    return LocalGateResult(
        per_model=per_model,
        n_models=n,
        n_passed=n_passed,
        k_required=k,
        passed=(n > 0 and n_passed >= k),
    )


def per_model_cloud_pass(
    baseline_accuracy: float, shortened_accuracy: float,
    rel_floor: float = 0.95,
) -> bool:
    """Per-model cloud pass: shortened >= rel_floor * baseline (SPEC §5)."""
    return shortened_accuracy >= rel_floor * baseline_accuracy


def evaluate_cloud_gate(
    per_model: list[PerModelGate],
) -> bool:
    """Cloud gate: PASS iff EVERY responsive cloud model passes (SPEC §5)."""
    return len(per_model) > 0 and all(pm.passed for pm in per_model)


@dataclass
class QuorumResult:
    """Self-validation quorum outcome for one prompt (SPEC §3.3)."""

    n_jurors: int
    n_agree: int
    quorum_q: int
    admitted: bool

    @property
    def agreement(self) -> float:
        """Fraction of jurors that agreed with the expected answer."""
        return self.n_agree / self.n_jurors if self.n_jurors else 0.0


def evaluate_quorum(
    juror_agreements: list[bool], quorum_q: int,
) -> QuorumResult:
    """Cross-model quorum admission (SPEC §3.3, DECISION #1).

    A prompt is admitted iff >= Q of N jurors agree with `expected`.
    `juror_agreements[i]` is True iff juror i's pick matched the expected
    classification (positive: picked the tool; negative: did NOT).
    """
    n = len(juror_agreements)
    agree = sum(1 for a in juror_agreements if a)
    return QuorumResult(
        n_jurors=n,
        n_agree=agree,
        quorum_q=quorum_q,
        admitted=(n > 0 and agree >= quorum_q),
    )


def mean_juror_agreement(quorum_results: list[QuorumResult]) -> float:
    """Mean per-prompt juror agreement across an oracle (SPEC §3.3 / §6).

    Used for the `low_self_validation_confidence` flag.
    """
    if not quorum_results:
        return 0.0
    return sum(q.agreement for q in quorum_results) / len(quorum_results)
