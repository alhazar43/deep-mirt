"""anchored_irt.py -- Fixed-item-parameter respondent ability scoring.

RQ5c scores a respondent's ability on a bank by ANCHORING: item difficulties
are held FIXED at their human-calibrated values and only the respondent's
scalar ability ``theta`` is estimated. This is the fixed-item-respondent-scoring
use of anchoring (consistent with the thesis spine: identifiability comes from
fixed reference parameters, not joint co-estimation).

Model: 1PL / Rasch on binary correctness.

    P(correct_i | theta) = sigma(theta - b_i)

where ``b_i`` is the FIXED difficulty of item ``i`` derived from the human
calibration. The human calibration gives a per-item *mistake rate* ``m_i``
(fraction of the human population that gets item ``i`` wrong, pooled over all
learners). The Rasch difficulty that reproduces that population mistake rate at
the population-mean ability (theta = 0) is the logit of the mistake rate::

    b_i = logit(m_i) = log(m_i / (1 - m_i))

This puts both banks' difficulties on a single ABSOLUTE, population-anchored
metric (the log-odds that a population-average human gets the item wrong), so a
respondent's anchored theta is directly comparable across banks. That cross-bank
comparability is the whole point of the counterfactual design.

Estimation: MAP (Newton on the scalar theta with an N(0, prior_sd^2) prior) or
MLE (prior_sd -> inf). Closed-form Newton converges in a few steps because the
1PL log-likelihood in theta is strictly concave.

This module is pure-Python / NumPy, no torch, no I/O, fully unit-testable.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Clamp for difficulty derivation so logit(0) / logit(1) do not blow up. A
# mistake rate of exactly 0 or 1 is finitely informative at best; clamp to a
# plausible bound for a pooled human population.
_DIFFICULTY_CLAMP = 1e-3


def mistake_rate_to_difficulty(
    mistake_rate: float,
    clamp: float = _DIFFICULTY_CLAMP,
) -> float:
    """Convert a human population mistake rate to a fixed Rasch difficulty.

    Args:
        mistake_rate: Fraction of the human population getting the item wrong,
            in ``[0, 1]``.
        clamp: Lower/upper clamp applied before the logit so 0 and 1 map to
            finite difficulties.

    Returns:
        ``b_i = logit(clamp(mistake_rate))`` -- higher means harder.
    """
    m = min(max(mistake_rate, clamp), 1.0 - clamp)
    return math.log(m / (1.0 - m))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def estimate_theta_rasch(
    correct: Sequence[int],
    difficulties: Sequence[float],
    prior_sd: float = 1.0,
    n_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    """MAP/MLE estimate of a single respondent's Rasch ability.

    Newton-Raphson on the scalar theta. The 1PL log-likelihood is strictly
    concave in theta, and with a Gaussian prior the posterior is too, so Newton
    converges globally in a handful of steps.

    Args:
        correct: Binary correctness per item (1 = correct, 0 = wrong), length n.
        difficulties: Fixed Rasch difficulty ``b_i`` per item, length n.
        prior_sd: Standard deviation of the N(0, prior_sd^2) prior on theta.
            Pass ``float('inf')`` for pure MLE (no prior).
        n_iter: Maximum Newton iterations.
        tol: Convergence tolerance on the theta step.

    Returns:
        The MAP (or MLE) point estimate of theta. Returns ``nan`` if there are
        no items.

    Notes:
        With pure MLE, an all-correct or all-wrong response pattern has no finite
        maximiser (theta -> +/-inf). A finite ``prior_sd`` regularises these to a
        finite value; this is why MAP is the default. The caller should report
        per-bank accuracy so degenerate (ceiling/floor) patterns are visible.
    """
    n = len(correct)
    if n == 0:
        return float("nan")
    if len(difficulties) != n:
        raise ValueError(
            f"correct (n={n}) and difficulties (n={len(difficulties)}) must align."
        )

    use_prior = math.isfinite(prior_sd) and prior_sd > 0
    prior_prec = 1.0 / (prior_sd * prior_sd) if use_prior else 0.0

    theta = 0.0
    for _ in range(n_iter):
        grad = 0.0
        hess = 0.0  # second derivative of the log-posterior (negative definite)
        for c, b in zip(correct, difficulties):
            p = _sigmoid(theta - b)
            grad += (c - p)
            hess += -p * (1.0 - p)
        if use_prior:
            grad += -prior_prec * theta
            hess += -prior_prec
        if abs(hess) < 1e-12:
            break
        step = grad / hess  # Newton: theta_new = theta - grad/hess
        theta_new = theta - step
        if not math.isfinite(theta_new):
            break
        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    # Guard against MLE divergence on degenerate patterns.
    if not math.isfinite(theta):
        return float("nan")
    return float(theta)


def proportion_correct(correct: Sequence[int]) -> float:
    """Raw proportion-correct sanity backup. Returns nan for empty input."""
    n = len(correct)
    if n == 0:
        return float("nan")
    return sum(correct) / n


# ---------------------------------------------------------------------------
# Rank statistics (no scipy) -- mirror tap_scaled_transfer for consistency.
# ---------------------------------------------------------------------------

def rank(vals: Sequence[float]) -> List[float]:
    """Average ranks (1-based), ties averaged."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[order[j + 1]] == vals[order[j]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rank(x), rank(y))


def split_half_reliability(
    theta_a: Sequence[float],
    theta_b: Sequence[float],
    spearman_brown: bool = True,
) -> float:
    """Correlation between two half-scale ability estimates.

    Args:
        theta_a: Ability per respondent from random half A of a bank.
        theta_b: Ability per respondent from random half B of the same bank.
        spearman_brown: If True, apply the Spearman-Brown prophecy correction to
            project the half-length reliability up to the full bank length.

    Returns:
        The (optionally corrected) Spearman correlation between the halves.
    """
    r = spearman(list(theta_a), list(theta_b))
    if not spearman_brown or not math.isfinite(r):
        return r
    denom = 1.0 + r
    if abs(denom) < 1e-12:
        return r
    return 2.0 * r / denom


__all__ = [
    "mistake_rate_to_difficulty",
    "estimate_theta_rasch",
    "proportion_correct",
    "rank",
    "pearson",
    "spearman",
    "split_half_reliability",
]
