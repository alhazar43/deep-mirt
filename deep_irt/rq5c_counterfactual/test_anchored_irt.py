"""Unit tests for the RQ5c anchored Rasch scoring core.

Run (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="deep_irt/rq5c_counterfactual;rl/src;ma-irt" \\
        python -m pytest deep_irt/rq5c_counterfactual/test_anchored_irt.py -q
"""

from __future__ import annotations

import math

import pytest

from anchored_irt import (
    estimate_theta_rasch,
    mistake_rate_to_difficulty,
    proportion_correct,
    spearman,
    pearson,
    split_half_reliability,
)


def test_difficulty_monotone_and_zero_centre():
    # Mistake rate 0.5 -> difficulty 0 (population-average item).
    assert abs(mistake_rate_to_difficulty(0.5)) < 1e-9
    # Harder item (more mistakes) -> larger difficulty.
    assert mistake_rate_to_difficulty(0.8) > mistake_rate_to_difficulty(0.5)
    assert mistake_rate_to_difficulty(0.2) < mistake_rate_to_difficulty(0.5)
    # Clamp keeps 0 and 1 finite.
    assert math.isfinite(mistake_rate_to_difficulty(0.0))
    assert math.isfinite(mistake_rate_to_difficulty(1.0))


def test_theta_empty_is_nan():
    assert math.isnan(estimate_theta_rasch([], []))


def test_theta_recovers_known_ability():
    # Simulate a respondent with true theta on a grid of item difficulties,
    # using the EXPECTED response (p>=0.5 -> correct) so recovery is well posed.
    true_theta = 0.8
    diffs = [(-2.0 + 0.1 * i) for i in range(41)]  # -2.0 .. +2.0
    correct = [1 if (1.0 / (1.0 + math.exp(-(true_theta - b))) >= 0.5) else 0
               for b in diffs]
    est = estimate_theta_rasch(correct, diffs, prior_sd=10.0)
    # Deterministic thresholded responses bracket theta between the hardest
    # passed and easiest failed item; with a weak prior it lands close.
    assert abs(est - true_theta) < 0.6


def test_theta_monotone_in_score():
    diffs = [-1.0, 0.0, 1.0, 2.0]
    low = estimate_theta_rasch([1, 0, 0, 0], diffs, prior_sd=2.0)
    high = estimate_theta_rasch([1, 1, 1, 0], diffs, prior_sd=2.0)
    assert high > low


def test_theta_prior_regularises_all_correct():
    # All-correct pattern: MLE diverges, MAP stays finite and positive.
    diffs = [0.0, 0.5, 1.0]
    est = estimate_theta_rasch([1, 1, 1], diffs, prior_sd=1.0)
    assert math.isfinite(est) and est > 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        estimate_theta_rasch([1, 0], [0.0])


def test_proportion_correct():
    assert proportion_correct([1, 1, 0, 0]) == 0.5
    assert math.isnan(proportion_correct([]))


def test_spearman_perfect_monotone():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 25.0, 40.0]  # monotone increasing
    assert abs(spearman(x, y) - 1.0) < 1e-9


def test_pearson_perfect_linear():
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    assert abs(pearson(x, y) - 1.0) < 1e-9


def test_split_half_spearman_brown_inflates_positive_r():
    # SB correction raises a positive half-length r toward the full-length value.
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.1, 1.9, 3.2, 3.8, 5.1]
    raw = spearman(a, b)
    sb = split_half_reliability(a, b, spearman_brown=True)
    assert sb >= raw
