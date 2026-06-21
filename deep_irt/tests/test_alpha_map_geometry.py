"""Tests for alpha-map geometry summary formulas."""

from __future__ import annotations

import math

from deep_irt.bench.analyze_alpha_map_geometry import (
    map_family,
    preconditioner_value,
    spearman,
)


def test_preconditioner_formulas_at_reference_alpha():
    alpha = 0.5
    assert preconditioner_value("identity", alpha, {}) == 1.0
    assert preconditioner_value("relu", alpha, {}) == 1.0
    assert math.isclose(preconditioner_value("exp", alpha, {}), 0.25)
    assert math.isclose(
        preconditioner_value("softplus", alpha, {}),
        (1.0 - math.exp(-alpha)) ** 2,
    )
    assert math.isclose(
        preconditioner_value("scaled_softplus", alpha, {"scale": 1.5}),
        (1.5 * (1.0 - math.exp(-alpha / 1.5))) ** 2,
    )
    assert math.isclose(
        preconditioner_value("temperature_softplus", alpha, {"temperature": 0.5}),
        (0.5 * (1.0 - math.exp(-alpha))) ** 2,
    )
    assert math.isclose(
        preconditioner_value("sigmoid", alpha, {}),
        (alpha * (1.0 - alpha)) ** 2,
    )
    assert math.isclose(
        preconditioner_value("scaled_sigmoid", alpha, {"scale": 2.0}),
        (alpha * (1.0 - alpha / 2.0)) ** 2,
    )
    assert math.isclose(
        preconditioner_value("square", alpha, {"epsilon": 1e-6}),
        4.0 * (alpha - 1e-6),
    )


def test_clipped_exp_preconditioner_zeroes_on_plateau():
    kwargs = {"clip_min": -1.0, "clip_max": 1.0}
    assert math.isclose(preconditioner_value("clipped_exp", 0.5, kwargs), 0.25)
    assert preconditioner_value("clipped_exp", math.exp(2.0), kwargs) == 0.0
    assert preconditioner_value("clipped_exp", math.exp(-2.0), kwargs) == 0.0


def test_map_family_groups():
    assert map_family("exp") == "exp_family"
    assert map_family("clipped-exp") == "exp_family"
    assert map_family("softplus_eps") == "smooth_softplus_family"
    assert map_family("scaled_sigmoid") == "bounded_sigmoid_family"
    assert map_family("identity") == "unconstrained_raw"
    assert map_family("relu") == "nonsmooth_relu"
    assert map_family("square") == "nonmonotone_square"


def test_spearman_handles_ties_and_nans():
    assert math.isclose(spearman([1, 2, 3], [1, 2, 3]), 1.0)
    assert math.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)
    assert math.isfinite(spearman([1, 1, 2, float("nan")], [3, 3, 4, 5]))
