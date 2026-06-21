"""Tests for the E8 contextual alpha residual null controls."""

from __future__ import annotations

import math

import numpy as np

from deep_irt.bench.run_alpha_residual_null import (
    category_variance,
    exposure_index,
    gpcm_probs,
    item_demean,
    residual_diagnostics,
)
from deep_irt.bench.datagen import BenchDataConfig, generate


def test_item_demean_zeroes_each_item_mean():
    values = np.array([1.0, 3.0, 2.0, 8.0, 4.0])
    items = np.array([0, 0, 1, 1, 1])

    means, residual, counts = item_demean(values, items, n_items=2)

    assert np.allclose(means, [2.0, 14.0 / 3.0])
    assert np.allclose(counts, [2.0, 3.0])
    for item in (0, 1):
        assert math.isclose(float(residual[items == item].mean()), 0.0, abs_tol=1e-12)


def test_exposure_index_counts_prior_item_occurrences():
    items = np.array([2, 1, 2, 2, 1, 0])
    assert np.allclose(exposure_index(items, n_items=3), [0, 0, 1, 2, 1, 0])


def test_binary_gpcm_variance_matches_pq():
    theta = np.array([0.0, 1.0])
    alpha = np.array([1.0, 2.0])
    beta = np.array([[0.0], [0.5]])

    probs = gpcm_probs(theta, alpha, beta)
    p1 = probs[:, 1]

    assert np.allclose(category_variance(probs), p1 * (1.0 - p1), atol=1e-12)


def test_residual_diagnostics_reports_zero_item_mean_residual():
    ds = generate(
        BenchDataConfig(
            name="tiny",
            kind="static",
            n_learners=10,
            n_items=5,
            seq_len=8,
            n_cats=4,
            seed=0,
        )
    )
    # Synthetic occurrence alpha with deterministic item means and mild context
    # variation.  This bypasses model fitting while exercising the diagnostic
    # decomposition.
    base = ds.gt.a[ds.items0]
    alpha = base * np.exp(0.01 * np.arange(ds.cfg.seq_len)[None, :])
    beta = ds.gt.b[ds.items0]
    theta_model = ds.theta_at_step

    summary, variables = residual_diagnostics(
        ds,
        {"alpha": alpha, "beta": beta, "theta_model": theta_model},
    )

    assert summary["delta_std"] > 0.0
    assert summary["static_log_alpha_spearman"] > 0.99
    assert {row["variable"] for row in variables} >= {
        "theta_true",
        "gap_true",
        "info_true",
        "history_pos",
        "exposure_count",
    }
