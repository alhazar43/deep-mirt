"""Smoke tests for the E0 trajectory-recovery module."""

import numpy as np
import torch

from deep_irt.traj_synth.data_gen import generate_learning_curves
from deep_irt.traj_synth import metrics as M


def test_generate_shapes_and_growth():
    d = generate_learning_curves(n_items=40, n_respondents=20, seq_len=15, K=4, seed=0)
    assert d["seq_item_ids"].shape == (20, 15)
    assert d["seq_responses"].shape == (20, 15)
    assert d["theta_true_traj"].shape == (20, 15)
    assert d["rate_true"].shape == (20,)
    # responses are valid categories
    assert int(d["seq_responses"].max()) <= 3 and int(d["seq_responses"].min()) >= 0
    # curves rise on average (gain is positive by construction)
    net = (d["theta_true_traj"][:, -1] - d["theta_true_traj"][:, 0]).mean().item()
    assert net > 0.0


def test_fit_rate_recovers_known_rate():
    # Noiseless curve: the fitter must return the true rate closely.
    T = 60
    t = np.arange(T, dtype=float)
    for r in (0.08, 0.2, 0.45):
        theta = 1.2 - (1.2 - (-0.9)) * np.exp(-r * t)
        r_hat, _, _ = M.fit_rate(theta)
        assert abs(r_hat - r) < 0.02, f"r={r} r_hat={r_hat}"


def test_rate_recovery_summary_on_true_curves():
    d = generate_learning_curves(n_items=40, n_respondents=40, seq_len=50, K=4, seed=1)
    r_truecurve = M.recover_rates(d["theta_true_traj"].numpy())
    summ = M.rate_recovery_summary(r_truecurve, d["rate_true"].numpy())
    # fitting the noiseless true trajectories recovers the rate near-perfectly
    assert summ["spearman"] > 0.95
