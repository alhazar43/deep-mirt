"""Bounds on :func:`phi_entropy`.

``phi`` is a per-row average of K-category Shannon entropy values, so
it must lie in ``[0, log K]`` for any input. The lower bound is
attained when the categorical concentrates on a single category; the
upper bound is attained at the uniform distribution.
"""

from __future__ import annotations

import math

import torch

from ordrec.reward.probe_entropy import phi_entropy


def test_phi_within_zero_logK_random() -> None:
    torch.manual_seed(0)
    B, D, Q, K, M = 8, 2, 16, 5, 10
    theta = torch.randn(B, D) * 3.0
    probe = torch.randint(1, Q + 1, (B, M))
    alpha = torch.abs(torch.randn(Q + 1, D)) * 2.0 + 0.1
    beta = torch.randn(Q + 1, K - 1) * 2.0

    phi = phi_entropy(theta, probe, alpha, beta)
    assert phi.shape == (B,)
    assert torch.isfinite(phi).all()
    log_K = math.log(K)
    assert (phi >= -1e-6).all(), f"phi has negative entry, min {phi.min().item()}"
    assert (phi <= log_K + 1e-6).all(), (
        f"phi exceeds log K = {log_K}, max {phi.max().item()}"
    )


def test_phi_uniform_limit() -> None:
    """Zero alpha => uniform categorical => phi == log K."""
    B, D, K, M = 3, 1, 4, 5
    Q = 8
    theta = torch.randn(B, D)
    probe = torch.randint(1, Q + 1, (B, M))
    alpha = torch.zeros(Q + 1, D)
    beta = torch.randn(Q + 1, K - 1)
    phi = phi_entropy(theta, probe, alpha, beta)
    log_K = math.log(K)
    assert torch.allclose(phi, torch.full((B,), log_K), atol=1e-5, rtol=0.0)


def test_phi_concentrated_limit_lower_bound_K_eq_2() -> None:
    """At K=2 with strong alpha and theta far above the threshold the
    distribution collapses to a single category and phi -> 0.

    At K>2 the cumulative-logit clamp at ``+/- 50`` caps how
    concentrated the categorical can become (the largest achievable
    log-odds between adjacent categories is the clamp value). So we
    use the K=2 path for the canonical "phi -> 0" check and verify a
    softer "phi well below log K" bound at K=4 below.
    """
    B, D, K, M = 2, 1, 2, 4
    Q = 6
    theta = torch.full((B, D), 5.0)
    probe = torch.randint(1, Q + 1, (B, M))
    alpha = torch.full((Q + 1, D), 10.0)
    beta = torch.full((Q + 1, K - 1), -5.0)
    phi = phi_entropy(theta, probe, alpha, beta)
    assert (phi <= 1e-3).all(), (
        f"expected ~0 entropy under strong K=2 concentration, got {phi}"
    )


def test_phi_strongly_concentrated_below_uniform_at_K_eq_4() -> None:
    """At K=4, strong alpha plus far-extreme theta yield phi << log K."""
    B, D, K, M = 2, 1, 4, 4
    Q = 6
    theta = torch.full((B, D), 5.0)
    probe = torch.randint(1, Q + 1, (B, M))
    alpha = torch.full((Q + 1, D), 3.0)
    beta = torch.full((Q + 1, K - 1), -3.0)
    phi = phi_entropy(theta, probe, alpha, beta)
    log_K = math.log(K)
    # Tight bound: phi is well below half of log K.
    assert (phi < log_K / 2).all(), (
        f"expected phi << log K under strong concentration; phi={phi}"
    )


def test_phi_extreme_theta_remains_finite() -> None:
    """Large theta does not produce NaN via logit clamping."""
    B, D, Q, K, M = 4, 1, 8, 4, 4
    theta = torch.tensor([[1000.0], [-1000.0], [50.0], [-50.0]])
    probe = torch.randint(1, Q + 1, (B, M))
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)
    phi = phi_entropy(theta, probe, alpha, beta)
    assert torch.isfinite(phi).all()
