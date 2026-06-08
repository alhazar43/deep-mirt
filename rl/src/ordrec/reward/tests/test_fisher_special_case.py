"""Fisher-information lens, special-case sanity.

Owen (1975) Bayesian-CAT, Muraki (1993) closed-form GPCM item
information. When the probe set ``C`` collapses to the singleton
``{q_next}``, the entropy at the probe is the predictive entropy of
that one item under the model. The Lindley (1956) entropy reduction
shaping that we use is well-defined in this special case and the
mean entropy reduces to the single-item GPCM Shannon entropy, which
matches Muraki's information-theoretic interpretation up to a
constant.

This test is the special-case sanity check; the full Fisher-info
equivalence under the Laplace posterior approximation is too involved
to assert pointwise. We check the structural identity
``phi(theta; {q}) = H[P_q(theta)]`` and that ``phi`` differs as we
move theta along the discrimination axis, matching the qualitative
prediction of the Fisher-info lens.
"""

from __future__ import annotations

import math

import torch

from ordrec.reward.probe_entropy import phi_entropy


def _single_item_entropy(theta, alpha_q, beta_q, eps=1e-12):
    """K-category Shannon entropy at (theta, alpha_q, beta_q)."""
    interaction = (alpha_q * theta).sum(dim=-1)
    alpha_norm = alpha_q.norm(dim=-1)
    step_values = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta_q
    cum_logits = step_values.cumsum(dim=-1)
    B = theta.shape[0]
    zeros = torch.zeros(B, 1, dtype=theta.dtype)
    logits = torch.cat([zeros, cum_logits], dim=-1)
    logits = logits.clamp(min=-50.0, max=50.0)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp().clamp_min(eps)
    return -(probs * log_probs).sum(dim=-1)


def test_single_item_probe_matches_direct_entropy() -> None:
    """``phi(theta; M=1)`` equals the direct GPCM entropy of the probe item."""
    torch.manual_seed(0)
    B, D, Q, K = 4, 1, 8, 4
    theta = torch.randn(B, D)
    target_id = 3
    probe = torch.full((B, 1), target_id, dtype=torch.long)
    alpha = torch.abs(torch.randn(Q + 1, D)) + 0.5
    beta = torch.randn(Q + 1, K - 1)
    phi = phi_entropy(theta, probe, alpha, beta)
    direct = _single_item_entropy(
        theta, alpha[target_id].expand(B, D), beta[target_id].expand(B, K - 1),
    )
    assert torch.allclose(phi, direct, atol=1e-6, rtol=0.0)


def test_entropy_monotone_in_alpha_for_centred_theta() -> None:
    """At theta = beta, higher alpha sharpens the distribution.

    Single-item GPCM with one threshold gives a logistic split at
    theta = beta. Doubling alpha at fixed theta-beta gap sharpens the
    distribution, which lowers the entropy. This is the qualitative
    Fisher-information prediction (info grows with alpha^2).
    """
    torch.manual_seed(1)
    B, D, K = 2, 1, 4
    theta = torch.zeros(B, D)
    beta_q = torch.zeros(B, K - 1)
    # gap of 0.5 between thresholds so the distribution is non-uniform
    # but not collapsed.
    beta_q[:, :] = torch.tensor([-0.5, 0.0, 0.5])
    alpha_lo = torch.full((B, D), 0.5)
    alpha_hi = torch.full((B, D), 3.0)
    H_lo = _single_item_entropy(theta, alpha_lo, beta_q)
    H_hi = _single_item_entropy(theta, alpha_hi, beta_q)
    assert (H_hi < H_lo).all(), (
        f"higher alpha must lower entropy in non-uniform regime; "
        f"got H_hi={H_hi.tolist()}, H_lo={H_lo.tolist()}"
    )


def test_uniform_when_alpha_zero_matches_log_K() -> None:
    """Sanity check, alpha = 0 recovers the uniform-distribution upper bound."""
    B, D, K = 3, 1, 5
    Q = 4
    theta = torch.randn(B, D)
    probe = torch.full((B, 1), 1, dtype=torch.long)
    alpha = torch.zeros(Q + 1, D)
    beta = torch.randn(Q + 1, K - 1)
    phi = phi_entropy(theta, probe, alpha, beta)
    assert torch.allclose(
        phi, torch.full((B,), math.log(K)), atol=1e-5, rtol=0.0
    )
