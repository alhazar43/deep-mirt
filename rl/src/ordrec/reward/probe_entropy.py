"""Vectorised GPCM predictive entropy over a probe.

The primary reward channel is the per-batch reduction in predictive
entropy on a fixed probe set ``C``. This module implements the entropy
function ``phi(theta) = mean_q H[P(Y_q | theta, alpha_q, beta_q)]`` in
a single batched PyTorch call. The probe is provided as item ids and
the ``(alpha, beta)`` tables come from the item cache built in E2, so
no encoder forwards are needed here.

The GPCM probabilistic head used by ma-irt collapses the per-step
interaction as ``alpha . theta`` and rescales beta by ``|alpha|_2``;
see ``ma-irt/models/components/irt.py::GPCMLogits``. This module
mirrors that formula exactly so the entropy evaluated here matches
the head the world model uses.

See ``docs/ordrec_impl_guide.md`` Section 3.4 for the formula and
Section 3.8 for the numerical-stability notes.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def phi_entropy(
    theta: Tensor,
    probe_ids: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    *,
    eps: float = 1e-12,
    logit_clip: Optional[float] = 50.0,
) -> Tensor:
    """Predictive entropy averaged over a probe.

    Computes ``phi(theta) = (1 / M) * sum_q H[P(Y_q | theta)]`` where
    ``H[P]`` is Shannon entropy in nats and ``P`` is the GPCM
    predictive distribution. The formula vectorises across the batch
    and the probe simultaneously.

    Args:
        theta: ``Tensor (B, D)``. Latent ability per batch row.
        probe_ids: ``LongTensor (B, M)``. 1-based item ids; ``0`` is
            reserved for padding and is allowed in case the caller
            pre-masks (the padding row of ``alpha_table`` defaults to
            unit alpha and zero beta, which yields the uniform
            distribution and a constant ``log K`` entropy that cancels
            in the shaping difference).
        alpha_table: ``Tensor (Q + 1, D)``. Per-item discrimination.
        beta_table: ``Tensor (Q + 1, K - 1)``. Per-item step
            thresholds.
        eps: Probability-side clamp guarding the ``p log p`` product.
            Applied to ``exp(log_probs)``, not to ``log_probs``,
            preserving ``log_softmax`` numerical accuracy.
        logit_clip: Optional symmetric clamp on the cumulative logits
            before the softmax. ``None`` disables. The default of
            ``+/- 50`` prevents NaN under extreme initial alpha.

    Returns:
        ``Tensor (B,)`` with the per-row entropy average. Float dtype
        matches ``theta``.

    Shape:
        ``theta (B, D)``, ``probe_ids (B, M)``, ``alpha_table (Q+1, D)``,
        ``beta_table (Q+1, K-1)`` -> output ``(B,)``.
    """
    if theta.ndim != 2:
        raise ValueError(
            f"theta must be 2D (B, D), got shape {tuple(theta.shape)}"
        )
    if probe_ids.ndim != 2:
        raise ValueError(
            f"probe_ids must be 2D (B, M), got shape {tuple(probe_ids.shape)}"
        )
    if alpha_table.ndim != 2 or beta_table.ndim != 2:
        raise ValueError(
            "alpha_table and beta_table must both be 2D, got "
            f"alpha {tuple(alpha_table.shape)}, beta {tuple(beta_table.shape)}"
        )

    B, D = theta.shape
    if alpha_table.shape[-1] != D:
        raise ValueError(
            f"alpha_table last dim {alpha_table.shape[-1]} must equal "
            f"theta dim D={D}."
        )
    if probe_ids.shape[0] != B:
        raise ValueError(
            f"probe_ids batch dim {probe_ids.shape[0]} != theta batch dim {B}."
        )

    # Cast tables to theta's dtype so concatenation and arithmetic do
    # not silently upcast/downcast. The cache is float32 on disk, theta
    # is whatever the caller passes (typically float32).
    alpha_table = alpha_table.to(dtype=theta.dtype)
    beta_table = beta_table.to(dtype=theta.dtype)

    # Per-row, per-probe-item gather. Resulting shapes:
    #   alpha_q (B, M, D)
    #   beta_q  (B, M, K-1)
    alpha_q = alpha_table[probe_ids]
    beta_q = beta_table[probe_ids]

    # Interaction term alpha . theta -> (B, M).
    interaction = (alpha_q * theta.unsqueeze(1)).sum(dim=-1)

    # |alpha|_2 along the trait dim -> (B, M).
    alpha_norm = alpha_q.norm(dim=-1)

    # Per-step values -> (B, M, K-1). Mirrors GPCMLogits.
    step_values = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta_q
    cum_logits = step_values.cumsum(dim=-1)  # (B, M, K-1)

    # Prepend the K=0 baseline column.
    M = probe_ids.shape[1]
    zeros = torch.zeros(B, M, 1, device=theta.device, dtype=theta.dtype)
    logits = torch.cat([zeros, cum_logits], dim=-1)  # (B, M, K)

    if logit_clip is not None:
        logits = logits.clamp(min=-float(logit_clip), max=float(logit_clip))

    log_probs = torch.log_softmax(logits, dim=-1)  # (B, M, K)
    probs = log_probs.exp().clamp_min(float(eps))  # (B, M, K)
    H_q = -(probs * log_probs).sum(dim=-1)  # (B, M)

    return H_q.mean(dim=1)  # (B,)


__all__ = ["phi_entropy"]
