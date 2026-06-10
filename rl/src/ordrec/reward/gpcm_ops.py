"""Core GPCM probability computation shared across reward and BC modules.

All four consumers that need GPCM log-probabilities (probe_entropy,
nll_anchor, bc, static_mve) call this single helper instead of
duplicating the formula. This removes the four-copy maintenance risk
identified in the E4.6a maintainability review (B1).

The GPCM head formula implemented here mirrors ``GPCMLogits`` in
``ma-irt/models/components/irt.py`` exactly. The helper does NOT
import from ma-irt so it stays usable without that package on
``PYTHONPATH``.

Mathematical formulation (GPCM, Muraki 1992 / Masters 1982 extension)::

    logit(P(Y >= k | theta, q)) = sum_{j=1}^{k}
        [alpha_q . theta - ||alpha_q|| * beta_{q,j}]

    = sum_{j=1}^{k} [interaction_q - alpha_norm_q * beta_{q,j}]

    where interaction_q = alpha_q . theta  (dot product over traits D)
          alpha_norm_q  = ||alpha_q||_2

    The cumulative sum gives step logits:
        cum_logit_{q,k} = sum_{j=1}^{k} step_value_{q,j}

    The K-category distribution is obtained by prepending a zero
    baseline and applying softmax::

        logits_q = [0, cum_logit_{q,1}, ..., cum_logit_{q,K-1}]
        P(Y = k | theta, q) = softmax(logits_q)_k

Shape conventions::

    theta     (B, D)
    item_ids  (B, N)          1-based item ids, 0 is pad
    alpha_table  (Q+1, D)     row 0 is the pad row (unused in output)
    beta_table   (Q+1, K-1)

    Returns log_probs (B, N, K) in log-probability space.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def gpcm_log_probs(
    theta: Tensor,
    item_ids: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    *,
    logit_clip: Optional[float] = 50.0,
) -> Tensor:
    """Compute GPCM log-probabilities for a batch of items.

    This is the shared low-level kernel for all reward and BC
    computations. It does not perform entropy, NLL, or Fisher
    aggregation; those are left to the caller.

    Args:
        theta: ``Tensor (B, D)`` latent ability.
        item_ids: ``LongTensor (B, N)`` 1-based item ids.
        alpha_table: ``Tensor (Q + 1, D)`` per-item discrimination.
            Row 0 is a pad row that callers may safely index into;
            its values affect no correct computation because item_ids
            in [1, Q] are the only valid indices.
        beta_table: ``Tensor (Q + 1, K - 1)`` per-item step thresholds.
        logit_clip: Optional symmetric clamp on the cumulative logits
            before the softmax. ``None`` disables. Default ``+/-50``
            prevents NaN under extreme initial parameters.

    Returns:
        ``Tensor (B, N, K)`` log-probabilities (output of
        ``torch.log_softmax``).

    Raises:
        ValueError: If ``theta`` is not 2D or ``item_ids`` is not 2D,
            or if the batch dimensions do not match.
    """
    if theta.ndim != 2:
        raise ValueError(
            f"theta must be 2D (B, D), got shape {tuple(theta.shape)}"
        )
    if item_ids.ndim != 2:
        raise ValueError(
            f"item_ids must be 2D (B, N), got shape {tuple(item_ids.shape)}"
        )
    B, D = theta.shape
    if item_ids.shape[0] != B:
        raise ValueError(
            f"item_ids batch dim {item_ids.shape[0]} != theta batch dim {B}."
        )

    # Cast tables to theta's dtype; the item cache stores float32.
    alpha_table = alpha_table.to(dtype=theta.dtype)
    beta_table = beta_table.to(dtype=theta.dtype)

    N = item_ids.shape[1]
    K = beta_table.shape[-1] + 1

    alpha_q = alpha_table[item_ids]          # (B, N, D)
    beta_q = beta_table[item_ids]            # (B, N, K-1)

    interaction = (alpha_q * theta.unsqueeze(1)).sum(dim=-1)  # (B, N)
    alpha_norm = alpha_q.norm(dim=-1)                          # (B, N)

    step_values = (
        interaction.unsqueeze(-1)
        - alpha_norm.unsqueeze(-1) * beta_q
    )  # (B, N, K-1)
    cum_logits = step_values.cumsum(dim=-1)  # (B, N, K-1)

    zeros = torch.zeros(B, N, 1, device=theta.device, dtype=theta.dtype)
    logits = torch.cat([zeros, cum_logits], dim=-1)  # (B, N, K)

    if logit_clip is not None:
        logits = logits.clamp(
            min=-float(logit_clip), max=float(logit_clip),
        )

    return torch.log_softmax(logits, dim=-1)  # (B, N, K)


__all__ = ["gpcm_log_probs"]
