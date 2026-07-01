"""_p2_ordinal_ce.py -- Cumulative-link (graded-response) ordinal cross-entropy.

Phase 2 build (blueprint section A).  GPCM currently trains on WeightedOrdinalLoss
(WOL), which the user bans as an improper scoring rule.  This module supplies the
STRICTLY-PROPER ordinal loss the recovery claim requires: the cumulative-link CE.

The loss
--------
Given GPCM per-category logits ``psi`` (``psi_0 = 0``, ``psi_k = cumsum_c
alpha*(theta - beta_c)``), form the K-1 cumulative probabilities

    F_k = P(Y >= k) = sum_{j>=k} softmax(psi)_j ,     k = 1 .. K-1

with cut targets ``t_k = 1[y >= k]``.  The loss is the sum of the K-1 binary
cross-entropies at the cut points:

    L(y) = - sum_{k=1}^{K-1} [ t_k log F_k + (1 - t_k) log(1 - F_k) ] .

Numerical stability (no clamps).  Because the softmax partitions the K logits,
``P(Y>=k)`` and ``P(Y<k)`` are the tail and head masses of the SAME normaliser:

    log F_k       = logsumexp(psi[k:])  - logsumexp(psi)
    log(1 - F_k)  = logsumexp(psi[:k])  - logsumexp(psi)

so both are exact partitioned log-sum-exps -- no ``log(clamp(p))`` and the two
add to 1 by construction.

Properties
----------
* STRICTLY PROPER: minimised in expectation only by the true category probs, so
  the recovered (alpha, beta) are Fisher-consistent under a GPCM DGP.  (This is
  the graded-response cumulative likelihood, not GPCM's adjacent-category
  likelihood; applied to GPCM cumulative probs it is still proper -- methods
  caveat to STATE, per the blueprint.)
* NOT order-blind: unlike softmax CE it penalises by cut-point distance.
* Reduces EXACTLY to 2PL binary CE at K=2 (unifies the family); the unit test
  ``_p2_ordinal_ce_test.py`` asserts this to machine precision.

Scratch file: ``_``-prefixed, gitignored.  Does NOT import from or edit Codex
core beyond the read-only decoder used by the convenience helper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core loss (operates on logits directly -- the training path calls this)
# ---------------------------------------------------------------------------

def cumulative_link_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cumulative-link ordinal cross-entropy over GPCM per-category logits.

    Parameters
    ----------
    logits : (N, K)  GPCM per-category logits ``psi`` (``psi_0`` need not be 0;
             only differences matter, the loss is shift-invariant per row).
    targets : (N,)   long ordinal labels in ``{0, .., K-1}``.
    reduction : "mean" | "sum" | "none".  "none" returns the per-example loss
                (N,) summed over the K-1 cut points -- the form a masked,
                variable-length training loop multiplies by its position mask.

    Returns
    -------
    scalar (mean/sum) or (N,) tensor (none).
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (N, K); got shape {tuple(logits.shape)}")
    N, K = logits.shape
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")
    if targets.dim() != 1 or targets.shape[0] != N:
        raise ValueError(
            f"targets must be (N,) matching logits N={N}; got {tuple(targets.shape)}"
        )

    logZ = torch.logsumexp(logits, dim=1, keepdim=True)          # (N, 1)

    # Partitioned log-sum-exp at each cut point k = 1 .. K-1.  Both slices are
    # non-empty for k in [1, K-1], so no empty-logsumexp edge case arises.
    log_F = []      # log P(Y >= k)
    log_1mF = []    # log P(Y <  k)
    for k in range(1, K):
        tail = torch.logsumexp(logits[:, k:], dim=1, keepdim=True) - logZ
        head = torch.logsumexp(logits[:, :k], dim=1, keepdim=True) - logZ
        log_F.append(tail)
        log_1mF.append(head)
    log_F = torch.cat(log_F, dim=1)          # (N, K-1) = log F_k
    log_1mF = torch.cat(log_1mF, dim=1)      # (N, K-1) = log(1 - F_k)

    # Cut targets t_k = 1[y >= k] for k = 1 .. K-1.
    ks = torch.arange(1, K, device=logits.device).unsqueeze(0)   # (1, K-1)
    t = (targets.unsqueeze(1) >= ks).to(log_F.dtype)             # (N, K-1)

    per_cut = -(t * log_F + (1.0 - t) * log_1mF)                 # (N, K-1)
    per_example = per_cut.sum(dim=1)                            # (N,)

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    if reduction == "mean":
        return per_example.mean()
    raise ValueError(f"reduction must be mean|sum|none; got {reduction!r}")


# ---------------------------------------------------------------------------
# Convenience: build GPCM logits then score (mirrors GPCMDecoder.logits exactly)
# ---------------------------------------------------------------------------

def gpcm_logits(
    theta: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """GPCM per-category logits ``psi`` (``psi_0 = 0``).

    Bit-identical to ``core.decoders.GPCMDecoder.logits`` so the loss scores the
    same readout the decoder produces.

    theta : (N,) or (N, 1);  alpha : (N, 1);  beta : (N, K-1)  ->  (N, K).
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(1)
    diff = alpha * (theta - beta)                 # (N, K-1)
    psi_pos = torch.cumsum(diff, dim=1)           # (N, K-1)
    zeros = torch.zeros(theta.size(0), 1, device=theta.device, dtype=psi_pos.dtype)
    return torch.cat([zeros, psi_pos], dim=1)     # (N, K)


def ordinal_ce_from_params(
    theta: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Convenience: (theta, alpha, beta) -> GPCM logits -> cumulative-link CE."""
    return cumulative_link_ce(gpcm_logits(theta, alpha, beta), targets, reduction)


# ---------------------------------------------------------------------------
# nn.Module wrapper (drop-in for a training loss slot)
# ---------------------------------------------------------------------------

class OrdinalCELoss(nn.Module):
    """Module wrapper around :func:`cumulative_link_ce` for GPCM logits.

    ``_p2_model._build_loss_fn`` swaps this in for GPCM in place of WOL when
    ``train.gpcm_loss == "ordinal_ce"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return cumulative_link_ce(logits, targets, reduction=self.reduction)
