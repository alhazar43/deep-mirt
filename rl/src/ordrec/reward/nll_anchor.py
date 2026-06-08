"""Terminal NLL anchor on the held-out probe ``H_probe``.

At the episode horizon, the reward composer adds a value-of-information
term ``w_voi * (NLL(theta_0) - NLL(theta_T))`` where the NLL is computed
under the GPCM head against the observed responses on a held-out probe
``H_probe``. This rewards the policy for picking items that genuinely
improve the encoder's read of the student on items the policy never
administered.

See ``docs/ordrec_impl_guide.md`` Section 3.7 for the integration
point and Section 3.8 for the numerical-stability notes.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def gpcm_nll(
    theta: Tensor,
    probe_ids: Tensor,
    probe_responses: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    *,
    logit_clip: Optional[float] = 50.0,
) -> Tensor:
    """Average negative log-likelihood under the GPCM head.

    Computes ``-mean_h log P(Y_h = y_h | theta, alpha_h, beta_h)`` over
    the held-out probe ``H_probe`` per batch row. ``log P`` is read
    from ``log_softmax(logits)`` via ``gather``, so no ``log(softmax)``
    composition appears.

    Args:
        theta: ``Tensor (B, D)``.
        probe_ids: ``LongTensor (B, H)``.
        probe_responses: ``LongTensor (B, H)`` with values in
            ``[0, K - 1]``.
        alpha_table: ``Tensor (Q + 1, D)``.
        beta_table: ``Tensor (Q + 1, K - 1)``.
        logit_clip: Optional symmetric clamp on the cumulative logits.

    Returns:
        ``Tensor (B,)`` per-row average NLL in nats.

    Shape:
        ``theta (B, D)``, ``probe_ids (B, H)``,
        ``probe_responses (B, H)`` -> output ``(B,)``.
    """
    if theta.ndim != 2:
        raise ValueError(
            f"theta must be 2D (B, D), got shape {tuple(theta.shape)}"
        )
    if probe_ids.shape != probe_responses.shape:
        raise ValueError(
            f"probe_ids and probe_responses must share shape, got "
            f"{tuple(probe_ids.shape)} vs {tuple(probe_responses.shape)}."
        )
    if probe_ids.ndim != 2:
        raise ValueError(
            f"probe_ids must be 2D (B, H), got shape {tuple(probe_ids.shape)}"
        )

    B, D = theta.shape
    if probe_ids.shape[0] != B:
        raise ValueError(
            f"probe_ids batch dim {probe_ids.shape[0]} != theta batch dim {B}."
        )

    alpha_table = alpha_table.to(dtype=theta.dtype)
    beta_table = beta_table.to(dtype=theta.dtype)
    K = beta_table.shape[-1] + 1

    alpha_q = alpha_table[probe_ids]  # (B, H, D)
    beta_q = beta_table[probe_ids]  # (B, H, K-1)

    interaction = (alpha_q * theta.unsqueeze(1)).sum(dim=-1)  # (B, H)
    alpha_norm = alpha_q.norm(dim=-1)  # (B, H)
    step_values = interaction.unsqueeze(-1) - alpha_norm.unsqueeze(-1) * beta_q
    cum_logits = step_values.cumsum(dim=-1)  # (B, H, K-1)

    H = probe_ids.shape[1]
    zeros = torch.zeros(B, H, 1, device=theta.device, dtype=theta.dtype)
    logits = torch.cat([zeros, cum_logits], dim=-1)  # (B, H, K)

    if logit_clip is not None:
        logits = logits.clamp(min=-float(logit_clip), max=float(logit_clip))

    log_probs = torch.log_softmax(logits, dim=-1)  # (B, H, K)

    # Validate response range up-front, gather raises on out-of-range
    # only with -1, which is not a useful guard.
    if (probe_responses < 0).any() or (probe_responses >= K).any():
        raise ValueError(
            f"probe_responses must be in [0, K - 1] = [0, {K - 1}]."
        )

    chosen_log_prob = log_probs.gather(
        dim=-1, index=probe_responses.long().unsqueeze(-1)
    ).squeeze(-1)  # (B, H)

    return -chosen_log_prob.mean(dim=1)  # (B,)


def terminal_anchor(
    theta_t: Tensor,
    theta_0: Tensor,
    probe_H_ids: Tensor,
    probe_H_resp: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    *,
    logit_clip: Optional[float] = 50.0,
) -> Tensor:
    """``nll_prior - nll_terminal`` on the held-out probe.

    Sign convention matches the reward composer, a positive number
    means the terminal posterior fits the held-out responses better
    than the prior theta did, which is the channel the policy should
    maximise.

    Args:
        theta_t: ``Tensor (B, D)`` terminal-step theta.
        theta_0: ``Tensor (B, D)`` prior theta captured at reset.
        probe_H_ids, probe_H_resp: held-out probe ids and responses.
        alpha_table, beta_table: per-item ``(alpha, beta)`` tables.
        logit_clip: Symmetric clamp on the cumulative logits.

    Returns:
        ``Tensor (B,)``.
    """
    nll_prior = gpcm_nll(
        theta_0, probe_H_ids, probe_H_resp, alpha_table, beta_table,
        logit_clip=logit_clip,
    )
    nll_t = gpcm_nll(
        theta_t, probe_H_ids, probe_H_resp, alpha_table, beta_table,
        logit_clip=logit_clip,
    )
    return nll_prior - nll_t


__all__ = ["gpcm_nll", "terminal_anchor"]
