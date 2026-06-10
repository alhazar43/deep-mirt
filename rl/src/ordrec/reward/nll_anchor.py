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

from typing import Any, Optional

import torch
from torch import Tensor

from .gpcm_ops import gpcm_log_probs


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

    K = beta_table.shape[-1] + 1

    # Validate response range up-front; gather raises on out-of-range
    # only with -1, which is not a useful guard.
    if (probe_responses < 0).any() or (probe_responses >= K).any():
        raise ValueError(
            f"probe_responses must be in [0, K - 1] = [0, {K - 1}]."
        )

    # Delegate GPCM log-prob computation to the shared kernel.
    log_probs = gpcm_log_probs(
        theta, probe_ids, alpha_table, beta_table, logit_clip=logit_clip,
    )  # (B, H, K)

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


def resample_probe_responses(
    world_model: "Any",
    history_q: Tensor,
    history_r: Tensor,
    probe_H_ids: Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample probe responses from the world model at the current history.

    Used by the E4.7 dynamic-world anchor
    (``RewardConfig.resample_probe_at_terminal = True``) to draw
    ``H_probe`` responses that reflect the student state the world model
    believes the student is in AFTER the full simulated session history,
    rather than the real historical-tail responses captured at episode
    reset.

    Anti-gaming invariant: ``probe_H_ids`` must be masked from the action
    set throughout the episode (enforced by :class:`OrdRecEnv`'s probe
    mask). This function only queries the world model's predictive
    distribution at those items; it does NOT administer them.

    Algorithm:

    1. Append ``probe_H_ids`` to ``history_q`` with zero placeholder
       responses (the world model conditions on the question ids, not
       the responses, when predicting the next-step distribution).
    2. Run one frozen forward pass.
    3. Sample one response per probe item from the predicted
       probabilities.

    Args:
        world_model: A :class:`~ordrec.envs.frozen_magpcm.FrozenMAGPCM`
            instance (or anything with ``forward_no_grad(q, r) -> dict``
            that returns a ``probs (B, S, K)`` tensor).
        history_q: ``LongTensor (B, S)`` question ids of the current
            simulated history.
        history_r: ``LongTensor (B, S)`` corresponding responses.
        probe_H_ids: ``LongTensor (B, H)`` probe item ids.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        ``LongTensor (B, H)`` sampled responses in ``[0, K - 1]``.
    """
    B, H = probe_H_ids.shape
    device = history_q.device

    # Append probe items to the history with zero placeholder responses.
    placeholder_r = torch.zeros(B, H, dtype=torch.long, device=device)
    extended_q = torch.cat([history_q, probe_H_ids], dim=1)
    extended_r = torch.cat([history_r, placeholder_r], dim=1)

    with torch.no_grad():
        out = world_model.forward_no_grad(extended_q, extended_r)

    # Take only the last H positions (the probe items).
    probe_probs = out["probs"][:, -H:, :]  # (B, H, K)
    K = probe_probs.shape[-1]
    flat = probe_probs.reshape(-1, K).clamp_min(1e-12)
    sampled = torch.multinomial(flat, num_samples=1, generator=generator)
    return sampled.view(B, H).to(dtype=torch.long)


__all__ = ["gpcm_nll", "resample_probe_responses", "terminal_anchor"]
