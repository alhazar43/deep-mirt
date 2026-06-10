"""Behaviour-cloning warm-start for the PPO actor.

The actor head is trained to mimic a teacher policy that scores items
by maximum Fisher information (Muraki 1993)::

    I(q, theta) = E[score(theta)^2 | theta, q]
                = sum_k k^2 P_qk(theta) - (sum_k k P_qk(theta))^2

Rather than a hard argmax target (one-hot on the single top item), the
BC loss uses a **top-k soft target**: the teacher distribution is the
renormalised Fisher scores over the top-5 most informative items, with
all other items given zero probability. This soft target is more
robust than the argmax when several items have nearly identical
Fisher information, avoids misleading log-prob=-inf on a minority of
seeds, and provides a richer training signal across the K items.

Match-rate reporting uses top-5 overlap (``teacher_top5_overlap``)
instead of argmax equality, since the soft target does not commit to a
single item.

See ``docs/ordrec_impl_guide.md`` Section 4.5 (item 20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from ..envs.ordrec_env import OrdRecEnv
from ..training.ppo import PPO, _masked_logits


# ---------------------------------------------------------------------------
# GPCM Fisher information
# ---------------------------------------------------------------------------


def gpcm_item_information(
    theta: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Per-item GPCM Fisher information evaluated at ``theta``.

    Muraki 1993, "Information Functions of the Generalized Partial
    Credit Model" (Applied Psychological Measurement).

    Args:
        theta: ``Tensor (B, D)`` latent ability.
        alpha_table: ``Tensor (Q + 1, D)``.
        beta_table: ``Tensor (Q + 1, K - 1)``.
        eps: Small floor preventing ``log(0)`` style issues. Only
            applied on the probability side of any ``p log p`` term.

    Returns:
        ``Tensor (B, Q + 1)`` per-item information. Row 0 (the pad
        slot) is masked downstream; we fill it with ``0`` so the
        teacher never selects it even before mask composition.

    Shapes::

        theta (B, D), alpha (Q+1, D), beta (Q+1, K-1) -> (B, Q+1)
    """
    if theta.ndim != 2:
        raise ValueError(
            f"theta must be 2D (B, D), got shape {tuple(theta.shape)}"
        )
    B, D = theta.shape
    Qp1, K_minus_1 = beta_table.shape
    K = K_minus_1 + 1
    if alpha_table.shape != (Qp1, D):
        raise ValueError(
            f"alpha_table shape {tuple(alpha_table.shape)} must equal "
            f"(Q+1={Qp1}, D={D})."
        )

    # All-items expansion: (B, Q+1, D), (B, Q+1, K-1).
    alpha = alpha_table.to(dtype=theta.dtype).unsqueeze(0).expand(B, -1, -1)
    beta = beta_table.to(dtype=theta.dtype).unsqueeze(0).expand(B, -1, -1)

    interaction = (alpha * theta.unsqueeze(1)).sum(dim=-1)  # (B, Q+1)
    alpha_norm = alpha.norm(dim=-1)                          # (B, Q+1)

    step_values = (
        interaction.unsqueeze(-1)
        - alpha_norm.unsqueeze(-1) * beta
    )  # (B, Q+1, K-1)
    cum_logits = step_values.cumsum(dim=-1)
    zeros = torch.zeros(B, Qp1, 1, device=theta.device, dtype=theta.dtype)
    logits = torch.cat([zeros, cum_logits], dim=-1)         # (B, Q+1, K)
    logits = logits.clamp(min=-50.0, max=50.0)
    probs = F.softmax(logits, dim=-1)
    probs = probs.clamp_min(eps)

    # k as a (K,) row.
    k = torch.arange(K, device=theta.device, dtype=theta.dtype)
    e_k = (probs * k).sum(dim=-1)               # (B, Q+1)
    e_k2 = (probs * k.pow(2)).sum(dim=-1)        # (B, Q+1)
    info = e_k2 - e_k.pow(2)                     # (B, Q+1)
    # Zero out the pad slot so teacher argmax never lands there.
    info[:, 0] = 0.0
    return info


# ---------------------------------------------------------------------------
# Max-Fisher teacher
# ---------------------------------------------------------------------------


def max_fisher_actions(
    theta: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    action_mask: Tensor,
) -> Tensor:
    """Greedy maximum-Fisher item per row, masked.

    Args:
        theta: ``Tensor (B, D)``.
        alpha_table: ``Tensor (Q + 1, D)``.
        beta_table: ``Tensor (Q + 1, K - 1)``.
        action_mask: ``BoolTensor (B, Q + 1)``. ``True`` allows the
            item. Pad slot must be ``False``; probe ids must be
            ``False``.

    Returns:
        ``LongTensor (B,)`` of teacher actions (argmax of Fisher info).
    """
    info = gpcm_item_information(theta, alpha_table, beta_table)
    if action_mask.shape != info.shape:
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} must match "
            f"info shape {tuple(info.shape)}."
        )
    masked = info.masked_fill(~action_mask, float("-inf"))
    return masked.argmax(dim=-1)


def top_k_fisher_soft_target(
    theta: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    action_mask: Tensor,
    *,
    k: int = 5,
) -> Tensor:
    """Top-k Fisher soft target distribution.

    Returns a probability vector over the full action space where
    only the top-k items (by masked Fisher information) receive
    non-zero probability. Within the top-k the distribution is
    proportional to the Fisher information scores, renormalised to
    sum to 1.0.

    Using this as the BC target instead of a one-hot argmax is more
    robust when several items have similar Fisher information and
    avoids the zero-gradient pathology for near-optimal items just
    outside rank 1.

    Args:
        theta: ``Tensor (B, D)``.
        alpha_table: ``Tensor (Q + 1, D)``.
        beta_table: ``Tensor (Q + 1, K - 1)``.
        action_mask: ``BoolTensor (B, Q + 1)``.
        k: Number of top items to include in the soft target.

    Returns:
        ``Tensor (B, Q + 1)`` soft probability distribution.
        Rows sum to 1.0. Disallowed positions are exactly zero.
    """
    info = gpcm_item_information(theta, alpha_table, beta_table)
    if action_mask.shape != info.shape:
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} must match "
            f"info shape {tuple(info.shape)}."
        )
    # Zero out disallowed positions.
    info_masked = info.clone()
    info_masked[~action_mask] = float("-inf")

    # Identify top-k positions per row.
    k_eff = min(k, int(action_mask.sum(dim=-1).min().item()))
    k_eff = max(1, k_eff)
    topk_vals, topk_idx = info_masked.topk(k_eff, dim=-1)  # (B, k)

    # Build soft target: renormalised Fisher scores at top-k positions.
    # Clamp any -inf values (shouldn't occur if k_eff <= allowed count).
    topk_vals = topk_vals.clamp_min(0.0)
    denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    topk_probs = topk_vals / denom  # (B, k)

    # Scatter into a full (B, Q+1) distribution.
    soft = torch.zeros_like(info)
    soft.scatter_add_(dim=1, index=topk_idx, src=topk_probs)
    return soft


# ---------------------------------------------------------------------------
# BC loss and one warm-start step
# ---------------------------------------------------------------------------


@dataclass
class BCStats:
    """Per-update behaviour-cloning diagnostics.

    Attributes:
        bc_loss: Cross-entropy loss between student and soft teacher.
        teacher_match_rate: Fraction of rows where the student argmax
            falls within the teacher's top-5 Fisher items. Kept for
            backward-compat logging; prefer ``teacher_top5_overlap``.
        teacher_top5_overlap: Fraction of rows where the student argmax
            is one of the top-5 Fisher items (same as
            ``teacher_match_rate`` for k=5, named explicitly for
            clarity).
        entropy: Mean student entropy.
        n_examples: Number of (student, teacher) pairs in the update.
    """

    bc_loss: float
    teacher_match_rate: float
    teacher_top5_overlap: float
    entropy: float
    n_examples: int


def bc_loss_step(
    ppo: PPO,
    obs: Tensor,
    action_mask: Tensor,
    teacher_soft: Tensor,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    top_k: int = 5,
) -> Tuple[Tensor, BCStats]:
    """Single behaviour-cloning gradient step with a soft teacher target.

    The student is the PPO actor. The loss is the cross-entropy between
    the student's masked distribution and the teacher's top-k soft
    target. We do not update the critic here; the critic gets the
    static MVE warm-start in ``static_mve.py``.

    Using a soft target (renormalised Fisher over top-k items) rather
    than a one-hot argmax is more robust when several items have nearly
    identical Fisher information and avoids the misleading log-prob=-inf
    pathology.

    Args:
        ppo: A constructed :class:`PPO` instance. The optimizer is
            reused if ``optimizer`` is ``None``.
        obs: ``Tensor (B, obs_dim)`` policy observations.
        action_mask: ``BoolTensor (B, n_actions)`` matching the env's
            mask at the same step.
        teacher_soft: ``FloatTensor (B, n_actions)`` soft target
            distribution summing to 1.0 per row. Produced by
            :func:`top_k_fisher_soft_target`. Rows must sum to 1.
        optimizer: Optional override of the PPO optimizer. Useful in
            tests that want isolated parameter steps.
        top_k: Number of top teacher items. Used only for the
            ``teacher_top5_overlap`` metric computation; the loss
            itself uses ``teacher_soft`` as provided.

    Returns:
        ``(loss, stats)``.
    """
    if obs.ndim != 2:
        raise ValueError(
            f"obs must be 2D (B, obs_dim), got {tuple(obs.shape)}"
        )
    if teacher_soft.shape[0] != obs.shape[0]:
        raise ValueError(
            f"teacher_soft batch dim {teacher_soft.shape[0]} != "
            f"obs batch dim {obs.shape[0]}."
        )
    if teacher_soft.ndim != 2:
        raise ValueError(
            f"teacher_soft must be 2D (B, n_actions), got "
            f"{tuple(teacher_soft.shape)}."
        )
    opt = optimizer if optimizer is not None else ppo.optimizer
    obs = obs.to(ppo.device)
    action_mask = action_mask.to(ppo.device)
    teacher_soft = teacher_soft.to(ppo.device)

    logits, _ = ppo.policy(obs)
    masked = _masked_logits(logits, action_mask)

    # KL-divergence loss: sum_a teacher_soft[a] * (-log student[a]).
    # Equivalent to cross-entropy H(teacher, student) when teacher sums
    # to 1. ``log_softmax`` handles -inf for disallowed actions; the
    # soft target already assigns zero probability there.
    log_probs = F.log_softmax(masked, dim=-1)
    loss = -(teacher_soft * log_probs).sum(dim=-1).mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        ppo.policy.parameters(), ppo.max_grad_norm,
    )
    opt.step()

    with torch.no_grad():
        student = Categorical(logits=masked)
        argmax = masked.argmax(dim=-1)  # student greedy pick
        # teacher top-k: positions with non-zero probability.
        teacher_topk = teacher_soft > 0.0
        # Top-5 overlap: fraction of rows where student argmax is in
        # the teacher's top-k items.
        overlap = teacher_topk.gather(
            dim=1, index=argmax.unsqueeze(1),
        ).squeeze(1).float().mean()
        entropy = student.entropy().mean()

    return loss.detach(), BCStats(
        bc_loss=float(loss.detach().item()),
        teacher_match_rate=float(overlap.item()),
        teacher_top5_overlap=float(overlap.item()),
        entropy=float(entropy.item()),
        n_examples=int(obs.shape[0]),
    )


def bc_warmstart(
    ppo: PPO,
    env: OrdRecEnv,
    *,
    n_updates: int = 20,
    n_episodes_per_update: int = 4,
    seed: int = 0,
    teacher_top_k: int = 5,
) -> list[BCStats]:
    """Warm-start ``ppo.policy`` against the top-k Fisher soft teacher.

    For each update we reset the env, then at each step we compute the
    teacher's top-k soft target (renormalised Fisher information over
    the top-5 items) under the env's mask, build a minibatch of
    ``(obs, mask, teacher_soft)`` tuples, and run one BC gradient step.

    The soft target is more robust than a hard argmax when several
    items have similar Fisher information.

    Args:
        ppo: Constructed PPO.
        env: :class:`OrdRecEnv`. Provides the item cache, the action
            mask machinery and ``theta_t``.
        n_updates: Number of warm-start updates.
        n_episodes_per_update: Episodes to collect per BC update.
        seed: Seed used for the env's resets.
        teacher_top_k: Number of top Fisher items in the soft target.
            Default 5 (top-5 soft target).

    Returns:
        List of :class:`BCStats`, one per update.
    """
    history: list[BCStats] = []
    K_B = int(getattr(env, "K_B", 1))
    for it in range(int(n_updates)):
        obs_buf: list[Tensor] = []
        mask_buf: list[Tensor] = []
        soft_buf: list[Tensor] = []
        for ep_idx in range(int(n_episodes_per_update)):
            state = env.reset(seed=seed + it * 1000 + ep_idx)
            done = False
            while not done:
                obs = state.to_tensor(ppo.device)  # (B, obs_dim)
                mask = state.action_mask.to(ppo.device)  # (B, Q+1)
                # Build K_B soft targets sequentially. The mask tightens
                # after each pick so the teacher never selects an item
                # that's already been chosen in this block.
                cur_mask = mask.clone()
                teacher_cols: list[Tensor] = []
                soft_cols: list[Tensor] = []
                for _ in range(K_B):
                    # Soft target over top-k Fisher items under cur_mask.
                    soft = top_k_fisher_soft_target(
                        state.theta_t.to(ppo.device),
                        env.alpha_table,
                        env.beta_table,
                        cur_mask,
                        k=teacher_top_k,
                    )
                    soft_cols.append(soft)
                    # Hard pick for env.step: argmax of Fisher info.
                    teacher_a = max_fisher_actions(
                        state.theta_t.to(ppo.device),
                        env.alpha_table,
                        env.beta_table,
                        cur_mask,
                    )
                    teacher_cols.append(teacher_a)
                    cur_mask = cur_mask.clone()
                    cur_mask.scatter_(
                        dim=1, index=teacher_a.unsqueeze(1),
                        src=torch.zeros(
                            teacher_a.shape[0], 1,
                            dtype=torch.bool, device=cur_mask.device,
                        ),
                    )
                teacher_action = torch.stack(teacher_cols, dim=1)
                # Store per-sub-step BC examples with the sub-step's
                # soft target. The mask tightens progressively.
                running_mask = mask.clone()
                for k in range(K_B):
                    obs_buf.append(obs.detach().cpu())
                    mask_buf.append(running_mask.detach().cpu())
                    soft_buf.append(soft_cols[k].detach().cpu())
                    running_mask = running_mask.clone()
                    running_mask.scatter_(
                        dim=1, index=teacher_cols[k].unsqueeze(1),
                        src=torch.zeros(
                            teacher_cols[k].shape[0], 1,
                            dtype=torch.bool, device=running_mask.device,
                        ),
                    )
                state, _, done, _ = env.step(teacher_action)
        if not obs_buf:
            continue
        obs_t = torch.cat(obs_buf, dim=0)
        mask_t = torch.cat(mask_buf, dim=0)
        soft_t = torch.cat(soft_buf, dim=0)
        _, stats = bc_loss_step(ppo, obs_t, mask_t, soft_t,
                                top_k=teacher_top_k)
        history.append(stats)
    return history


__all__ = [
    "BCStats",
    "bc_loss_step",
    "bc_warmstart",
    "gpcm_item_information",
    "max_fisher_actions",
    "top_k_fisher_soft_target",
]
