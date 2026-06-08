"""Behaviour-cloning warm-start for the PPO actor.

The actor head is trained to mimic a deterministic teacher policy
that selects items by maximum Fisher information. Muraki (1993)
gives the GPCM item-information formula::

    I(q, theta) = E[score(theta)^2 | theta, q]
                = sum_k k^2 P_qk(theta) - (sum_k k P_qk(theta))^2

This is the score variance of the GPCM at ``theta``. Higher
``I(q, theta)`` selects the item whose response best distinguishes
abilities in a Cramer-Rao sense. The Lindley/Owen lens used in the
reward design reduces to maximum Fisher information when the probe
``C`` collapses to a single candidate item, so the teacher is a
principled warm-start anchor even when the PPO reward involves a
larger probe (see plan Section 3.5).

The teacher is masked by the env's action mask so it never picks
probe items or already-administered items. The student is the PPO
actor; the BC loss is the negative log-likelihood of the teacher's
choice under the student's masked categorical.

The full impl-guide design specifies a 50/30/20 mixture over
``max-Fisher / ReflectionLayer-greedy / Thompson``. The other two
teachers depend on infrastructure (ReflectionLayer, posterior
sampling) we have not built. E4 ships the max-Fisher teacher only;
the mixture is a v2 enhancement.

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
        ``LongTensor (B,)`` of teacher actions.
    """
    info = gpcm_item_information(theta, alpha_table, beta_table)
    if action_mask.shape != info.shape:
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} must match "
            f"info shape {tuple(info.shape)}."
        )
    masked = info.masked_fill(~action_mask, float("-inf"))
    return masked.argmax(dim=-1)


# ---------------------------------------------------------------------------
# BC loss and one warm-start step
# ---------------------------------------------------------------------------


@dataclass
class BCStats:
    """Per-update behaviour-cloning diagnostics."""

    bc_loss: float
    teacher_match_rate: float
    entropy: float
    n_examples: int


def bc_loss_step(
    ppo: PPO,
    obs: Tensor,
    action_mask: Tensor,
    teacher_actions: Tensor,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[Tensor, BCStats]:
    """Single behaviour-cloning gradient step.

    The student is the PPO actor. Loss is the cross-entropy between
    the masked categorical and the teacher's one-hot. We do not
    update the critic here; the critic gets the static MVE warm-start
    in ``static_mve.py``.

    Args:
        ppo: A constructed :class:`PPO` instance. The optimizer is
            reused if ``optimizer`` is ``None``.
        obs: ``Tensor (B, obs_dim)`` policy observations.
        action_mask: ``BoolTensor (B, n_actions)`` matching the env's
            mask at the same step.
        teacher_actions: ``LongTensor (B,)`` of teacher choices.
        optimizer: Optional override of the PPO optimizer. Useful in
            tests that want isolated parameter steps.

    Returns:
        ``(loss, stats)``.
    """
    if obs.ndim != 2:
        raise ValueError(
            f"obs must be 2D (B, obs_dim), got {tuple(obs.shape)}"
        )
    if teacher_actions.ndim != 1 or teacher_actions.shape[0] != obs.shape[0]:
        raise ValueError(
            f"teacher_actions must be 1D (B,), got "
            f"{tuple(teacher_actions.shape)} for obs B={obs.shape[0]}."
        )
    opt = optimizer if optimizer is not None else ppo.optimizer
    obs = obs.to(ppo.device)
    action_mask = action_mask.to(ppo.device)
    teacher_actions = teacher_actions.to(ppo.device)

    logits, _ = ppo.policy(obs)
    masked = _masked_logits(logits, action_mask)
    # Cross-entropy under the masked distribution. ``log_softmax``
    # handles ``-inf`` rows by yielding ``-inf`` on disallowed actions,
    # which is fine because the teacher's action is always allowed.
    log_probs = F.log_softmax(masked, dim=-1)
    nll = -log_probs.gather(1, teacher_actions.unsqueeze(1)).squeeze(1)
    loss = nll.mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        ppo.policy.parameters(), ppo.max_grad_norm,
    )
    opt.step()

    with torch.no_grad():
        student = Categorical(logits=masked)
        argmax = masked.argmax(dim=-1)
        match = (argmax == teacher_actions).float().mean()
        entropy = student.entropy().mean()
    return loss.detach(), BCStats(
        bc_loss=float(loss.detach().item()),
        teacher_match_rate=float(match.item()),
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
) -> list[BCStats]:
    """Warm-start ``ppo.policy`` against the max-Fisher teacher.

    For each update we reset the env, then at each step we ask the
    teacher for its max-Fisher choice under the env's mask, sample a
    minibatch of ``(obs, mask, teacher_action)`` tuples, and run one
    BC gradient step. The reward channel is not used; the env is
    driven by the teacher so the trajectories the student learns to
    match are also the ones the teacher would have produced.

    Args:
        ppo: Constructed PPO.
        env: :class:`OrdRecEnv`. Provides the item cache, the action
            mask machinery and ``theta_t``.
        n_updates: Number of warm-start updates.
        n_episodes_per_update: Episodes to collect per BC update.
        seed: Seed used for the env's resets.

    Returns:
        List of :class:`BCStats`, one per update.
    """
    history: list[BCStats] = []
    K_B = int(getattr(env, "K_B", 1))
    for it in range(int(n_updates)):
        obs_buf: list[Tensor] = []
        mask_buf: list[Tensor] = []
        teach_buf: list[Tensor] = []
        for ep_idx in range(int(n_episodes_per_update)):
            state = env.reset(seed=seed + it * 1000 + ep_idx)
            done = False
            while not done:
                obs = state.to_tensor(ppo.device)  # (B, obs_dim)
                mask = state.action_mask.to(ppo.device)  # (B, Q+1)
                # Sample K_B teacher actions sequentially, updating
                # the mask after each pick so the env's no-repeat
                # invariant holds.
                cur_mask = mask.clone()
                teacher_cols: list[Tensor] = []
                for _ in range(K_B):
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
                # Store per-sub-step BC examples. The student sees the
                # same observation for each sub-step but the mask
                # tightens after each prior pick.
                running_mask = mask.clone()
                for k in range(K_B):
                    obs_buf.append(obs.detach().cpu())
                    mask_buf.append(running_mask.detach().cpu())
                    teach_buf.append(teacher_cols[k].detach().cpu())
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
        teach_t = torch.cat(teach_buf, dim=0)
        _, stats = bc_loss_step(ppo, obs_t, mask_t, teach_t)
        history.append(stats)
    return history


__all__ = [
    "BCStats",
    "bc_loss_step",
    "bc_warmstart",
    "gpcm_item_information",
    "max_fisher_actions",
]
