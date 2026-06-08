"""Static ``K ** K_B`` MVE warm-start for the PPO critic.

The Multiple-Value-Expectation warm-start estimates ``V(s)`` by
expanding the exact joint distribution over the next ``K_B`` items
the teacher would administer, then averaging the probe-entropy
reward under every joint outcome.

For each ``(state, teacher_action_batch)`` pair we form ``K ** K_B``
outcomes. ``K = 4`` and ``K_B = 5`` give 1024 outcomes. Each
outcome is a length-``K_B`` ordinal vector ``y in [0, K-1]^K_B``.
Under the frozen world model the joint probability ``P(y | theta_t,
a_1, ..., a_K_B)`` factorises as a product of GPCM per-item
probabilities evaluated at the teacher's items. We freeze ``theta_t``
between sub-steps, which is the static MVE simplification (the
"static" name); the next-state critic target then equals

    V_MVE(s) = sum_y joint_prob(y) * phi(theta_t, probe_C, ...)

where ``phi`` is evaluated against the fixed probe set captured at
``env.reset``. With theta frozen during expansion, ``phi`` is the
same constant under every outcome, so the warm-start collapses to a
single deterministic target. This is the v1 simplification mentioned
in the impl guide; v2 expands ``theta(y)`` per-outcome through the
frozen encoder.

The regression target is computed with autograd disabled (the world
model is frozen) and the critic is fit with an MSE loss for a small
number of steps. The actor is not touched here; the BC warm-start in
``bc.py`` handles the actor side.

See ``docs/ordrec_impl_guide.md`` Section 4.5 (item 21).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..envs.ordrec_env import OrdRecEnv
from ..reward.probe_entropy import phi_entropy
from ..training.ppo import PPO
from .bc import max_fisher_actions


@dataclass
class MVEStats:
    """Per-update static-MVE warm-start diagnostics."""

    critic_loss: float
    target_mean: float
    target_std: float
    n_examples: int


def static_mve_target(
    theta: Tensor,
    teacher_action: Tensor,
    alpha_table: Tensor,
    beta_table: Tensor,
    probe_C_ids: Tensor,
    *,
    n_categories: int,
    eps: float = 1e-12,
) -> Tensor:
    """Static ``K ** K_B`` MVE target for ``V(theta)``.

    The joint over the next ``K_B`` GPCM outcomes factorises into
    per-item categoricals because the frozen world model evaluates
    each item independently at ``theta``. Under the static-theta
    simplification ``phi(theta)`` is constant across outcomes, so the
    expectation reduces to ``phi(theta, probe_C, ...)`` itself. The
    function still computes the joint expansion explicitly so that
    callers can pass an outcome-dependent value function in future
    expansions without changing the signature.

    Args:
        theta: ``Tensor (B, D)``.
        teacher_action: ``LongTensor (B, K_B)`` of next teacher items.
        alpha_table: ``Tensor (Q + 1, D)``.
        beta_table: ``Tensor (Q + 1, K - 1)``.
        probe_C_ids: ``LongTensor (B, M)`` probe items for ``phi``.
        n_categories: ``K``.
        eps: Clamp guard inside ``phi``.

    Returns:
        ``Tensor (B,)`` static-MVE value target.
    """
    B, K_B = teacher_action.shape
    K = int(n_categories)
    if alpha_table.shape[-1] != theta.shape[-1]:
        raise ValueError(
            f"alpha_table trait dim {alpha_table.shape[-1]} must equal "
            f"theta trait dim {theta.shape[-1]}."
        )

    # Per-item GPCM probabilities along the next K_B teacher items.
    # alpha_q, beta_q -> (B, K_B, D), (B, K_B, K-1)
    alpha_q = alpha_table[teacher_action]
    beta_q = beta_table[teacher_action]
    interaction = (alpha_q * theta.unsqueeze(1)).sum(dim=-1)  # (B, K_B)
    alpha_norm = alpha_q.norm(dim=-1)                          # (B, K_B)
    step_values = (
        interaction.unsqueeze(-1)
        - alpha_norm.unsqueeze(-1) * beta_q
    )                                                          # (B, K_B, K-1)
    cum_logits = step_values.cumsum(dim=-1)
    zeros = torch.zeros(B, K_B, 1, device=theta.device, dtype=theta.dtype)
    logits = torch.cat([zeros, cum_logits], dim=-1).clamp(-50.0, 50.0)
    log_probs = F.log_softmax(logits, dim=-1)                  # (B, K_B, K)

    # Joint over all K ** K_B outcomes. Construct the outcome tensor
    # explicitly (small for K=4, K_B=5 -> 1024 outcomes).
    outcomes = torch.tensor(
        list(product(range(K), repeat=int(K_B))),
        dtype=torch.long, device=theta.device,
    )  # (K**K_B, K_B)
    n_outcomes = outcomes.shape[0]

    # Joint log-probability of each outcome: sum across K_B of the
    # selected per-item log-probs. log_probs gathered at outcomes.
    # Expand log_probs to (B, K**K_B, K_B, K) and gather. To keep
    # memory bounded we loop over the K_B axis instead.
    log_joint = torch.zeros(B, n_outcomes, device=theta.device, dtype=theta.dtype)
    for k_idx in range(int(K_B)):
        col = outcomes[:, k_idx]  # (n_outcomes,)
        # gather along K -> (B, n_outcomes)
        lp_k = log_probs[:, k_idx, :]  # (B, K)
        log_joint = log_joint + lp_k[:, col]
    joint = log_joint.exp()  # (B, n_outcomes)
    # Sanity normalisation; floating-point drift can leave row sums
    # slightly off 1.0.
    joint = joint / joint.sum(dim=-1, keepdim=True).clamp_min(eps)

    # phi(theta) over the fixed probe set is constant under outcomes
    # in the v1 static-theta simplification. The expectation is just
    # phi itself, but we still combine through joint for forward
    # compatibility with outcome-dependent value functions.
    phi = phi_entropy(theta, probe_C_ids, alpha_table, beta_table, eps=eps)
    phi_ex = phi.unsqueeze(-1).expand(-1, n_outcomes)
    return (joint * phi_ex).sum(dim=-1)


def static_mve_critic_step(
    ppo: PPO,
    obs: Tensor,
    target: Tensor,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> MVEStats:
    """One MSE update on the critic head against the MVE target.

    Args:
        ppo: PPO instance whose critic head we train.
        obs: ``Tensor (B, obs_dim)``.
        target: ``Tensor (B,)`` MVE target.
        optimizer: Optional optimizer override; defaults to ``ppo.optimizer``.

    Returns:
        :class:`MVEStats`.
    """
    if obs.ndim != 2:
        raise ValueError(
            f"obs must be 2D (B, obs_dim), got {tuple(obs.shape)}"
        )
    if target.ndim != 1 or target.shape[0] != obs.shape[0]:
        raise ValueError(
            f"target must be 1D (B,), got {tuple(target.shape)} for "
            f"B={obs.shape[0]}."
        )
    opt = optimizer if optimizer is not None else ppo.optimizer
    obs = obs.to(ppo.device)
    target = target.to(ppo.device).to(torch.float32)

    _, value = ppo.policy(obs)
    loss = F.mse_loss(value, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        ppo.policy.parameters(), ppo.max_grad_norm,
    )
    opt.step()
    return MVEStats(
        critic_loss=float(loss.detach().item()),
        target_mean=float(target.mean().item()),
        target_std=float(target.std().item()) if target.numel() > 1 else 0.0,
        n_examples=int(obs.shape[0]),
    )


def mve_warmstart_critic(
    ppo: PPO,
    env: OrdRecEnv,
    *,
    n_updates: int = 10,
    n_episodes_per_update: int = 4,
    seed: int = 0,
) -> List[MVEStats]:
    """Critic warm-start via static-MVE targets along teacher trajectories.

    For each update we drive the env with the max-Fisher teacher,
    collect ``(obs, theta_t, teacher_action, probe_C)`` tuples at
    every env step, compute the static-MVE target ``V_MVE`` for each
    tuple, and fit the critic by MSE against those targets.

    Args:
        ppo: PPO instance.
        env: :class:`OrdRecEnv`.
        n_updates: Number of MVE updates.
        n_episodes_per_update: Episodes per update.
        seed: Env reset seed.

    Returns:
        List of :class:`MVEStats`, one per update.
    """
    history: List[MVEStats] = []
    K_B = int(getattr(env, "K_B", 1))
    n_categories = env.n_categories
    for it in range(int(n_updates)):
        obs_buf: List[Tensor] = []
        target_buf: List[Tensor] = []
        for ep_idx in range(int(n_episodes_per_update)):
            state = env.reset(seed=seed + 1_000_000 + it * 1000 + ep_idx)
            done = False
            while not done:
                obs = state.to_tensor(ppo.device)
                theta = state.theta_t.to(ppo.device)
                mask = state.action_mask.to(ppo.device)
                # Teacher batch via sequential max-Fisher under the
                # tightening mask.
                cur_mask = mask.clone()
                cols = []
                for _ in range(K_B):
                    a = max_fisher_actions(
                        theta, env.alpha_table, env.beta_table, cur_mask,
                    )
                    cols.append(a)
                    cur_mask = cur_mask.clone()
                    cur_mask.scatter_(
                        dim=1, index=a.unsqueeze(1),
                        src=torch.zeros(
                            a.shape[0], 1,
                            dtype=torch.bool, device=cur_mask.device,
                        ),
                    )
                teacher_action = torch.stack(cols, dim=1)
                with torch.no_grad():
                    target = static_mve_target(
                        theta=theta,
                        teacher_action=teacher_action,
                        alpha_table=env.alpha_table,
                        beta_table=env.beta_table,
                        probe_C_ids=state.raw_info["probe_C_ids"].to(ppo.device),
                        n_categories=n_categories,
                    )
                obs_buf.append(obs.detach().cpu())
                target_buf.append(target.detach().cpu())
                state, _, done, _ = env.step(teacher_action)
        if not obs_buf:
            continue
        obs_t = torch.cat(obs_buf, dim=0)
        target_t = torch.cat(target_buf, dim=0)
        stats = static_mve_critic_step(ppo, obs_t, target_t)
        history.append(stats)
    return history


__all__ = [
    "MVEStats",
    "mve_warmstart_critic",
    "static_mve_critic_step",
    "static_mve_target",
]
