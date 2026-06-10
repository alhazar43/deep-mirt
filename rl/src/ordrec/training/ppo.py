"""PPO with clipped surrogate, value clipping and KL early stop.

Concrete :class:`RLAlgorithm` implementation for OrdRec. Follows the
CleanRL (Huang et al. 2022) recipe and the impl-guide hyperparameters
in Section 4.4. Key differences from CleanRL.

1. Discrete action over ``Q + 1`` items with a per-step action mask
   passed from the env. Mask-respecting categorical via logit
   ``masked_fill(-inf)``.
2. The action emitted by ``act`` is one item id sampled under the
   mask. The env interprets a ``K_B``-sized batch of consecutive
   actions as one ``step``; the algorithm runs ``K_B`` ``act`` calls
   in lockstep between two env steps. This keeps the buffer per-step
   storage canonical.
3. Two transitions per episode at ``K_B = 5, T = 10``. Many
   episodes (32) per update and few (4) epochs per update.

Update body (loss formulas from Schulman et al. 2017).

```
ratio       = exp(new_lp - old_lp)
unclipped   = ratio * advantages
clipped     = clamp(ratio, 1 - eps, 1 + eps) * advantages
policy_loss = -min(unclipped, clipped).mean()

v_clip      = old_values + clamp(values - old_values, -eps, eps)
value_loss  = 0.5 * max((values - returns)^2, (v_clip - returns)^2).mean()

loss = policy_loss + value_coef * value_loss - ent_coef * entropy
```

See ``docs/ordrec_impl_guide.md`` Section 4.4 for the full table.

The DQN and SAC entries in this directory are intentionally omitted
in E4; placeholders live below as comments per the impl guide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from ..envs.base import OrdinalState
from .base import RLAlgorithm, RolloutStats, UpdateStats
from .rollout import RolloutBuffer
from .utils import linear_anneal


# ---------------------------------------------------------------------------
# Actor-critic head
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-trunk discrete actor-critic.

    A small MLP trunk feeds two heads.

    - Actor head, ``(hidden -> n_actions)`` linear, masked categorical.
    - Critic head, ``(hidden -> 1)`` linear scalar value estimate.

    The trunk is shared so policy gradients backpropagate through it
    on every update. Hidden width defaults to 128 per the impl guide.
    """

    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        *,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError(
                f"observation_dim must be > 0, got {observation_dim}"
            )
        if n_actions <= 1:
            raise ValueError(f"n_actions must be > 1, got {n_actions}")

        dims = [observation_dim] + [hidden_dim] * n_hidden_layers
        trunk: list[nn.Module] = []
        for i in range(len(dims) - 1):
            trunk.append(nn.Linear(dims[i], dims[i + 1]))
            trunk.append(nn.Tanh())
        self.trunk = nn.Sequential(*trunk)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal init with small actor gain. Standard PPO recipe
        # (Engstrom et al. 2020, "Implementation Matters in Deep
        # Policy Gradients") for stable early training.
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(logits (B, n_actions), value (B,))``."""
        h = self.trunk(obs)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


# ---------------------------------------------------------------------------
# Masked categorical helper
# ---------------------------------------------------------------------------


def _masked_logits(logits: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Apply a boolean mask to logits with ``masked_fill(-inf)``.

    Args:
        logits: ``Tensor (B, n_actions)``.
        mask: ``BoolTensor (B, n_actions)`` or ``None``. ``True``
            indicates allowed actions. When ``None`` the logits are
            returned unchanged.

    Returns:
        Masked logits. Rows with no allowed action have all-``-inf``
        logits, which downstream samplers must handle (PPO's policy
        sees masks built by the env, which guarantees at least one
        allowed action per row, so this case is only relevant in
        tests).
    """
    if mask is None:
        return logits
    if mask.shape != logits.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match logits "
            f"{tuple(logits.shape)}"
        )
    neg_inf = torch.full_like(logits, float("-inf"))
    return torch.where(mask, logits, neg_inf)


# ---------------------------------------------------------------------------
# PPO algorithm
# ---------------------------------------------------------------------------


class PPO(RLAlgorithm):
    """PPO with clipped surrogate, value clipping and KL early stop.

    Args:
        observation_dim: Policy input dimension.
        action_dim: ``Q + 1`` for OrdRec.
        device: Torch device for the policy.
        seed: Seeds a local ``torch.Generator`` used for action
            sampling. The constructor does NOT touch global RNG
            state; callers that need deterministic weight init must
            seed at the script or test level before construction.
        hidden_dim: Trunk width.
        n_hidden_layers: Number of trunk layers.
        learning_rate: Adam learning rate. Default ``3e-4``.
        adam_eps: Adam epsilon. Default ``1e-5``.
        clip_eps: Both the policy and value clip range. Default ``0.2``.
        gamma: GAE discount.
        gae_lambda: GAE lambda.
        entropy_coef_initial: ``ent0``.
        entropy_coef_final: ``ent_T``.
        entropy_anneal_fraction: Fraction of ``total_updates`` over
            which entropy anneals from ``ent0`` to ``ent_T``.
        value_coef: Weight on the value loss in the joint objective.
        max_grad_norm: Gradient clip.
        n_epochs: Number of epochs per update.
        minibatch_size: PPO mini-batch size.
        target_kl: KL early-stop target.
        n_episodes_per_update: Episodes collected before each update.
        max_steps_per_episode: Used to size the rollout buffer.
        total_updates: Number of ``update`` calls intended for the
            entropy anneal schedule.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        device: Optional[torch.device] = None,
        seed: int = 0,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        learning_rate: float = 3e-4,
        adam_eps: float = 1e-5,
        clip_eps: float = 0.2,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        entropy_coef_initial: float = 0.01,
        entropy_coef_final: float = 0.0,
        entropy_anneal_fraction: float = 0.5,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        minibatch_size: int = 32,
        target_kl: float = 0.02,
        n_episodes_per_update: int = 32,
        max_steps_per_episode: int = 2,
        total_updates: int = 1000,
    ) -> None:
        super().__init__(observation_dim, action_dim, device=device, seed=seed)
        # Local generator for action sampling. Constructing a PPO must
        # not mutate global torch/numpy/random state (a global
        # ``set_seed`` here was the root cause of order-sensitive test
        # flakes). Weight init still draws from the ambient global RNG;
        # scripts and tests seed that explicitly before construction.
        self._sample_rng = torch.Generator(device=self.device)
        self._sample_rng.manual_seed(int(seed))

        self.hidden_dim = int(hidden_dim)
        self.n_hidden_layers = int(n_hidden_layers)
        self.clip_eps = float(clip_eps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef_initial = float(entropy_coef_initial)
        self.entropy_coef_final = float(entropy_coef_final)
        self.entropy_anneal_fraction = float(entropy_anneal_fraction)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)
        self.target_kl = float(target_kl)
        self.n_episodes_per_update = int(n_episodes_per_update)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.total_updates = int(total_updates)

        self.policy = ActorCritic(
            observation_dim=self.observation_dim,
            n_actions=self.action_dim,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(learning_rate),
            eps=float(adam_eps),
        )

        # Capacity: exactly n_episodes * max_steps rows, one per env-step
        # per episode row.  K_B items are packed into each row's action
        # field, so the buffer is never over-filled by the K_B factor.
        capacity = max(
            1, self.n_episodes_per_update * self.max_steps_per_episode,
        )
        self._K_B_buf = 1  # updated when rollout sees a K_B env
        self.buffer = RolloutBuffer(
            capacity=capacity,
            observation_dim=self.observation_dim,
            n_actions=self.action_dim,
            K_B=1,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            action_dtype=torch.long,
            store_action_masks=True,
        )

        self._update_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        state: Any,
        *,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample one masked categorical action per batch row.

        Args:
            state: Either a flat ``Tensor (B, obs_dim)`` already on the
                policy device, or an :class:`OrdinalState`. When an
                ``OrdinalState`` is provided this method calls
                ``state.to_tensor(self.device)`` and reads
                ``state.action_mask``.
            deterministic: When ``True`` returns the argmax of the
                masked logits instead of a sample.

        Returns:
            ``(action (B,) long, extras)``. ``extras`` carries
            ``log_prob (B,)``, ``value (B,)``, ``entropy (B,)`` and
            ``action_mask (B, n_actions)`` for the buffer to record.
        """
        obs_t, mask = self._unpack_state(state)
        obs_t = obs_t.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        logits, value = self.policy(obs_t)
        masked = _masked_logits(logits, mask)
        dist = Categorical(logits=masked)
        if deterministic:
            action = torch.argmax(masked, dim=-1)
        else:
            action = self._sample_masked(masked)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.detach(), {
            "log_prob": log_prob.detach(),
            "value": value.detach(),
            "entropy": entropy.detach(),
            "action_mask": (
                torch.ones_like(logits, dtype=torch.bool)
                if mask is None else mask.detach()
            ),
        }

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(self, env: Any, n_episodes: int) -> RolloutStats:
        """Collect ``n_episodes`` episodes into ``self.buffer``.

        One buffer entry is written per env step per episode row. When
        ``K_B > 1`` the K_B item ids and their joint log-prob (sum of
        sequential per-pick log-probs) are packed into that single
        entry. Capacity is therefore exactly
        ``n_episodes * max_steps_per_episode`` regardless of K_B.

        Terminal rewards (e.g. ``r_voi``) are stored in the terminal
        step entry because the buffer is filled at most once per episode
        step per row, not ``K_B`` times.

        Args:
            env: An :class:`~ordrec.envs.OrdinalEnvBase` compatible
                env with ``K_B``, ``horizon_steps``, ``B`` attributes.
                The PPO toy env exposes the same interface.
            n_episodes: Episodes to roll out. For OrdRec one "episode"
                is one student-trajectory of length ``horizon_steps``
                replicated across ``env.B`` rows; the buffer sees
                ``env.B * horizon_steps`` transitions per episode.

        Returns:
            :class:`RolloutStats` summarising the rollout.
        """
        K_B = int(getattr(env, "K_B", 1))
        # Rebuild the buffer when K_B changes (first OrdRec env use) so
        # capacity stays exactly n_episodes * max_steps and action
        # storage is correctly shaped for the vector-action case.
        if K_B != self._K_B_buf:
            capacity = max(
                1, self.n_episodes_per_update * self.max_steps_per_episode,
            )
            self._K_B_buf = K_B
            self.buffer = RolloutBuffer(
                capacity=capacity,
                observation_dim=self.observation_dim,
                n_actions=self.action_dim,
                K_B=K_B,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                action_dtype=torch.long,
                store_action_masks=True,
            )

        self.buffer.reset()
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        component_sums: Dict[str, float] = {
            "r_info": 0.0, "r_cost": 0.0, "r_expo": 0.0, "r_voi": 0.0,
        }
        # One observation count per key. Components can appear at
        # different frequencies in the env's info dict (e.g. r_voi only
        # at the terminal step on some envs), so a shared counter would
        # skew every reported average by the frequency mismatch.
        component_counts: Dict[str, int] = {k: 0 for k in component_sums}

        for _ in range(int(n_episodes)):
            state = env.reset()
            done = False
            episode_start = True
            ep_return = torch.zeros(_state_batch_size(state), dtype=torch.float32)
            step_count = 0

            while not done:
                action, extras = self._policy_step(env, state)
                next_state, reward, done, info = env.step(action)

                # The env may return either (B,) per-row reward or scalar.
                if isinstance(reward, Tensor):
                    reward_per_row = reward.detach().cpu().to(torch.float32)
                else:
                    reward_per_row = torch.full(
                        (_state_batch_size(state),), float(reward), dtype=torch.float32,
                    )
                ep_return = ep_return + reward_per_row
                step_count += 1

                self._insert_transition(
                    state=state, env=env,
                    action_per_row=extras["action_per_row"],
                    log_prob_per_row=extras["log_prob_per_row"],
                    value_per_row=extras["value_per_row"],
                    mask_per_row=extras["mask_per_row"],
                    reward_per_row=reward_per_row,
                    done=bool(done),
                    episode_start=episode_start,
                )

                for key in component_sums.keys():
                    val = info.get(key)
                    if isinstance(val, Tensor):
                        component_sums[key] += float(val.detach().mean().item())
                        component_counts[key] += 1
                state = next_state
                episode_start = False
                if self.buffer.full:
                    break

            for r in ep_return.tolist():
                ep_returns.append(float(r))
                ep_lengths.append(int(step_count))
            if self.buffer.full:
                break

        # Final value bootstrap (always 0 in v1, fixed-horizon).
        self.buffer.compute_advantages(last_value=0.0)

        info: Dict[str, float] = {}
        for key, total in component_sums.items():
            n_obs = component_counts[key]
            info[key] = total / n_obs if n_obs > 0 else 0.0
        return RolloutStats(
            mean_episode_return=(
                sum(ep_returns) / len(ep_returns) if ep_returns else 0.0
            ),
            mean_episode_length=(
                sum(ep_lengths) / len(ep_lengths) if ep_lengths else 0.0
            ),
            n_transitions=self.buffer.size,
            info=info,
        )

    def _policy_step(
        self, env: Any, state: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """One env-level action call. Returns ``(action_for_env, extras)``.

        For OrdRec the env expects ``(B, K_B)`` actions. We run ``K_B``
        independent masked categorical samples per row and stack them.
        For the toy env (no ``K_B``) we run a single sample per row.
        """
        K_B = int(getattr(env, "K_B", 1))
        if K_B == 1:
            action, extras = self.act(state, deterministic=False)
            action_per_row = action.detach().cpu().to(torch.long).reshape(-1)
            log_prob_per_row = extras["log_prob"].detach().cpu().to(torch.float32).reshape(-1)
            value_per_row = extras["value"].detach().cpu().to(torch.float32).reshape(-1)
            mask_per_row = extras["action_mask"].detach().cpu().to(torch.bool)
            return action, {
                "action_per_row": action_per_row,
                "log_prob_per_row": log_prob_per_row,
                "value_per_row": value_per_row,
                "mask_per_row": mask_per_row,
            }
        # OrdRec batched env. Sample K_B without replacement under the
        # current mask. Sequential sampling so the mask after each pick
        # forbids that item, matching the env's no-repeat invariant.
        obs_t, mask = self._unpack_state(state)
        obs_t = obs_t.to(self.device)
        cur_mask = (mask if mask is not None else None)
        if cur_mask is not None:
            cur_mask = cur_mask.to(self.device).clone()
        B = obs_t.shape[0]
        action_cols: list[Tensor] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        masks: list[Tensor] = []
        with torch.no_grad():
            logits, value = self.policy(obs_t)
            for _ in range(K_B):
                masked = _masked_logits(logits, cur_mask)
                dist = Categorical(logits=masked)
                a = self._sample_masked(masked)
                lp = dist.log_prob(a)
                action_cols.append(a)
                log_probs.append(lp)
                values.append(value)
                masks.append(
                    cur_mask if cur_mask is not None
                    else torch.ones_like(logits, dtype=torch.bool)
                )
                if cur_mask is not None:
                    cur_mask = cur_mask.clone()
                    cur_mask.scatter_(
                        dim=1, index=a.unsqueeze(1),
                        src=torch.zeros(B, 1, dtype=torch.bool, device=cur_mask.device),
                    )
        action = torch.stack(action_cols, dim=1).to(dtype=torch.long)
        # Per-row, per-sub-step buffer entries. Each sub-step sees the
        # same observation; the value is identical across sub-steps.
        # We log B * K_B transitions per env.step, one per (row, k).
        action_per_row = action.reshape(-1).detach().cpu()
        lp = torch.stack(log_probs, dim=1).reshape(-1).detach().cpu().to(torch.float32)
        v = torch.stack(values, dim=1).reshape(-1).detach().cpu().to(torch.float32)
        masks_stack = torch.stack(masks, dim=1)
        masks_stack = masks_stack.reshape(-1, masks_stack.shape[-1])
        return action, {
            "action_per_row": action_per_row,
            "log_prob_per_row": lp,
            "value_per_row": v,
            "mask_per_row": masks_stack.detach().cpu().to(torch.bool),
        }

    def _insert_transition(
        self,
        *,
        state: Any,
        env: Any,
        action_per_row: Tensor,
        log_prob_per_row: Tensor,
        value_per_row: Tensor,
        mask_per_row: Tensor,
        reward_per_row: Tensor,
        done: bool,
        episode_start: bool,
    ) -> None:
        """Insert one buffer entry per episode row for one env step.

        Credit assignment. One entry is written per row. The entry's
        reward is the full env-step reward for that row (not a
        per-sub-step slice). When K_B > 1 the action field holds all
        K_B item ids as a 1D tensor and log_prob holds the joint
        log-prob (sum of sequential per-pick log-probs). This is the
        formulation that keeps the PPO importance ratio correct:

            ratio = exp(joint_lp_new - joint_lp_old)

        where joint_lp_new is re-evaluated during the PPO update by
        summing the log-probs of the K_B picks under the current policy
        with the same sequential no-repeat mask.

        Args:
            state: OrdinalState or Tensor observation at the start of
                this env step. Used to read the per-row obs.
            env: The driving env (provides K_B via ``getattr``).
            action_per_row: ``LongTensor (B * K_B,)`` flattened actions,
                row-major order (b=0 k=0, b=0 k=1, ..., b=B-1 k=K_B-1).
            log_prob_per_row: ``FloatTensor (B * K_B,)`` per-pick
                log-probs in the same order.
            value_per_row: ``FloatTensor (B * K_B,)`` critic values
                (identical across sub-steps for the same row).
            mask_per_row: ``BoolTensor (B * K_B, n_actions)`` per-pick
                action masks (the mask narrows after each pick).
            reward_per_row: ``FloatTensor (B,)`` per-row env-step reward.
            done: Terminated flag from the env for this step.
            episode_start: True at the first step of the episode.
        """
        obs_t, _ = self._unpack_state(state)
        obs_cpu = obs_t.detach().cpu().to(torch.float32)
        B = obs_cpu.shape[0]
        K_B = int(getattr(env, "K_B", 1))
        N = action_per_row.shape[0]
        if N != B * K_B:
            raise ValueError(
                f"action_per_row size {N} does not match B * K_B = {B} * {K_B}."
            )

        for b in range(B):
            if K_B == 1:
                # Single-item env: scalar action, scalar log_prob.
                act = action_per_row[b]
                joint_lp = float(log_prob_per_row[b].item())
                # Mask for slot b (first-and-only sub-step of row b).
                mask_b = mask_per_row[b]
            else:
                # Multi-item block: pack all K_B ids and sum log_probs.
                start = b * K_B
                act = action_per_row[start : start + K_B]   # (K_B,)
                joint_lp = float(
                    log_prob_per_row[start : start + K_B].sum().item()
                )
                # Use the first sub-step's mask as the stored mask so
                # the PPO re-eval starts from the same pre-step mask.
                mask_b = mask_per_row[start]

            self.buffer.insert(
                state=obs_cpu[b],
                action=act,
                reward=float(reward_per_row[b].item()),
                log_prob=joint_lp,
                value=float(value_per_row[b * K_B].item()),
                done=bool(done),
                action_mask=mask_b,
                episode_start=(bool(episode_start) and b == 0),
            )
            if self.buffer.full:
                return

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, buffer: Any = None) -> UpdateStats:
        """Run ``n_epochs`` passes of clipped-surrogate PPO over the buffer."""
        buf = buffer if buffer is not None else self.buffer
        if buf.size == 0:
            return UpdateStats(
                policy_loss=0.0, value_loss=0.0, entropy=0.0,
                approx_kl=0.0, clipfrac=0.0, n_grad_steps=0,
                extras={"early_stop": False, "skipped": True},
            )
        ent_coef = linear_anneal(
            self._update_count,
            anneal_steps=max(
                1, int(self.entropy_anneal_fraction * self.total_updates),
            ),
            start=self.entropy_coef_initial,
            end=self.entropy_coef_final,
        )

        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clipfrac": 0.0}
        n_steps = 0
        early_stop = False
        K_B_buf = int(getattr(buf, "K_B", 1))
        for epoch in range(self.n_epochs):
            for batch in buf.iter_minibatches(
                minibatch_size=self.minibatch_size,
                shuffle=True,
                normalize_advantages=True,
            ):
                batch = batch.to(self.device)
                logits, values = self.policy(batch.states)
                masked = _masked_logits(logits, batch.action_masks)

                if K_B_buf <= 1:
                    # Scalar action case: standard categorical log_prob.
                    dist = Categorical(logits=masked)
                    new_log_probs = dist.log_prob(batch.actions)
                    entropy = dist.entropy().mean()
                else:
                    # Vector action case (K_B > 1). Re-evaluate the joint
                    # log-prob by replaying the sequential sampling with
                    # a tightening no-repeat mask, matching the sampling
                    # procedure in _policy_step. Entropy is the mean of
                    # the per-pick categorical entropies over the K_B
                    # sub-steps (the stored mask is the pre-step mask).
                    B_mb = batch.states.shape[0]
                    joint_lp = torch.zeros(B_mb, device=self.device)
                    ent_sum = torch.zeros(B_mb, device=self.device)
                    cur_mask = (
                        batch.action_masks.clone()
                        if batch.action_masks is not None
                        else None
                    )
                    for k in range(K_B_buf):
                        cur_masked = _masked_logits(logits, cur_mask)
                        dist_k = Categorical(logits=cur_masked)
                        a_k = batch.actions[:, k]
                        joint_lp = joint_lp + dist_k.log_prob(a_k)
                        ent_sum = ent_sum + dist_k.entropy()
                        if cur_mask is not None:
                            cur_mask = cur_mask.clone()
                            cur_mask.scatter_(
                                dim=1, index=a_k.unsqueeze(1),
                                src=torch.zeros(B_mb, 1, dtype=torch.bool, device=self.device),
                            )
                    new_log_probs = joint_lp
                    entropy = (ent_sum / K_B_buf).mean()

                # Scalar action case entropy (only used when K_B==1 path falls
                # through to here via the dist variable): already set above.
                if K_B_buf <= 1:
                    pass  # entropy already set in the scalar branch

                ratio = (new_log_probs - batch.old_log_probs).exp()
                unclipped = ratio * batch.advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps,
                ) * batch.advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value clipping (CleanRL style).
                v_clipped = batch.old_values + torch.clamp(
                    values - batch.old_values, -self.clip_eps, self.clip_eps,
                )
                v_losses = torch.max(
                    (values - batch.returns).pow(2),
                    (v_clipped - batch.returns).pow(2),
                )
                value_loss = 0.5 * v_losses.mean()

                loss = (
                    policy_loss + self.value_coef * value_loss
                    - ent_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch.old_log_probs - new_log_probs).mean()
                    clipfrac = (
                        (ratio - 1.0).abs().gt(self.clip_eps).float().mean()
                    )

                sums["policy_loss"] += float(policy_loss.item())
                sums["value_loss"] += float(value_loss.item())
                sums["entropy"] += float(entropy.item())
                sums["approx_kl"] += float(approx_kl.item())
                sums["clipfrac"] += float(clipfrac.item())
                n_steps += 1

                if approx_kl.item() > 1.5 * self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        denom = max(1, n_steps)
        self._update_count += 1
        return UpdateStats(
            policy_loss=sums["policy_loss"] / denom,
            value_loss=sums["value_loss"] / denom,
            entropy=sums["entropy"] / denom,
            approx_kl=sums["approx_kl"] / denom,
            clipfrac=sums["clipfrac"] / denom,
            n_grad_steps=n_steps,
            extras={
                "entropy_coef": float(ent_coef),
                "early_stop": bool(early_stop),
                "update_count": int(self._update_count),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": type(self).__name__,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "n_hidden_layers": self.n_hidden_layers,
                "update_count": self._update_count,
                "seed": self.seed,
            },
            p,
        )
        return p

    def load(self, path: Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No PPO checkpoint at {p}")
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._update_count = int(ckpt.get("update_count", 0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_masked(self, masked_logits: Tensor) -> Tensor:
        """Sample one action per row using the local generator.

        Equivalent in distribution to ``Categorical(logits=masked).sample()``
        but draws from ``self._sample_rng`` instead of global torch RNG,
        so PPO's action sampling is reproducible given ``seed`` and
        independent of ambient global state.

        Args:
            masked_logits: ``Tensor (B, n_actions)``; disallowed actions
                are ``-inf``. Rows must have at least one finite entry.

        Returns:
            ``LongTensor (B,)`` sampled actions.
        """
        probs = torch.softmax(masked_logits, dim=-1)
        return torch.multinomial(
            probs, num_samples=1, generator=self._sample_rng,
        ).squeeze(-1)

    def _unpack_state(self, state: Any) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(state, OrdinalState):
            obs = state.to_tensor(self.device)
            return obs, state.action_mask
        if isinstance(state, Mapping):
            obs = state["obs"]
            mask = state.get("action_mask")
            if not isinstance(obs, Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            return obs.to(self.device), mask
        if isinstance(state, Tensor):
            obs = state if state.ndim == 2 else state.unsqueeze(0)
            return obs.to(self.device), None
        raise TypeError(
            f"Unsupported state type {type(state).__name__}. "
            "Expected OrdinalState, mapping or Tensor."
        )


def _state_batch_size(state: Any) -> int:
    if isinstance(state, OrdinalState):
        return state.theta_t.shape[0]
    if isinstance(state, Mapping):
        obs = state.get("obs")
        if isinstance(obs, Tensor):
            return obs.shape[0] if obs.ndim == 2 else 1
        return 1
    if isinstance(state, Tensor):
        return state.shape[0] if state.ndim == 2 else 1
    return 1


# ---------------------------------------------------------------------------
# Future-fillable sketches (impl guide Sections 4.5 and 4.6)
# ---------------------------------------------------------------------------
# DQN, file at training/dqn.py. Mnih et al. 2015 Nature with Double DQN
# (van Hasselt et al. 2016 AAAI). Off-policy using ReplayBuffer in
# training/replay.py. Target network target_q_net is a frozen lagged
# copy, hard update every target_update_interval=1000 steps or soft
# polyak at tau=0.005. Epsilon-greedy 1.0 -> 0.05 linearly over the
# first 10% of training. Double-DQN argmax with Q_online, value with
# Q_target. SmoothL1 loss. Adam lr 1e-4, grad clip 0.5. Action masking
# via masked_fill on the Q head.
#
# SAC-discrete, file at training/sac.py. Christodoulou 2019. Twin Qs
# Q1, Q2 with polyak targets at tau=0.005. Stochastic categorical
# policy. Auto-tuned temperature with target entropy 0.98 * log(|A|).
# Critic target uses sum_a pi(a|s') * (min(Q1_t, Q2_t)(s', a) -
# alpha log pi(a|s')). Actor minimises sum_a pi(a|s) * (alpha
# log pi(a|s) - min(Q1, Q2)(s, a)). Same ReplayBuffer as DQN.


__all__ = ["ActorCritic", "PPO"]
