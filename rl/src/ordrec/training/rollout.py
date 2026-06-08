"""``RolloutBuffer`` for on-policy PPO.

Pre-allocates ``(capacity, ...)`` storage so the rollout hot loop does
not pay Python-side append cost. Capacity equals
``n_episodes_per_update * max_steps_per_episode``. For OrdRec with
``T / K_B = 2`` and 32 episodes per update this is 64.

The buffer stores per-step state, action, reward, log-prob, value,
done and per-step action mask. ``compute_advantages`` runs GAE
segment-by-segment over episode boundaries and ``iter_minibatches``
shuffles and batches the flat storage for SGD.

See ``docs/ordrec_impl_guide.md`` Section 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from .gae import compute_gae


@dataclass
class RolloutBatch:
    """One PPO mini-batch view over the flat buffer storage."""

    states: Tensor          # (B, obs_dim)
    actions: Tensor         # (B,) long, or (B, K) if vector action
    old_log_probs: Tensor   # (B,)
    advantages: Tensor      # (B,)
    returns: Tensor         # (B,)
    old_values: Tensor      # (B,)
    action_masks: Optional[Tensor] = None  # (B, n_actions) bool

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            old_log_probs=self.old_log_probs.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            old_values=self.old_values.to(device),
            action_masks=(
                None if self.action_masks is None
                else self.action_masks.to(device)
            ),
        )


class RolloutBuffer:
    """Pre-allocated on-policy buffer with GAE.

    Args:
        capacity: Maximum number of transitions stored.
        observation_dim: ``obs_dim``. The state tensor written by
            ``insert`` must have this last-dim length.
        n_actions: ``Q + 1``. Used to size the action-mask buffer.
        device: Device on which the storage tensors live. Defaults to
            CPU so big buffers do not hog GPU memory.
        gamma: GAE discount, default ``0.95``.
        gae_lambda: GAE lambda, default ``0.95``.
        action_dtype: Dtype of the action storage. Defaults to long.
            Pass ``torch.long`` for discrete actions.
        store_action_masks: When ``True`` the buffer reserves
            ``(capacity, n_actions)`` boolean mask storage. Set to
            ``False`` for tests on envs without action masks.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        n_actions: int,
        *,
        device: Optional[torch.device] = None,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        action_dtype: torch.dtype = torch.long,
        store_action_masks: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}.")
        if observation_dim <= 0:
            raise ValueError(
                f"observation_dim must be > 0, got {observation_dim}."
            )
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.n_actions = int(n_actions)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.store_action_masks = bool(store_action_masks)

        self._dtype = torch.float32
        self.states = torch.zeros(
            self.capacity, self.observation_dim, dtype=self._dtype, device=self.device,
        )
        self.actions = torch.zeros(self.capacity, dtype=action_dtype, device=self.device)
        self.rewards = torch.zeros(self.capacity, dtype=self._dtype, device=self.device)
        self.log_probs = torch.zeros(self.capacity, dtype=self._dtype, device=self.device)
        self.values = torch.zeros(self.capacity, dtype=self._dtype, device=self.device)
        self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.episode_starts = torch.zeros(
            self.capacity, dtype=torch.bool, device=self.device,
        )
        if self.store_action_masks:
            self.action_masks = torch.ones(
                self.capacity, self.n_actions,
                dtype=torch.bool, device=self.device,
            )
        else:
            self.action_masks = None

        self.advantages = torch.zeros(
            self.capacity, dtype=self._dtype, device=self.device,
        )
        self.returns = torch.zeros(
            self.capacity, dtype=self._dtype, device=self.device,
        )

        self._size = 0
        self._advantages_computed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the buffer without reallocating storage."""
        self._size = 0
        self._advantages_computed = False

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size >= self.capacity

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        *,
        action_mask: Optional[Tensor] = None,
        episode_start: bool = False,
    ) -> None:
        """Append a single transition.

        Args:
            state: ``Tensor (obs_dim,)`` or ``(1, obs_dim)`` flattened
                policy observation.
            action: scalar long-valued Tensor for the discrete action.
            reward: Scalar reward.
            log_prob: ``log pi(a | s)`` evaluated at sampling time.
            value: ``V(s)`` evaluated at sampling time.
            done: Terminal flag for this transition.
            action_mask: Optional ``BoolTensor (n_actions,)`` recorded
                so the PPO update can re-evaluate the policy under the
                same mask. Required when ``store_action_masks=True``.
            episode_start: ``True`` at the first transition of a new
                episode. Used by ``compute_advantages`` to split GAE
                segments.
        """
        if self._size >= self.capacity:
            raise IndexError(
                f"RolloutBuffer is full ({self._size}/{self.capacity})"
            )
        idx = self._size
        s_flat = state.reshape(-1).to(device=self.device, dtype=self._dtype)
        if s_flat.numel() != self.observation_dim:
            raise ValueError(
                f"state has {s_flat.numel()} elements, expected "
                f"{self.observation_dim}."
            )
        self.states[idx] = s_flat
        self.actions[idx] = action.to(device=self.device, dtype=self.actions.dtype)
        self.rewards[idx] = float(reward)
        self.log_probs[idx] = float(log_prob)
        self.values[idx] = float(value)
        self.dones[idx] = bool(done)
        self.episode_starts[idx] = bool(episode_start)
        if self.action_masks is not None:
            if action_mask is None:
                raise ValueError(
                    "action_mask is required when store_action_masks=True."
                )
            m = action_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
            if m.numel() != self.n_actions:
                raise ValueError(
                    f"action_mask has {m.numel()} elements, expected "
                    f"{self.n_actions}."
                )
            self.action_masks[idx] = m
        self._size += 1
        self._advantages_computed = False

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: float = 0.0) -> None:
        """Run GAE per-episode and fill ``advantages`` and ``returns``.

        Episode boundaries are detected from ``episode_starts``. Each
        segment between two starts (or between a start and the buffer
        end) is treated as an independent trajectory. ``last_value`` is
        used at the buffer-end boundary of an incomplete episode; if
        the trajectory terminated cleanly with ``done=True`` the
        bootstrap is forced to zero regardless.
        """
        if self._size == 0:
            return
        segments = self._episode_segments()
        for start, stop in segments:
            seg_rewards = self.rewards[start:stop]
            seg_values = self.values[start:stop]
            seg_dones = self.dones[start:stop]
            # Bootstrap value at the segment end:
            #   - if the last step is done, zero
            #   - if this is the buffer-tail incomplete segment, last_value
            #   - otherwise zero (segment ended cleanly per episode_starts)
            if bool(seg_dones[-1].item()):
                bootstrap = 0.0
            elif stop == self._size:
                bootstrap = float(last_value)
            else:
                bootstrap = 0.0
            adv, ret = compute_gae(
                seg_rewards, seg_values, seg_dones.to(seg_rewards.dtype),
                gamma=self.gamma, lambda_=self.gae_lambda,
                last_value=bootstrap,
            )
            self.advantages[start:stop] = adv
            self.returns[start:stop] = ret
        self._advantages_computed = True

    def _episode_segments(self) -> List[Tuple[int, int]]:
        """Return ``[(start, stop), ...]`` intervals spanning ``self._size``."""
        if self._size == 0:
            return []
        starts: List[int] = []
        for i in range(self._size):
            if i == 0 or bool(self.episode_starts[i].item()):
                starts.append(i)
        starts.append(self._size)
        return [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def iter_minibatches(
        self,
        minibatch_size: int,
        *,
        shuffle: bool = True,
        normalize_advantages: bool = True,
    ) -> Iterator[RolloutBatch]:
        """Yield mini-batches covering every transition exactly once.

        Args:
            minibatch_size: ``B`` for each yielded mini-batch.
            shuffle: When ``True`` indices are permuted before slicing
                so PPO sees a fresh order each epoch.
            normalize_advantages: When ``True`` advantages are
                normalised within the buffer (zero mean, unit std) once
                before iteration. Skipped when only one transition is
                stored to avoid a divide-by-zero. See impl-guide test
                ``advantage_normalization_off_when_n_eq_1``.

        Yields:
            :class:`RolloutBatch`. The number of yielded batches is
            ``ceil(size / minibatch_size)``.
        """
        if not self._advantages_computed:
            raise RuntimeError(
                "compute_advantages must be called before iter_minibatches."
            )
        if minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be > 0, got {minibatch_size}.")

        N = self._size
        adv = self.advantages[:N].clone()
        if normalize_advantages and N > 1:
            std = adv.std()
            if std.item() > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)

        if shuffle:
            perm = torch.randperm(N, device=self.device)
        else:
            perm = torch.arange(N, device=self.device)

        for start in range(0, N, minibatch_size):
            stop = min(start + minibatch_size, N)
            idx = perm[start:stop]
            yield RolloutBatch(
                states=self.states[idx],
                actions=self.actions[idx],
                old_log_probs=self.log_probs[idx],
                advantages=adv[idx],
                returns=self.returns[idx],
                old_values=self.values[idx],
                action_masks=(
                    None if self.action_masks is None
                    else self.action_masks[idx]
                ),
            )


__all__ = ["RolloutBuffer", "RolloutBatch"]
