"""``OrdRecEnv``, the concrete Gym-style env around a frozen MAGPCM.

Wraps a frozen MA-IRT world model, an item-cache (``alpha_table``,
``beta_table``), the OrdinalRewardCompute composer and a data
adapter into a single environment that the OrdRec policy can drive.

State flow per episode.

1. ``reset`` samples a batch of students from the data adapter,
   captures their first-``S_init`` items as a warmup history,
   forwards the frozen world model to read ``theta_0``, samples the
   per-episode probe sets ``C`` and ``H_probe`` plus the held-out
   probe responses, and emits the initial :class:`OrdinalState`.
2. Each ``step(action)`` extends the history by ``K_B`` items per
   row. The corresponding ordinal responses are simulated by
   sampling from the frozen world model's predicted distribution at
   the appended positions, which keeps the reward shaping consistent
   with the model's own beliefs. The composer then runs.
3. After ``T // K_B`` steps the episode terminates. The env updates
   its fleet exposure EMA from per-session administration counts.

See ``docs/ordrec_impl_guide.md`` Sections 3.3 and 5.1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..data.base import OrdinalDatasetBase
from ..reward.config import RewardConfig
from ..reward.exposure import update_fleet_exposure
from ..reward.nll_anchor import resample_probe_responses
from ..reward.ordinal_reward import OrdinalRewardCompute
from .action_mask import (
    build_admin_mask,
    build_probe_mask,
    compose_action_mask,
    update_no_repeat_mask,
)
from .base import OrdinalEnvBase, OrdinalState
from .frozen_magpcm import FrozenMAGPCM
from .item_cache import ItemCache
from .probe_sampler import ProbeSampler, make_probe_sampler


# Maximum history length fed to the frozen encoder on every forward.
# Histories longer than this are right-truncated (most recent tokens
# kept) before the frozen forward to prevent unbounded sequence growth.
# 200 tokens covers warmup_len (5) + T (10) with a large safety margin.
_MAX_HISTORY_LEN: int = 200


class OrdRecEnv(OrdinalEnvBase):
    """Gym-style env over a frozen MA-IRT world model.

    The env is deliberately batched. ``B`` is the number of
    concurrent students driven through each ``step``. PPO collects
    one transition per row per step so a single env instance supports
    on-policy parallel rollout without the overhead of per-student
    Python loops.

    Args:
        world_model: Frozen wrapper around an MA-IRT model.
        adapter: Data adapter providing sequences to sample from at
            reset. The adapter must expose ``_questions``,
            ``_responses`` and ``get_split``; the standard E1/E2
            adapters satisfy this.
        item_cache: Per-item ``(alpha, beta)`` lookup tables.
        reward_fn: :class:`OrdinalRewardCompute` instance. The env
            owns it so test fixtures can swap the reward without
            touching the policy.
        cfg: :class:`RewardConfig` mirroring ``reward_fn.cfg``;
            stored separately so the env can read ``T`` and
            ``K_B`` without reaching through the reward.
        batch_size: ``B``. Defaults to 8.
        warmup_len: Number of initial items used to prime the
            encoder before the first reward computation. Drawn from
            the student's real history.
        split: Adapter split to draw students from. Defaults to
            ``"train"``.
        device: Device on which the world-model forward runs.
        seed: RNG seed for student sampling and probe draws.
        probe_sampler: Optional pre-built :class:`~ordrec.envs.probe_sampler.ProbeSampler`.
            When ``None`` the sampler is built from
            ``cfg.probe_sampler`` (default ``"stratified"``) using the
            item cache's beta table. Pass an explicit instance in tests
            to avoid the beta-table dependency.
    """

    def __init__(
        self,
        world_model: FrozenMAGPCM,
        adapter: OrdinalDatasetBase,
        item_cache: ItemCache,
        reward_fn: OrdinalRewardCompute,
        cfg: RewardConfig,
        *,
        batch_size: int = 8,
        warmup_len: int = 5,
        split: str = "train",
        device: Optional[torch.device] = None,
        seed: int = 0,
        probe_sampler: Optional[ProbeSampler] = None,
    ) -> None:
        super().__init__()
        self.world = world_model
        self.adapter = adapter
        self.item_cache = item_cache
        self.reward_fn = reward_fn
        self.cfg = cfg

        self.B = int(batch_size)
        self.warmup_len = int(warmup_len)
        self.split = split
        self.device = torch.device(device) if device is not None else world_model.device
        self.seed = int(seed)

        self.n_questions = int(item_cache.n_questions)
        self.n_categories = int(item_cache.n_categories)
        self.n_traits = int(item_cache.n_traits)
        self.K_B = int(cfg.K_B)
        self.T = int(cfg.T)
        self.horizon_steps = self.T // self.K_B

        # Tensor copies of the cache tables live on the env device so
        # the reward composer does not reshuffle them every call.
        self.alpha_table = item_cache.alpha_tensor(self.device)
        self.beta_table = item_cache.beta_tensor(self.device)

        # Buffers
        self.admin_mask = build_admin_mask(self.n_questions, device=self.device)
        self.fleet_expo = torch.zeros(
            self.n_questions + 1, dtype=torch.float32, device=self.device
        )

        # Per-episode mutable state filled by ``reset``.
        self._history_q: Optional[Tensor] = None
        self._history_r: Optional[Tensor] = None
        self._exposure_counts: Optional[Tensor] = None
        self._no_repeat: Optional[Tensor] = None
        self._probe_mask: Optional[Tensor] = None
        self._probe_C: Optional[Tensor] = None
        self._probe_H: Optional[Tensor] = None
        self._probe_H_resp: Optional[Tensor] = None
        self._theta_prev: Optional[Tensor] = None
        self._theta_0: Optional[Tensor] = None
        # Terminal probe responses re-sampled from the world model at the
        # end of the episode when cfg.resample_probe_at_terminal is True.
        # None between reset() and the terminal step.
        self._probe_H_resp_terminal: Optional[Tensor] = None
        self._episode_step: int = 0
        self._terminated: bool = False
        self._episode_idx: int = 0
        self._rng: torch.Generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

        # Probe sampler: stratified by default (uses beta_table for item
        # difficulty ranking); callers can override for tests.
        if probe_sampler is not None:
            self._probe_sampler: ProbeSampler = probe_sampler
        else:
            sampler_mode = getattr(cfg, "probe_sampler", "stratified")
            self._probe_sampler = make_probe_sampler(
                sampler_mode,
                beta_table=item_cache.beta_table,
                n_strata=int(cfg.n_difficulty_strata),
            )

    # ------------------------------------------------------------------
    # Static shape accessors
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self.n_traits + max(1, self.horizon_steps)

    @property
    def action_dim(self) -> int:
        return self.n_questions + 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> OrdinalState:
        """Start a new episode and return the initial observation."""
        if seed is not None:
            self.seed = int(seed)
            self._rng = torch.Generator(device="cpu").manual_seed(self.seed)
            self._np_rng = np.random.default_rng(self.seed)

        self._episode_idx += 1

        # 1. Sample B students from the adapter split.
        idx = self.adapter.get_split(self.split)
        if idx.size == 0:
            raise ValueError(
                f"Adapter has no rows in split {self.split!r}; cannot reset env."
            )
        sampled_rows = self._np_rng.choice(
            idx, size=self.B, replace=(idx.size < self.B)
        )

        # 2. Build the warmup history. Each student is required to have
        # at least ``warmup_len + horizon`` real items so that the probes
        # have a held-out tail; if a student is short we wrap by indexing
        # modulo the available length so the env never crashes during
        # tests.
        warmup_q = torch.zeros(
            self.B, self.warmup_len, dtype=torch.long, device=self.device,
        )
        warmup_r = torch.zeros(
            self.B, self.warmup_len, dtype=torch.long, device=self.device,
        )
        # Track each student's "next probe pool", real items not in the
        # warmup, used to populate H_probe with real responses.
        tail_q: List[List[int]] = []
        tail_r: List[List[int]] = []
        for b, row in enumerate(sampled_rows):
            qs = self.adapter._questions[int(row)]
            rs = self.adapter._responses[int(row)]
            L = len(qs)
            if L < 1:
                raise ValueError(
                    f"Adapter row {int(row)} is empty."
                )
            for s in range(self.warmup_len):
                warmup_q[b, s] = int(qs[s % L])
                warmup_r[b, s] = int(rs[s % L])
            # Use anything past the warmup as the held-out tail.
            tail_start = self.warmup_len % L
            tail_q.append([int(qs[(tail_start + j) % L]) for j in range(L)])
            tail_r.append([int(rs[(tail_start + j) % L]) for j in range(L)])

        # 3. Probe sampling. Stratification-by-difficulty is the
        # production recipe; the contract-level test env uses a
        # uniform sampler that respects the disjointness invariant.
        probe_C, probe_H, probe_H_resp = self._sample_probes(
            warmup_q=warmup_q, tail_q=tail_q, tail_r=tail_r,
        )

        # 4. World-model warmup forward to read theta_0.
        out = self.world.forward_no_grad(
            self._truncate_history(warmup_q),
            self._truncate_history(warmup_r),
        )
        theta_0 = out["theta"][:, -1, :].detach()

        # 5. Mask initialisation.
        probe_mask = build_probe_mask(
            probe_C, probe_H, self.n_questions,
        ).to(device=self.device)
        no_repeat = torch.ones(
            self.B, self.n_questions + 1, dtype=torch.bool, device=self.device,
        )
        # Mark warmup items as already administered.
        update_no_repeat_mask(no_repeat, warmup_q)
        exposure_counts = torch.zeros(
            self.B, self.n_questions + 1, dtype=torch.long, device=self.device,
        )
        for b in range(self.B):
            exposure_counts[b].scatter_add_(
                dim=0,
                index=warmup_q[b].clamp(min=0),
                src=torch.ones_like(warmup_q[b]),
            )

        # 6. Pack the episode state.
        self._history_q = warmup_q
        self._history_r = warmup_r
        self._exposure_counts = exposure_counts
        self._no_repeat = no_repeat
        self._probe_mask = probe_mask
        self._probe_C = probe_C
        self._probe_H = probe_H
        self._probe_H_resp = probe_H_resp
        self._theta_0 = theta_0
        self._theta_prev = theta_0
        self._probe_H_resp_terminal = None
        self._episode_step = 0
        self._terminated = False

        action_mask = compose_action_mask(
            self.admin_mask, self._probe_mask, self._no_repeat,
        )

        return self._build_state(
            theta_t=theta_0,
            action_mask=action_mask,
            episode_step=0,
            terminated=False,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action: Tensor
    ) -> Tuple[OrdinalState, Tensor, bool, Dict[str, Any]]:
        """Advance the env by one batch of ``K_B`` items."""
        if self._terminated:
            raise RuntimeError(
                "step called on a terminated env; call reset() first."
            )
        if self._history_q is None:
            raise RuntimeError("step called before reset.")

        if action.shape != (self.B, self.K_B):
            raise ValueError(
                f"action must be ({self.B}, {self.K_B}), got "
                f"{tuple(action.shape)}."
            )
        action = action.to(device=self.device, dtype=torch.long)

        # 1. Enforce mask. Any action selecting a masked id is an error;
        # the policy is responsible for sampling only over allowed ids.
        current_mask = compose_action_mask(
            self.admin_mask, self._probe_mask, self._no_repeat,
        )
        # ``gather`` indexes the (B, Q+1) mask with the (B, K_B) action;
        # if any entry is False the policy violated the mask.
        gathered = current_mask.gather(dim=1, index=action.clamp(min=0))
        if not bool(gathered.all().item()):
            raise ValueError(
                "step received a masked action; policy must respect mask."
            )

        # 2. Append the action to the history and simulate responses.
        #
        # Sampling semantics (joint, one forward per step):
        # We do a single forward pass with the new K_B items and
        # placeholder zero responses, sample all K_B responses jointly
        # from the predicted distribution at those positions, then store
        # the sampled responses in the history. This is the "joint from
        # model's conditional" approach documented in Section B7 of the
        # maintainability review. It ignores intra-block auto-regression
        # (i.e., response r_{t+1} is sampled independently of r_t within
        # the block) but correctly captures inter-block auto-regression
        # (each step sees the sampled responses from all prior steps).
        # Over T/K_B=2 steps this is a good approximation.
        #
        # History truncation (B8): cap to the most recent
        # _MAX_HISTORY_LEN items before every frozen forward to prevent
        # unbounded sequence growth that would slow down the encoder.
        prev_q = self._truncate_history(self._history_q)
        prev_r = self._truncate_history(self._history_r)
        new_q = torch.cat([prev_q, action], dim=1)
        placeholder_r = torch.cat(
            [prev_r, torch.zeros_like(action)], dim=1,
        )
        with torch.no_grad():
            out = self.world.forward_no_grad(new_q, placeholder_r)

        # Sample responses at the K_B newly appended positions from the
        # single forward's predicted probabilities.
        new_probs = out["probs"][:, -self.K_B:, :]  # (B, K_B, K)
        flat = new_probs.reshape(-1, new_probs.shape[-1])
        sampled = torch.multinomial(
            flat.clamp_min(1e-12), num_samples=1, generator=None,
        ).view(self.B, self.K_B).to(dtype=torch.long, device=self.device)
        new_r = torch.cat([prev_r, sampled], dim=1)

        # theta_t is read from the single forward. Because responses at
        # the K_B new positions were zero (placeholder) during this
        # forward, theta_t reflects the history WITHOUT the new responses
        # for this step. The next step's forward will use the sampled
        # responses, giving the correct inter-block autoregression.
        theta_t = out["theta"][:, -1, :].detach()

        # 3. Update buffers. Store the full extended history (with sampled
        # responses) so that subsequent steps can forward with the correct
        # response context. Truncation happens at forward time, not here.
        self._history_q = torch.cat([self._history_q, action], dim=1)
        self._history_r = torch.cat([self._history_r, sampled], dim=1)
        update_no_repeat_mask(self._no_repeat, action)
        for b in range(self.B):
            self._exposure_counts[b].scatter_add_(
                dim=0,
                index=action[b].clamp(min=0),
                src=torch.ones_like(action[b]),
            )

        self._episode_step += 1
        terminated = self._episode_step >= self.horizon_steps

        # 4a. Dynamic-world probe resampling at the terminal step.
        #
        # When cfg.resample_probe_at_terminal is True, re-draw the
        # H_probe responses by running one more frozen forward conditioned
        # on the full simulated history (including this step's responses).
        # This makes the terminal NLL anchor compare theta_0 vs theta_T
        # on what the world model predicts the student should answer NOW,
        # not on stale historical responses captured at reset.
        #
        # The probe mask keeps H_probe ids out of the action set throughout
        # the episode, so probing them here does not constitute
        # administration (anti-gaming invariant preserved).
        if terminated and getattr(self.cfg, "resample_probe_at_terminal", False):
            self._probe_H_resp_terminal = resample_probe_responses(
                self.world,
                self._truncate_history(self._history_q),
                self._truncate_history(self._history_r),
                self._probe_H,
                generator=self._rng,
            )

        # 4. Reward.
        info = self._build_info(step_index=self._episode_step)
        state_prev = {"theta": self._theta_prev}
        state_next = {"theta": theta_t}
        reward, breakdown = self.reward_fn(
            state_prev, action, state_next, info,
        )
        self._theta_prev = theta_t

        # 5. Update the per-episode fleet exposure when the episode ends.
        if terminated:
            session_admin = (self._exposure_counts > 0).to(
                dtype=torch.float32
            ).mean(dim=0)  # (Q+1,) fraction of sessions
            update_fleet_exposure(
                self.fleet_expo,
                session_admin,
                decay=self.cfg.expo_ema_decay,
            )
            self._terminated = True

        action_mask = compose_action_mask(
            self.admin_mask, self._probe_mask, self._no_repeat,
        )
        next_state = self._build_state(
            theta_t=theta_t,
            action_mask=action_mask,
            episode_step=self._episode_step,
            terminated=terminated,
        )
        info_out = dict(breakdown)
        info_out["step_index"] = self._episode_step
        info_out["episode_idx"] = self._episode_idx
        return next_state, reward, terminated, info_out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_history(self, seq: Tensor) -> Tensor:
        """Truncate a history tensor to the most recent ``_MAX_HISTORY_LEN`` tokens.

        Applied before every frozen encoder forward to bound sequence
        length. The most recent tokens are kept (right-aligned) so the
        encoder sees the freshest context.

        Args:
            seq: ``LongTensor (B, S)`` of item ids or responses.

        Returns:
            ``seq`` if ``S <= _MAX_HISTORY_LEN``, otherwise the last
            ``_MAX_HISTORY_LEN`` columns of ``seq``.
        """
        S = seq.shape[1]
        if S <= _MAX_HISTORY_LEN:
            return seq
        return seq[:, -_MAX_HISTORY_LEN:]

    def _sample_probes(
        self,
        warmup_q: Tensor,
        tail_q: Sequence[Sequence[int]],
        tail_r: Sequence[Sequence[int]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample per-episode probe sets using ``self._probe_sampler``.

        Delegates the item-id draw to the configured
        :class:`~ordrec.envs.probe_sampler.ProbeSampler` (stratified
        by default, uniform when ``cfg.probe_sampler == "uniform"``).

        Returns:
            ``probe_C (B, M)``, ``probe_H (B, H)``,
            ``probe_H_resp (B, H)`` long tensors on ``self.device``.
        """
        M = int(self.cfg.probe_M)
        H = int(self.cfg.probe_H)
        Q = self.n_questions
        if M + H > Q:
            raise ValueError(
                f"probe_M + probe_H ({M + H}) exceeds item bank size {Q}."
            )

        probe_C = torch.zeros(self.B, M, dtype=torch.long, device=self.device)
        probe_H = torch.zeros(self.B, H, dtype=torch.long, device=self.device)
        probe_H_resp = torch.zeros(self.B, H, dtype=torch.long, device=self.device)

        for b in range(self.B):
            warmup_set = set(int(x) for x in warmup_q[b].cpu().tolist())
            warmup_set.discard(0)
            allowed = np.array(
                [q for q in range(1, Q + 1) if q not in warmup_set],
                dtype=np.int64,
            )
            if allowed.size < M + H:
                # Fall back to allowing duplicates with warmup; the
                # admin mask still forbids overlap with warmup via the
                # no-repeat path, but this keeps the test path running
                # on tiny item banks.
                allowed = np.arange(1, Q + 1, dtype=np.int64)
            choice = self._probe_sampler.sample(self._np_rng, allowed, M + H)
            probe_C[b] = torch.from_numpy(choice[:M].astype(np.int64)).to(
                self.device
            )
            probe_H[b] = torch.from_numpy(choice[M:M + H].astype(np.int64)).to(
                self.device
            )

            # Draw the held-out probe responses from the student's tail
            # if any real attempts are available, otherwise from the
            # student's overall response distribution as a fallback.
            tq = tail_q[b]
            tr = tail_r[b]
            id_to_resp: Dict[int, int] = {}
            for q_item, r_item in zip(tq, tr):
                id_to_resp.setdefault(int(q_item), int(r_item))
            for h in range(H):
                pid = int(probe_H[b, h].item())
                resp = id_to_resp.get(
                    pid,
                    int(tr[h % len(tr)]) if len(tr) > 0 else 0,
                )
                probe_H_resp[b, h] = max(0, min(self.n_categories - 1, int(resp)))

        return probe_C, probe_H, probe_H_resp

    def _build_state(
        self,
        *,
        theta_t: Tensor,
        action_mask: Tensor,
        episode_step: int,
        terminated: bool,
    ) -> OrdinalState:
        raw_info = self._build_info(step_index=episode_step)
        raw_info["horizon_steps"] = self.horizon_steps
        return OrdinalState(
            theta_t=theta_t,
            history_questions=self._history_q,
            history_responses=self._history_r,
            exposure_counts=self._exposure_counts,
            episode_step=int(episode_step),
            terminated=bool(terminated),
            truncated=False,
            action_mask=action_mask,
            raw_info=raw_info,
        )

    def _build_info(self, *, step_index: int) -> Dict[str, Any]:
        # When resample_probe_at_terminal is active, the terminal step
        # provides probe_H_resp_terminal (responses sampled from the world
        # model conditioned on the full simulated history).  The reward
        # composer reads probe_H_resp_terminal first when present; at all
        # non-terminal steps only the reset-time probe_H_resp is present.
        info: Dict[str, Any] = {
            "probe_C_ids": self._probe_C,
            "probe_H_ids": self._probe_H,
            "probe_H_resp": self._probe_H_resp,
            "alpha_table": self.alpha_table,
            "beta_table": self.beta_table,
            "fleet_expo": self.fleet_expo,
            "theta_0": self._theta_0,
            "step_index": int(step_index),
            "horizon_steps": self.horizon_steps,
        }
        if self._probe_H_resp_terminal is not None:
            info["probe_H_resp_terminal"] = self._probe_H_resp_terminal
        return info


__all__ = ["OrdRecEnv"]
