"""R1 exposure-penalty ablation: find w_expo / r_max so max-Fisher beats random.

Evaluates three design families without PPO training. For each cell, the test
is: does max-Fisher achieve higher mean return than uniform random with
non-overlapping 95% bootstrap CIs?

Design (a) -- fleet-EMA, vary w_expo and r_max.
Design (b) -- per-episode accounting: penalty fires on intra-episode item
    concentration (fraction of the K_B*horizon budget spent on a single
    difficulty stratum beyond an intra-episode quota), no fleet memory.
Design (c) -- keep fleet-EMA at nominal (a) defaults but sample from
    top-k Fisher instead of argmax, making the effective policy more
    dispersed and less likely to hammer a small set of items.

Usage (from repo root)::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE python rl/scripts/ablation_exposure.py \\
        --config rl/configs/ppo_synth_e45.yaml \\
        --n-episodes 200 \\
        --seed 1234 \\
        --output rl/results/E46b_R1_ablation.md

The script writes a Markdown report to --output and prints results as it goes.
It also writes a companion JSON at the same path with .json extension for
programmatic reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_ppo import _build_adapter, _build_world_model  # type: ignore[import-not-found]

from ordrec.bc_warmstart.bc import (
    gpcm_item_information,
    max_fisher_actions,
    top_k_fisher_soft_target,
)
from ordrec.envs import FrozenMAGPCM, OrdRecEnv, build_item_cache
from ordrec.envs.ordrec_env import OrdRecEnv
from ordrec.reward import OrdinalRewardCompute, RewardConfig
from ordrec.reward.exposure import update_fleet_exposure
from ordrec.training import pick_device, set_seed


# ---------------------------------------------------------------------------
# Per-episode exposure penalty (design b)
# ---------------------------------------------------------------------------


class PerEpisodeExposurePenalty:
    """Intra-episode diversity penalty (design b).

    Instead of a fleet-wide EMA, the penalty fires when the current
    episode has spent more than ``quota_frac`` of its total item budget
    on items from a single difficulty stratum.  Strata are defined by
    the ``strata_map`` argument: a (Q+1,) integer tensor mapping each
    item id to its stratum index (0-indexed, 0 for the pad slot).

    The penalty per batch is::

        sum_s max(0, stratum_frac_s - quota_frac) * w_expo

    where ``stratum_frac_s`` is the fraction of items administered so
    far in the episode that belong to stratum s.

    This is reset to zero each episode by the caller via
    ``reset()``.
    """

    def __init__(
        self,
        strata_map: torch.Tensor,
        n_strata: int,
        *,
        quota_frac: float = 0.40,
    ) -> None:
        self.strata_map = strata_map.long()
        self.n_strata = int(n_strata)
        self.quota_frac = float(quota_frac)
        # Episode-level per-row stratum counts -- initialised at reset.
        self._counts: Optional[torch.Tensor] = None
        self._total: int = 0

    def reset(self, B: int, device: torch.device) -> None:
        self._counts = torch.zeros(
            B, self.n_strata, dtype=torch.long, device=device
        )
        self._total = 0

    def update_and_penalise(
        self, action: torch.Tensor, w_expo: float
    ) -> torch.Tensor:
        """Update episode counts and return penalty (B,).

        Args:
            action: ``LongTensor (B, K_B)`` item ids (1-based; 0 = pad).
            w_expo: Penalty weight.

        Returns:
            ``Tensor (B,)`` penalty (non-negative).
        """
        if self._counts is None:
            raise RuntimeError("Call reset() before update_and_penalise().")
        B, K_B = action.shape
        device = action.device
        strata_map = self.strata_map.to(device)
        # Map each action item to its stratum.  Pad slot -> stratum 0.
        action_strata = strata_map[action.clamp(min=0).long()]  # (B, K_B)
        # Update per-row, per-stratum counts.
        for k in range(K_B):
            col = action_strata[:, k]  # (B,)
            for b in range(B):
                s = int(col[b].item())
                self._counts[b, s] += 1
        self._total += K_B

        if self._total == 0:
            return torch.zeros(B, device=device)

        # Fraction of total items from each stratum per row.
        frac = self._counts.float() / float(self._total)  # (B, n_strata)
        # Hinge above quota.
        hinge = (frac - self.quota_frac).clamp_min(0.0)  # (B, n_strata)
        penalty = w_expo * hinge.sum(dim=-1)  # (B,)
        return penalty


# ---------------------------------------------------------------------------
# Env builder that accepts an arbitrary RewardConfig override
# ---------------------------------------------------------------------------


def _build_env_with_cfg(
    base_cfg: Dict[str, Any],
    adapter: Any,
    world: FrozenMAGPCM,
    device: torch.device,
    reward_cfg: RewardConfig,
) -> OrdRecEnv:
    """Build an OrdRecEnv with a given reward config (override base_cfg.reward)."""
    env_cfg = base_cfg.get("env", {})
    cache = build_item_cache(
        world,
        n_contexts=int(env_cfg.get("cache_n_contexts", 2)),
        context_seq_len=int(env_cfg.get("cache_context_seq_len", 1)),
        dataset_name=base_cfg["experiment_name"],
    )
    n_categories = adapter.get_n_categories()
    reward = OrdinalRewardCompute(reward_cfg, n_categories=n_categories)
    env = OrdRecEnv(
        world_model=world,
        adapter=adapter,
        item_cache=cache,
        reward_fn=reward,
        cfg=reward_cfg,
        batch_size=int(env_cfg.get("batch_size", 4)),
        warmup_len=int(env_cfg.get("warmup_len", 3)),
        split="test",
        device=device,
        seed=int(base_cfg.get("seed", 0)),
    )
    return env


# ---------------------------------------------------------------------------
# Policy runners
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Single policy result for one ablation cell."""
    name: str
    mean_return: float
    std_return: float
    ci_lo: float
    ci_hi: float
    component_means: Dict[str, float]
    max_exposure: float
    frac_above_r_max: float
    n_episodes: int
    n_students: int  # B * n_episodes


def _bootstrap_ci(
    arr: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return lo, hi


def _run_max_fisher(
    env: OrdRecEnv,
    *,
    n_episodes: int,
    seed: int,
    top_k: Optional[int] = None,
) -> RunResult:
    """Run greedy max-Fisher (top_k=None) or top-k stochastic (top_k=int)."""
    name = "max-Fisher" if top_k is None else f"top-{top_k}-Fisher"
    returns: List[float] = []
    comp_sums: Dict[str, List[float]] = {
        "r_info": [], "r_cost": [], "r_expo": [], "r_voi": [],
    }
    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        done = False
        ep_return = torch.zeros(env.B, dtype=torch.float32)
        while not done:
            cur_mask = state.action_mask.to(env.device).clone()
            theta = state.theta_t.to(env.device)
            cols = []
            for _ in range(env.K_B):
                if top_k is None:
                    a = max_fisher_actions(
                        theta, env.alpha_table, env.beta_table, cur_mask
                    )
                else:
                    soft = top_k_fisher_soft_target(
                        theta, env.alpha_table, env.beta_table, cur_mask, k=top_k,
                    )
                    soft = soft.clamp_min(0.0)
                    a = torch.multinomial(
                        soft.clamp_min(1e-12), num_samples=1,
                    ).squeeze(1)
                cols.append(a)
                cur_mask = cur_mask.clone()
                cur_mask.scatter_(
                    dim=1, index=a.unsqueeze(1),
                    src=torch.zeros(a.shape[0], 1, dtype=torch.bool, device=cur_mask.device),
                )
            action = torch.stack(cols, dim=1).to(dtype=torch.long, device="cpu")
            next_state, reward, done, info = env.step(action)
            r = reward.detach().cpu().to(torch.float32)
            ep_return = ep_return + r
            for key in comp_sums:
                val = info.get(key)
                if isinstance(val, torch.Tensor):
                    comp_sums[key].append(float(val.detach().mean().item()))
            state = next_state
        returns.extend(ep_return.tolist())

    arr = np.asarray(returns, dtype=np.float64)
    fleet = env.fleet_expo.detach().cpu().numpy()
    ci_lo, ci_hi = _bootstrap_ci(arr)
    return RunResult(
        name=name,
        mean_return=float(arr.mean()),
        std_return=float(arr.std(ddof=0)),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        component_means={k: float(np.mean(v)) if v else 0.0 for k, v in comp_sums.items()},
        max_exposure=float(np.max(fleet)),
        frac_above_r_max=float(np.mean(fleet > env.cfg.r_max)),
        n_episodes=n_episodes,
        n_students=n_episodes * env.B,
    )


def _run_random(
    env: OrdRecEnv,
    *,
    n_episodes: int,
    seed: int,
) -> RunResult:
    """Run uniform random policy."""
    returns: List[float] = []
    comp_sums: Dict[str, List[float]] = {
        "r_info": [], "r_cost": [], "r_expo": [], "r_voi": [],
    }
    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        done = False
        ep_return = torch.zeros(env.B, dtype=torch.float32)
        while not done:
            prob = state.action_mask.to(torch.float32)
            prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            action = torch.multinomial(prob, num_samples=env.K_B, replacement=False)
            next_state, reward, done, info = env.step(action)
            r = reward.detach().cpu().to(torch.float32)
            ep_return = ep_return + r
            for key in comp_sums:
                val = info.get(key)
                if isinstance(val, torch.Tensor):
                    comp_sums[key].append(float(val.detach().mean().item()))
            state = next_state
        returns.extend(ep_return.tolist())

    arr = np.asarray(returns, dtype=np.float64)
    fleet = env.fleet_expo.detach().cpu().numpy()
    ci_lo, ci_hi = _bootstrap_ci(arr)
    return RunResult(
        name="random",
        mean_return=float(arr.mean()),
        std_return=float(arr.std(ddof=0)),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        component_means={k: float(np.mean(v)) if v else 0.0 for k, v in comp_sums.items()},
        max_exposure=float(np.max(fleet)),
        frac_above_r_max=float(np.mean(fleet > env.cfg.r_max)),
        n_episodes=n_episodes,
        n_students=n_episodes * env.B,
    )


def _beats_random(fisher: RunResult, rand: RunResult) -> bool:
    """True when max-Fisher mean > random mean AND CIs do not overlap."""
    return (
        fisher.mean_return > rand.mean_return
        and fisher.ci_lo > rand.ci_hi
    )


# ---------------------------------------------------------------------------
# Per-episode env wrapper (design b)
# ---------------------------------------------------------------------------


class _PerEpisodeEnv:
    """Thin wrapper that overrides the reward's r_expo component.

    Design (b): no fleet memory.  The exposure penalty fires only when
    the current episode concentrates items in a single difficulty stratum
    beyond ``quota_frac`` of the total items administered so far.

    This wraps OrdRecEnv.step() to intercept the reward breakdown and
    replace r_expo with the intra-episode penalty.  The underlying env
    is unchanged; only the returned reward differs.
    """

    def __init__(
        self,
        env: OrdRecEnv,
        strata_map: torch.Tensor,
        n_strata: int,
        *,
        quota_frac: float = 0.40,
        w_expo: float = 0.10,
    ) -> None:
        self._env = env
        self._pen = PerEpisodeExposurePenalty(
            strata_map, n_strata, quota_frac=quota_frac,
        )
        self.w_expo = float(w_expo)
        # Expose env attributes for policy runners.
        self.B = env.B
        self.K_B = env.K_B
        self.device = env.device
        self.alpha_table = env.alpha_table
        self.beta_table = env.beta_table
        self.cfg = env.cfg
        self.fleet_expo = env.fleet_expo

    def reset(self, seed: Optional[int] = None) -> Any:
        state = self._env.reset(seed=seed)
        self._pen.reset(self.B, self.device)
        return state

    def step(self, action: torch.Tensor) -> Tuple[Any, torch.Tensor, bool, Dict]:
        next_state, reward, done, info = self._env.step(action)
        # Override the r_expo component.
        new_r_expo = -self._pen.update_and_penalise(
            action.to(self.device), self.w_expo
        ).cpu()
        old_r_expo = info.get("r_expo", torch.zeros(self.B))
        if isinstance(old_r_expo, torch.Tensor):
            old_r_expo = old_r_expo.detach().cpu()
        delta = new_r_expo - old_r_expo
        # Patch reward and info.
        reward = reward.detach().cpu() + delta
        info = dict(info)
        info["r_expo"] = new_r_expo
        info["r_total"] = reward
        return next_state, reward, done, info


def _run_policies_in_wrapper(
    wrapper: _PerEpisodeEnv,
    *,
    n_episodes: int,
    seed: int,
    use_max_fisher: bool,
    top_k: Optional[int] = None,
) -> RunResult:
    """Run a policy through a _PerEpisodeEnv wrapper."""
    name: str
    if use_max_fisher:
        name = "max-Fisher (per-ep)" if top_k is None else f"top-{top_k}-Fisher (per-ep)"
    else:
        name = "random (per-ep)"

    returns: List[float] = []
    comp_sums: Dict[str, List[float]] = {
        "r_info": [], "r_cost": [], "r_expo": [], "r_voi": [],
    }
    env = wrapper._env
    for ep in range(n_episodes):
        state = wrapper.reset(seed=seed + ep)
        done = False
        ep_return = torch.zeros(wrapper.B, dtype=torch.float32)
        while not done:
            if use_max_fisher:
                cur_mask = state.action_mask.to(wrapper.device).clone()
                theta = state.theta_t.to(wrapper.device)
                cols = []
                for _ in range(wrapper.K_B):
                    if top_k is None:
                        a = max_fisher_actions(
                            theta, wrapper.alpha_table, wrapper.beta_table, cur_mask
                        )
                    else:
                        soft = top_k_fisher_soft_target(
                            theta, wrapper.alpha_table, wrapper.beta_table,
                            cur_mask, k=top_k,
                        )
                        a = torch.multinomial(
                            soft.clamp_min(1e-12), num_samples=1,
                        ).squeeze(1)
                    cols.append(a)
                    cur_mask = cur_mask.clone()
                    cur_mask.scatter_(
                        dim=1, index=a.unsqueeze(1),
                        src=torch.zeros(a.shape[0], 1, dtype=torch.bool, device=cur_mask.device),
                    )
                action = torch.stack(cols, dim=1).to(dtype=torch.long, device="cpu")
            else:
                prob = state.action_mask.to(torch.float32)
                prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                action = torch.multinomial(prob, num_samples=wrapper.K_B, replacement=False)

            next_state, reward, done, info = wrapper.step(action)
            r = reward.detach().cpu().to(torch.float32)
            ep_return = ep_return + r
            for key in comp_sums:
                val = info.get(key)
                if isinstance(val, torch.Tensor):
                    comp_sums[key].append(float(val.detach().mean().item()))
            state = next_state
        returns.extend(ep_return.tolist())

    arr = np.asarray(returns, dtype=np.float64)
    fleet = wrapper.fleet_expo.detach().cpu().numpy()
    ci_lo, ci_hi = _bootstrap_ci(arr)
    r_max = float(wrapper.cfg.r_max)
    return RunResult(
        name=name,
        mean_return=float(arr.mean()),
        std_return=float(arr.std(ddof=0)),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        component_means={k: float(np.mean(v)) if v else 0.0 for k, v in comp_sums.items()},
        max_exposure=float(np.max(fleet)),
        frac_above_r_max=float(np.mean(fleet > r_max)),
        n_episodes=n_episodes,
        n_students=n_episodes * wrapper.B,
    )


# ---------------------------------------------------------------------------
# Ablation cells
# ---------------------------------------------------------------------------


def _base_reward_cfg(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_cfg = base_cfg.get("env", {})
    r = dict(base_cfg.get("reward", {}))
    r.setdefault("K_B", env_cfg.get("K_B", 5))
    r.setdefault("T", env_cfg.get("T", 10))
    return r


def run_ablation(
    base_cfg: Dict[str, Any],
    adapter: Any,
    world: FrozenMAGPCM,
    device: torch.device,
    *,
    n_episodes: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run all ablation cells and return a list of result dicts."""
    base_r = _base_reward_cfg(base_cfg)
    K_B = int(base_r.get("K_B", 5))
    T = int(base_r.get("T", 10))

    results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Design (a): fleet-EMA, sweep w_expo x r_max
    # ------------------------------------------------------------------
    design_a_grid = [
        (0.00, 0.20),
        (0.02, 0.20),
        (0.05, 0.20),
        (0.10, 0.20),  # original defaults
        (0.00, 0.40),
        (0.02, 0.40),
        (0.05, 0.40),
        (0.10, 0.40),
    ]

    for w_expo, r_max in design_a_grid:
        label = f"a(w={w_expo:.2f},r={r_max:.2f})"
        print(f"\n--- {label} ---", flush=True)
        rdict = dict(base_r)
        rdict["w_expo"] = w_expo
        rdict["r_max"] = r_max
        rc = RewardConfig.from_dict(rdict)
        env = _build_env_with_cfg(base_cfg, adapter, world, device, rc)

        env.fleet_expo.zero_()
        fisher_res = _run_max_fisher(env, n_episodes=n_episodes, seed=seed)
        print(
            f"  max-Fisher: {fisher_res.mean_return:+.4f} CI [{fisher_res.ci_lo:+.4f}, {fisher_res.ci_hi:+.4f}]"
            f"  r_expo={fisher_res.component_means['r_expo']:+.4f}  max_expo={fisher_res.max_exposure:.3f}",
            flush=True,
        )

        env.fleet_expo.zero_()
        random_res = _run_random(env, n_episodes=n_episodes, seed=seed)
        print(
            f"  random:     {random_res.mean_return:+.4f} CI [{random_res.ci_lo:+.4f}, {random_res.ci_hi:+.4f}]"
            f"  r_expo={random_res.component_means['r_expo']:+.4f}  max_expo={random_res.max_exposure:.3f}",
            flush=True,
        )
        beats = _beats_random(fisher_res, random_res)
        print(f"  fisher_beats_random={beats}", flush=True)

        results.append({
            "design": "a",
            "label": label,
            "w_expo": w_expo,
            "r_max": r_max,
            "fisher": _result_to_dict(fisher_res),
            "random": _result_to_dict(random_res),
            "fisher_beats_random": beats,
        })

    # ------------------------------------------------------------------
    # Design (b): per-episode stratum diversity penalty (nominal weights)
    # ------------------------------------------------------------------
    print("\n--- b(per-episode, nominal w=0.10, quota=0.40) ---", flush=True)
    rdict_b = dict(base_r)
    # Fleet EMA weight is zeroed because per-episode penalty replaces it.
    rdict_b["w_expo"] = 0.0
    rc_b = RewardConfig.from_dict(rdict_b)
    env_b = _build_env_with_cfg(base_cfg, adapter, world, device, rc_b)

    # Build strata_map from item betas: mean beta across K-1 thresholds,
    # quantile-bin into n_strata bins.
    n_strata = int(base_r.get("n_difficulty_strata", 5))
    beta = env_b.beta_table.cpu()  # (Q+1, K-1)
    mean_beta = beta.mean(dim=-1)  # (Q+1,) -- pad slot is 0
    # Assign strata by quantile of mean_beta[1:].
    item_means = mean_beta[1:].numpy()
    bins = np.nanpercentile(item_means, np.linspace(0, 100, n_strata + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    strata_labels = np.digitize(item_means, bins[1:-1])  # 0-indexed strata
    strata_map = torch.zeros(len(mean_beta), dtype=torch.long)
    strata_map[1:] = torch.from_numpy(strata_labels.astype(np.int64))

    wrap_b = _PerEpisodeEnv(
        env_b, strata_map, n_strata, quota_frac=0.40, w_expo=0.10
    )

    wrap_b._env.fleet_expo.zero_()
    fisher_b = _run_policies_in_wrapper(
        wrap_b, n_episodes=n_episodes, seed=seed, use_max_fisher=True,
    )
    print(
        f"  max-Fisher (per-ep): {fisher_b.mean_return:+.4f} CI [{fisher_b.ci_lo:+.4f}, {fisher_b.ci_hi:+.4f}]"
        f"  r_expo={fisher_b.component_means['r_expo']:+.4f}",
        flush=True,
    )

    wrap_b._env.fleet_expo.zero_()
    random_b = _run_policies_in_wrapper(
        wrap_b, n_episodes=n_episodes, seed=seed, use_max_fisher=False,
    )
    print(
        f"  random (per-ep):     {random_b.mean_return:+.4f} CI [{random_b.ci_lo:+.4f}, {random_b.ci_hi:+.4f}]"
        f"  r_expo={random_b.component_means['r_expo']:+.4f}",
        flush=True,
    )
    beats_b = _beats_random(fisher_b, random_b)
    print(f"  fisher_beats_random={beats_b}", flush=True)

    results.append({
        "design": "b",
        "label": "b(per-episode, quota=0.40, w=0.10)",
        "w_expo": 0.10,
        "r_max": None,
        "quota_frac": 0.40,
        "fisher": _result_to_dict(fisher_b),
        "random": _result_to_dict(random_b),
        "fisher_beats_random": beats_b,
    })

    # ------------------------------------------------------------------
    # Design (c): top-k stochastic Fisher, nominal fleet-EMA weights
    # ------------------------------------------------------------------
    print("\n--- c(top-20-Fisher stochastic, nominal w=0.10, r_max=0.20) ---", flush=True)
    rdict_c = dict(base_r)
    rc_c = RewardConfig.from_dict(rdict_c)
    env_c = _build_env_with_cfg(base_cfg, adapter, world, device, rc_c)

    env_c.fleet_expo.zero_()
    fisher_c = _run_max_fisher(env_c, n_episodes=n_episodes, seed=seed, top_k=20)
    print(
        f"  top-20-Fisher: {fisher_c.mean_return:+.4f} CI [{fisher_c.ci_lo:+.4f}, {fisher_c.ci_hi:+.4f}]"
        f"  r_expo={fisher_c.component_means['r_expo']:+.4f}  max_expo={fisher_c.max_exposure:.3f}",
        flush=True,
    )

    env_c.fleet_expo.zero_()
    random_c = _run_random(env_c, n_episodes=n_episodes, seed=seed)
    print(
        f"  random:        {random_c.mean_return:+.4f} CI [{random_c.ci_lo:+.4f}, {random_c.ci_hi:+.4f}]"
        f"  r_expo={random_c.component_means['r_expo']:+.4f}  max_expo={random_c.max_exposure:.3f}",
        flush=True,
    )
    beats_c = _beats_random(fisher_c, random_c)
    print(f"  fisher_beats_random={beats_c}", flush=True)

    results.append({
        "design": "c",
        "label": "c(top-20-Fisher, w=0.10, r_max=0.20)",
        "w_expo": float(rdict_c.get("w_expo", 0.10)),
        "r_max": float(rdict_c.get("r_max", 0.20)),
        "top_k": 20,
        "fisher": _result_to_dict(fisher_c),
        "random": _result_to_dict(random_c),
        "fisher_beats_random": beats_c,
    })

    return results


def _result_to_dict(r: RunResult) -> Dict[str, Any]:
    return {
        "name": r.name,
        "mean_return": r.mean_return,
        "std_return": r.std_return,
        "ci_lo": r.ci_lo,
        "ci_hi": r.ci_hi,
        "component_means": r.component_means,
        "max_exposure": r.max_exposure,
        "frac_above_r_max": r.frac_above_r_max,
        "n_episodes": r.n_episodes,
        "n_students": r.n_students,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_report(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# E4.6b R1 Exposure Penalty Ablation")
    lines.append("")
    lines.append(
        "Test: max-Fisher must beat uniform random on mean return "
        "with non-overlapping 95% bootstrap CIs."
    )
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| design | label | fisher_mean | fisher_CI | random_mean | random_CI"
        " | fisher_beats_random | max_expo |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for r in results:
        f = r["fisher"]
        rand = r["random"]
        beaten = "YES" if r["fisher_beats_random"] else "no"
        lines.append(
            f"| {r['design']} | {r['label']} "
            f"| {f['mean_return']:+.4f} | [{f['ci_lo']:+.4f}, {f['ci_hi']:+.4f}] "
            f"| {rand['mean_return']:+.4f} | [{rand['ci_lo']:+.4f}, {rand['ci_hi']:+.4f}] "
            f"| {beaten} | {f['max_exposure']:.3f} |"
        )
    lines.append("")
    lines.append("## Per-component means (max-Fisher)")
    lines.append("")
    lines.append("| label | r_info | r_cost | r_expo | r_voi |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        f = r["fisher"]
        c = f["component_means"]
        lines.append(
            f"| {r['label']} | {c['r_info']:+.4f} | {c['r_cost']:+.4f}"
            f" | {c['r_expo']:+.4f} | {c['r_voi']:+.4f} |"
        )
    lines.append("")
    lines.append("## Per-component means (random)")
    lines.append("")
    lines.append("| label | r_info | r_cost | r_expo | r_voi |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        rand = r["random"]
        c = rand["component_means"]
        lines.append(
            f"| {r['label']} | {c['r_info']:+.4f} | {c['r_cost']:+.4f}"
            f" | {c['r_expo']:+.4f} | {c['r_voi']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 exposure ablation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    with args.config.open("r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    set_seed(int(base_cfg.get("seed", 0)))
    device = pick_device(base_cfg.get("device"))
    base_cfg["env"]["split"] = "test"

    adapter = _build_adapter(base_cfg["adapter"])
    world = _build_world_model(base_cfg.get("world_model", {}), adapter, device)

    results = run_ablation(
        base_cfg, adapter, world, device,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )

    report = _format_report(results)
    out = args.output or Path("rl/results") / "E46b_R1_ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {out}")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
