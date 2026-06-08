"""Evaluation harness for a trained PPO checkpoint.

Loads a checkpoint, runs ``n_episodes`` through the env using both the
trained policy and a uniform-random baseline (under the same mask),
and writes a markdown summary to ``results/<experiment>_eval.md``.

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \\
        python rl/scripts/eval_policy.py \\
            --config rl/configs/ppo_synth_smoke.yaml \\
            --checkpoint outputs/ordrec_synth_smoke/best.pt \\
            --n-episodes 16

Output report contains, mean episode return, per-component reward
averages (``r_info``, ``r_cost``, ``r_expo``, ``r_voi``), exposure
diagnostics (per-item fleet rate quantiles), and a side-by-side
comparison against the uniform-random baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# Re-use the train_ppo builders to avoid duplicating adapter / env / world setup.
from train_ppo import _build_adapter, _build_env, _build_ppo, _build_world_model  # type: ignore[import-not-found]

from ordrec.envs import OrdRecEnv
from ordrec.training import PPO, pick_device, set_seed


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------


@dataclass
class _PolicyResult:
    name: str
    mean_return: float
    std_return: float
    component_means: Dict[str, float]
    exposure_quantiles: Dict[str, float]
    n_episodes: int


def _run_policy(
    env: OrdRecEnv,
    *,
    policy: Optional[PPO],
    n_episodes: int,
    seed: int,
    name: str,
    deterministic: bool = False,
) -> _PolicyResult:
    """Run a policy through the env. ``policy=None`` is uniform random."""
    returns: List[float] = []
    comp_sums: Dict[str, List[float]] = {
        "r_info": [], "r_cost": [], "r_expo": [], "r_voi": [],
    }
    for ep in range(int(n_episodes)):
        state = env.reset(seed=seed + ep)
        done = False
        ep_return = torch.zeros(env.B, dtype=torch.float32)
        while not done:
            if policy is None:
                # Uniform random under the env mask.
                prob = state.action_mask.to(torch.float32)
                prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                action = torch.multinomial(
                    prob, num_samples=env.K_B, replacement=False,
                )
            else:
                # Same K_B sequential sampling that PPO uses internally.
                obs_t, mask = policy._unpack_state(state)
                obs_t = obs_t.to(policy.device)
                mask_dev = mask.to(policy.device) if mask is not None else None
                with torch.no_grad():
                    from ordrec.training.ppo import _masked_logits
                    from torch.distributions import Categorical
                    logits, _ = policy.policy(obs_t)
                    cur_mask = mask_dev.clone() if mask_dev is not None else None
                    cols = []
                    for _ in range(env.K_B):
                        masked = _masked_logits(logits, cur_mask)
                        dist = Categorical(logits=masked)
                        a = (
                            torch.argmax(masked, dim=-1)
                            if deterministic else dist.sample()
                        )
                        cols.append(a)
                        if cur_mask is not None:
                            cur_mask = cur_mask.clone()
                            cur_mask.scatter_(
                                dim=1, index=a.unsqueeze(1),
                                src=torch.zeros(
                                    a.shape[0], 1,
                                    dtype=torch.bool, device=cur_mask.device,
                                ),
                            )
                    action = torch.stack(cols, dim=1).to(dtype=torch.long)
            next_state, reward, done, info = env.step(action)
            r = reward.detach().cpu().to(torch.float32) if isinstance(reward, torch.Tensor) else torch.tensor([float(reward)] * env.B)
            ep_return = ep_return + r
            for key in comp_sums.keys():
                val = info.get(key)
                if isinstance(val, torch.Tensor):
                    comp_sums[key].append(float(val.detach().mean().item()))
            state = next_state
        for v in ep_return.tolist():
            returns.append(float(v))

    arr = np.asarray(returns, dtype=np.float64)
    component_means = {
        k: float(np.mean(v)) if v else 0.0 for k, v in comp_sums.items()
    }
    fleet = env.fleet_expo.detach().cpu().numpy()
    exposure_quantiles = {
        "max": float(np.max(fleet)),
        "p99": float(np.quantile(fleet, 0.99)) if fleet.size > 1 else float(np.max(fleet)),
        "p95": float(np.quantile(fleet, 0.95)) if fleet.size > 1 else float(np.max(fleet)),
        "p50": float(np.quantile(fleet, 0.50)) if fleet.size > 1 else float(np.median(fleet)),
        "frac_above_r_max": float(
            np.mean(fleet > env.cfg.r_max)
        ),
    }
    return _PolicyResult(
        name=name,
        mean_return=float(arr.mean()),
        std_return=float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
        component_means=component_means,
        exposure_quantiles=exposure_quantiles,
        n_episodes=int(arr.size),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_markdown(
    cfg_path: Path, ckpt_path: Path,
    trained: _PolicyResult, random: _PolicyResult,
) -> str:
    def _fmt(d: Dict[str, float]) -> str:
        return ", ".join(f"{k}={v:+.4f}" for k, v in d.items())
    lines: List[str] = []
    lines.append(f"# OrdRec PPO evaluation, {cfg_path.stem}")
    lines.append("")
    lines.append(f"Config, `{cfg_path}`.")
    lines.append(f"Checkpoint, `{ckpt_path}`.")
    lines.append("")
    lines.append("## Mean episode return")
    lines.append("")
    lines.append("| policy | mean | std | n_episodes |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| trained PPO | {trained.mean_return:+.4f} "
        f"| {trained.std_return:.4f} | {trained.n_episodes} |"
    )
    lines.append(
        f"| uniform random | {random.mean_return:+.4f} "
        f"| {random.std_return:.4f} | {random.n_episodes} |"
    )
    lines.append("")
    delta = trained.mean_return - random.mean_return
    lines.append(f"Delta (trained minus random), {delta:+.4f}.")
    lines.append("")
    lines.append("## Per-component reward, mean over evaluation")
    lines.append("")
    lines.append("| policy | r_info | r_cost | r_expo | r_voi |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in (trained, random):
        c = r.component_means
        lines.append(
            f"| {r.name} | {c['r_info']:+.4f} | {c['r_cost']:+.4f} "
            f"| {c['r_expo']:+.4f} | {c['r_voi']:+.4f} |"
        )
    lines.append("")
    lines.append("## Exposure diagnostics (fleet EMA after eval)")
    lines.append("")
    lines.append("| policy | max | p99 | p95 | p50 | frac > r_max |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in (trained, random):
        q = r.exposure_quantiles
        lines.append(
            f"| {r.name} | {q['max']:.4f} | {q['p99']:.4f} "
            f"| {q['p95']:.4f} | {q['p50']:.4f} | {q['frac_above_r_max']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"config not found at {args.config}")
    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.get("device"))

    adapter = _build_adapter(cfg["adapter"])
    world = _build_world_model(cfg.get("world_model", {}), adapter, device)
    env = _build_env(cfg, adapter, world, device)
    ppo = _build_ppo(cfg, env, device)
    ppo.load(args.checkpoint)

    trained = _run_policy(
        env, policy=ppo, n_episodes=args.n_episodes, seed=args.seed,
        name="trained PPO", deterministic=args.deterministic,
    )
    # Reset fleet EMA before the random baseline so its exposure
    # diagnostics reflect only the random rollouts.
    env.fleet_expo.zero_()
    random = _run_policy(
        env, policy=None, n_episodes=args.n_episodes, seed=args.seed,
        name="uniform random",
    )

    out = args.output or Path("results") / f"{cfg['experiment_name']}_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _format_markdown(args.config, args.checkpoint, trained, random),
        encoding="utf-8",
    )
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
