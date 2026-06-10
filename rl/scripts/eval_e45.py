"""E4.5 four-policy evaluation harness.

Compares four policies on the held-out synthetic test split:
  1. trained PPO (best.pt)
  2. BC-only (bc_warmstart.pt, policy frozen after warm-start)
  3. greedy max-Fisher (teacher from BC warm-start)
  4. uniform random (action-mask-respecting)

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \\
        python rl/scripts/eval_e45.py \\
            --config rl/configs/ppo_synth_e45.yaml \\
            --ppo-checkpoint outputs/ordrec_synth_e45/best.pt \\
            --bc-checkpoint outputs/ordrec_synth_e45/bc_warmstart.pt \\
            --n-episodes 200 \\
            --output rl/results/e45_eval.md

Notes on design choices:
  - All four policies share the same env reset seed sequence so
    comparisons are on the same student trajectories.
  - Fleet exposure EMA is reset between each policy so exposure
    diagnostics reflect only that policy's rollouts.
  - max-Fisher is evaluated via repeated argmax (no sampling),
    matching the teacher used in BC warm-start.
  - Probe sampler is uniform (stratified is E4.6b per plan).
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

# Re-use train_ppo builders.
from train_ppo import _build_adapter, _build_env, _build_ppo, _build_world_model  # type: ignore[import-not-found]

from ordrec.bc_warmstart.bc import gpcm_item_information, max_fisher_actions
from ordrec.envs import OrdRecEnv
from ordrec.training import PPO, pick_device, set_seed


# ---------------------------------------------------------------------------
# Single policy runner
# ---------------------------------------------------------------------------


@dataclass
class PolicyResult:
    """Per-policy evaluation summary."""
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
    use_max_fisher: bool = False,
) -> PolicyResult:
    """Run a single policy and collect statistics.

    Args:
        env: OrdRecEnv instance.
        policy: PPO object (trained or BC-only). None -> random.
        n_episodes: Number of episodes to roll out.
        seed: Base seed for env resets.
        name: Display name for this policy.
        use_max_fisher: If True, ignore policy and use greedy max-Fisher.
    """
    returns: List[float] = []
    comp_sums: Dict[str, List[float]] = {
        "r_info": [], "r_cost": [], "r_expo": [], "r_voi": [],
    }

    for ep in range(int(n_episodes)):
        state = env.reset(seed=seed + ep)
        done = False
        ep_return = torch.zeros(env.B, dtype=torch.float32)

        while not done:
            if use_max_fisher:
                # Greedy max-Fisher action for all K_B slots sequentially.
                cur_mask = state.action_mask.to(env.device).clone()
                theta = state.theta_t.to(env.device)
                cols = []
                for _ in range(env.K_B):
                    a = max_fisher_actions(
                        theta, env.alpha_table, env.beta_table, cur_mask
                    )
                    cols.append(a)
                    cur_mask = cur_mask.clone()
                    cur_mask.scatter_(
                        dim=1, index=a.unsqueeze(1),
                        src=torch.zeros(
                            a.shape[0], 1, dtype=torch.bool, device=cur_mask.device,
                        ),
                    )
                action = torch.stack(cols, dim=1).to(dtype=torch.long, device="cpu")

            elif policy is None:
                # Uniform random under the env mask.
                prob = state.action_mask.to(torch.float32)
                prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                action = torch.multinomial(prob, num_samples=env.K_B, replacement=False)

            else:
                # PPO or BC-only policy (stochastic).
                from ordrec.training.ppo import _masked_logits
                from torch.distributions import Categorical

                obs_t = state.to_tensor(policy.device)
                mask_dev = state.action_mask.to(policy.device)
                with torch.no_grad():
                    logits, _ = policy.policy(obs_t)
                    cur_mask = mask_dev.clone()
                    cols = []
                    for _ in range(env.K_B):
                        masked = _masked_logits(logits, cur_mask)
                        a = Categorical(logits=masked).sample()
                        cols.append(a)
                        cur_mask = cur_mask.clone()
                        cur_mask.scatter_(
                            dim=1, index=a.unsqueeze(1),
                            src=torch.zeros(
                                a.shape[0], 1, dtype=torch.bool, device=cur_mask.device,
                            ),
                        )
                    action = torch.stack(cols, dim=1).to(dtype=torch.long, device="cpu")

            next_state, reward, done, info = env.step(action)
            r = (
                reward.detach().cpu().to(torch.float32)
                if isinstance(reward, torch.Tensor)
                else torch.tensor([float(reward)] * env.B)
            )
            ep_return = ep_return + r
            for key in comp_sums:
                val = info.get(key)
                if isinstance(val, torch.Tensor):
                    comp_sums[key].append(float(val.detach().mean().item()))
                elif val is not None:
                    comp_sums[key].append(float(val))
            state = next_state

        for v in ep_return.tolist():
            returns.append(float(v))

    arr = np.asarray(returns, dtype=np.float64)
    component_means = {k: float(np.mean(v)) if v else 0.0 for k, v in comp_sums.items()}
    fleet = env.fleet_expo.detach().cpu().numpy()
    exposure_quantiles = {
        "max": float(np.max(fleet)),
        "p99": float(np.quantile(fleet, 0.99)) if fleet.size > 1 else float(np.max(fleet)),
        "p95": float(np.quantile(fleet, 0.95)) if fleet.size > 1 else float(np.max(fleet)),
        "p50": float(np.quantile(fleet, 0.50)) if fleet.size > 1 else float(np.median(fleet)),
        "frac_above_r_max": float(np.mean(fleet > env.cfg.r_max)),
    }
    return PolicyResult(
        name=name,
        mean_return=float(arr.mean()),
        std_return=float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
        component_means=component_means,
        exposure_quantiles=exposure_quantiles,
        n_episodes=int(arr.size),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_markdown(
    cfg_path: Path,
    ppo_ckpt: Path,
    bc_ckpt: Optional[Path],
    results: List[PolicyResult],
    notes: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# OrdRec E4.5 four-policy evaluation")
    lines.append("")
    lines.append(f"Config: `{cfg_path}`.")
    lines.append(f"PPO checkpoint: `{ppo_ckpt}`.")
    if bc_ckpt is not None:
        lines.append(f"BC checkpoint: `{bc_ckpt}`.")
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("## Mean episode return")
    lines.append("")
    lines.append("| policy | mean return | std | n_episodes |")
    lines.append("| --- | --- | --- | --- |")
    for r in results:
        lines.append(
            f"| {r.name} | {r.mean_return:+.4f} | {r.std_return:.4f} | {r.n_episodes} |"
        )
    lines.append("")
    lines.append("## Per-component reward means")
    lines.append("")
    lines.append("| policy | r_info | r_cost | r_expo | r_voi |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        c = r.component_means
        lines.append(
            f"| {r.name} | {c['r_info']:+.4f} | {c['r_cost']:+.4f} "
            f"| {c['r_expo']:+.4f} | {c['r_voi']:+.4f} |"
        )
    lines.append("")
    lines.append("## Exposure diagnostics (fleet EMA at end of policy eval)")
    lines.append("")
    lines.append("| policy | max | p99 | p95 | p50 | frac > r_max |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        q = r.exposure_quantiles
        lines.append(
            f"| {r.name} | {q['max']:.4f} | {q['p99']:.4f} "
            f"| {q['p95']:.4f} | {q['p50']:.4f} | {q['frac_above_r_max']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E4.5 four-policy evaluation (PPO, BC-only, max-Fisher, random)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.get("device"))

    # Override eval split to test.
    cfg["adapter"]["split"] = "test" if "split" not in cfg.get("adapter", {}) else cfg["adapter"].get("split", "test")
    if "env" in cfg:
        cfg["env"]["split"] = "test"

    adapter = _build_adapter(cfg["adapter"])
    world = _build_world_model(cfg.get("world_model", {}), adapter, device)
    env = _build_env(cfg, adapter, world, device)
    ppo = _build_ppo(cfg, env, device)

    # 1. Trained PPO
    ppo.load(args.ppo_checkpoint)
    env.fleet_expo.zero_()
    result_ppo = _run_policy(
        env, policy=ppo, n_episodes=args.n_episodes, seed=args.seed,
        name="trained PPO",
    )
    print(f"trained PPO: mean_return={result_ppo.mean_return:+.4f}", flush=True)

    # 2. BC-only
    result_bc: Optional[PolicyResult] = None
    if args.bc_checkpoint is not None and args.bc_checkpoint.exists():
        ppo_bc = _build_ppo(cfg, env, device)
        ppo_bc.load(args.bc_checkpoint)
        env.fleet_expo.zero_()
        result_bc = _run_policy(
            env, policy=ppo_bc, n_episodes=args.n_episodes, seed=args.seed,
            name="BC-only",
        )
        print(f"BC-only: mean_return={result_bc.mean_return:+.4f}", flush=True)

    # 3. Max-Fisher
    env.fleet_expo.zero_()
    result_fisher = _run_policy(
        env, policy=None, n_episodes=args.n_episodes, seed=args.seed,
        name="max-Fisher", use_max_fisher=True,
    )
    print(f"max-Fisher: mean_return={result_fisher.mean_return:+.4f}", flush=True)

    # 4. Uniform random
    env.fleet_expo.zero_()
    result_random = _run_policy(
        env, policy=None, n_episodes=args.n_episodes, seed=args.seed,
        name="uniform random",
    )
    print(f"uniform random: mean_return={result_random.mean_return:+.4f}", flush=True)

    results = [result_ppo]
    if result_bc is not None:
        results.append(result_bc)
    results += [result_fisher, result_random]

    notes = [
        "Probe sampler is uniform (stratified lands in E4.6b).",
        "K_B credit assignment: full reward on first sub-step (known-suboptimal, fixed in E4.6b). "
        "This is side A of the E4.6b A/B comparison.",
        f"Evaluated on test split, {args.n_episodes} episodes per policy.",
        f"PPO checkpoint: {args.ppo_checkpoint}.",
    ]
    if args.bc_checkpoint is not None:
        notes.append(f"BC checkpoint: {args.bc_checkpoint}.")
    else:
        notes.append("BC-only checkpoint not available, skipped.")

    report = _format_markdown(
        args.config, args.ppo_checkpoint, args.bc_checkpoint, results, notes
    )

    out = args.output or Path("rl/results") / f"{cfg['experiment_name']}_e45_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out}", flush=True)
    print(report)

    # Save JSON alongside markdown for programmatic use.
    json_out = out.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "mean_return": r.mean_return,
                    "std_return": r.std_return,
                    "component_means": r.component_means,
                    "exposure_quantiles": r.exposure_quantiles,
                    "n_episodes": r.n_episodes,
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
