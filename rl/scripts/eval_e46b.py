"""E4.6b four-policy evaluation harness.

Re-uses eval_e45.py logic with paths updated for the E4.6b B-side run.
Outputs to rl/results/E46b_bside_eval.{md,json}.

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \\
        python rl/scripts/eval_e46b.py \\
            --config rl/configs/ppo_synth_e46b.yaml \\
            --ppo-checkpoint outputs/ordrec_synth_e46b/best.pt \\
            --bc-checkpoint outputs/ordrec_synth_e46b/bc_warmstart.pt \\
            --n-episodes 200 \\
            --output rl/results/E46b_bside_eval.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

# eval_e45 is in the same scripts directory; the import works when
# the scripts directory is on sys.path (set by PYTHONPATH or cwd).
import os
sys.path.insert(0, str(Path(__file__).parent))

from eval_e45 import (  # type: ignore[import-not-found]
    PolicyResult,
    _run_policy,
    _format_markdown,
)
from train_ppo import (  # type: ignore[import-not-found]
    _build_adapter,
    _build_env,
    _build_ppo,
    _build_world_model,
)

from ordrec.training import PPO, pick_device, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E4.6b four-policy evaluation (PPO, BC-only, max-Fisher, random)."
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

    # Force test split for evaluation.
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
        "B-side of E4.5/E4.6b A/B comparison.",
        "Buffer fix (RC1): one entry per env-step per row; terminal r_voi enters training.",
        "Reward recalibration (RC2): w_expo=0.02, r_max=0.40 (was 0.10, 0.20 in E4.5).",
        "Stratified probe sampler (B5): difficulty-stratified, 5 strata.",
        "BC teacher uses top-5 soft target (R3).",
        f"Evaluated on test split, {args.n_episodes} episodes per policy.",
        f"PPO checkpoint: {args.ppo_checkpoint}.",
    ]
    if args.bc_checkpoint is not None:
        notes.append(f"BC checkpoint: {args.bc_checkpoint}.")

    report = _format_markdown(
        args.config, args.ppo_checkpoint, args.bc_checkpoint, results, notes
    )

    out = args.output or Path("rl/results") / "E46b_bside_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out}", flush=True)
    print(report)

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
