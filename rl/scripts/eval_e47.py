"""E4.7 four-policy evaluation for staircase and random-walk cohorts.

Runs PPO, BC-only, max-Fisher, and uniform-random policies through the
OrdRec env (200 episodes each, B=32) for both the staircase and
random-walk cohorts.  Writes per-policy results to JSON and produces
a markdown summary.

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \\
        python rl/scripts/eval_e47.py

All paths are hard-coded relative to the repo root.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# Add rl/scripts to path so we can import train_ppo builders.
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from train_ppo import _build_adapter, _build_env, _build_world_model  # noqa: E402

from ordrec.envs import OrdRecEnv
from ordrec.reward.probe_entropy import phi_entropy
from ordrec.training import PPO, pick_device, set_seed

# ---------------------------------------------------------------------------
# Policy runner
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    name: str
    mean_return: float
    std_return: float
    ci_low: float
    ci_high: float
    r_info: float
    r_cost: float
    r_expo: float
    r_voi: float
    r_voi_positive_frac: float
    n_episodes: int


def _run_policy(
    env: OrdRecEnv,
    *,
    policy: Optional[PPO],
    n_episodes: int,
    seed: int,
    name: str,
    deterministic: bool = False,
    use_fisher: bool = False,
) -> PolicyResult:
    """Run a policy. policy=None -> uniform random. use_fisher=True -> max-Fisher."""
    returns: List[float] = []
    comp_sums: Dict[str, List[float]] = {
        "r_info": [], "r_cost": [], "r_expo": [], "r_voi": [],
    }
    r_voi_all: List[float] = []

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        done = False
        ep_return = torch.zeros(env.B, dtype=torch.float32)
        while not done:
            if use_fisher:
                # Max-Fisher: select items with highest GPCM predictive entropy
                # (entropy as a proxy for expected Fisher information).
                from ordrec.reward.gpcm_ops import gpcm_log_probs
                action_mask = state.action_mask  # (B, Q+1)
                theta_b = state.theta_t           # (B, D)
                alpha_t = env.alpha_table         # (Q+1, D)
                beta_t = env.beta_table           # (Q+1, K-1)
                B_env, Q1 = action_mask.shape
                K_B = env.K_B

                all_ids = torch.arange(Q1, device=env.device).unsqueeze(0).expand(B_env, -1)
                log_p = gpcm_log_probs(theta_b, all_ids, alpha_t, beta_t)  # (B, Q+1, K)
                p = log_p.exp()
                entropy = -(p * log_p.clamp_min(-30)).sum(dim=-1)  # (B, Q+1)
                entropy = entropy.masked_fill(~action_mask, -1e9)
                _, top_ids = entropy.topk(K_B, dim=-1)  # (B, K_B)
                action = top_ids.to(dtype=torch.long)

            elif policy is not None:
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

            else:
                # Uniform random under mask.
                prob = state.action_mask.to(torch.float32)
                prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                action = torch.multinomial(prob, num_samples=env.K_B, replacement=False)

            next_state, reward, done, info = env.step(action)
            r = reward.detach().cpu().to(torch.float32) if isinstance(reward, torch.Tensor) else torch.tensor([float(reward)] * env.B)
            ep_return = ep_return + r
            for key in comp_sums.keys():
                val = info.get(key)
                if isinstance(val, torch.Tensor):
                    v = float(val.detach().mean().item())
                    comp_sums[key].append(v)
                    if key == "r_voi":
                        r_voi_all.append(v)
            state = next_state

        for v in ep_return.tolist():
            returns.append(float(v))

    arr = np.asarray(returns, dtype=np.float64)
    n = len(arr)
    se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    ci_low = float(arr.mean() - 1.96 * se)
    ci_high = float(arr.mean() + 1.96 * se)
    r_voi_arr = np.asarray(r_voi_all, dtype=np.float64)
    r_voi_pos_frac = float(np.mean(r_voi_arr > 0)) if r_voi_arr.size > 0 else 0.0

    return PolicyResult(
        name=name,
        mean_return=float(arr.mean()),
        std_return=float(arr.std(ddof=0)),
        ci_low=ci_low,
        ci_high=ci_high,
        r_info=float(np.mean(comp_sums["r_info"])) if comp_sums["r_info"] else 0.0,
        r_cost=float(np.mean(comp_sums["r_cost"])) if comp_sums["r_cost"] else 0.0,
        r_expo=float(np.mean(comp_sums["r_expo"])) if comp_sums["r_expo"] else 0.0,
        r_voi=float(np.mean(comp_sums["r_voi"])) if comp_sums["r_voi"] else 0.0,
        r_voi_positive_frac=r_voi_pos_frac,
        n_episodes=int(arr.size),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_WORKTREE = Path(__file__).parent.parent.parent.resolve()
_REPO = Path("C:/Users/steph/Documents/deep-mirt")
_RL_SCRIPTS = Path(__file__).parent.resolve()

_CONFIGS = {
    "staircase": _WORKTREE / "rl/configs/ppo_synth_e47_stair.yaml",
    "randomwalk": _WORKTREE / "rl/configs/ppo_synth_e47_rw.yaml",
}
_CHECKPOINTS = {
    "staircase": _REPO / "rl/outputs/ordrec_synth_e47_stair/best.pt",
    "randomwalk": _REPO / "rl/outputs/ordrec_synth_e47_rw/best.pt",
}
_BC_CHECKPOINTS = {
    "staircase": _REPO / "rl/outputs/ordrec_synth_e47_stair/bc_warmstart.pt",
    "randomwalk": _REPO / "rl/outputs/ordrec_synth_e47_rw/bc_warmstart.pt",
}
_N_EPISODES = 50  # 50 resets x B=32 = 1600 trajectories per policy; 95% CI < 0.02
_EVAL_SEED = 1234
_RESULTS_DIR = _WORKTREE / "rl/results"


def run_cohort_eval(cohort_name: str) -> Dict[str, Any]:
    cfg_path = _CONFIGS[cohort_name]
    ppo_ckpt = _CHECKPOINTS[cohort_name]
    bc_ckpt = _BC_CHECKPOINTS[cohort_name]

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.get("device"))

    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating cohort: {cohort_name}", flush=True)
    print(f"Config: {cfg_path}", flush=True)
    print(f"{'='*60}", flush=True)

    adapter = _build_adapter(cfg["adapter"])
    world = _build_world_model(cfg.get("world_model", {}), adapter, device)
    env = _build_env(cfg, adapter, world, device)
    # Force test split.
    env.split = "test"
    ppo = None

    results = {}

    # -- Trained PPO --
    print("Running trained PPO...", flush=True)
    ppo_inst = type("PPO_stub", (), {})()  # temp placeholder
    from ordrec.training import PPO
    ppo_obj = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        device=device,
        seed=int(cfg.get("seed", 0)),
        hidden_dim=int(cfg.get("ppo", {}).get("hyperparameters", {}).get("hidden_dim", 128)),
        n_hidden_layers=int(cfg.get("ppo", {}).get("hyperparameters", {}).get("n_hidden_layers", 2)),
    )
    ppo_obj.load(ppo_ckpt)
    env.fleet_expo.zero_()
    results["ppo"] = _run_policy(
        env, policy=ppo_obj, n_episodes=_N_EPISODES * env.B,
        seed=_EVAL_SEED, name="trained PPO",
    )
    print(f"  PPO mean_return={results['ppo'].mean_return:.4f} r_voi={results['ppo'].r_voi:.4f}", flush=True)

    # -- BC-only --
    print("Running BC-only...", flush=True)
    bc_obj = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        device=device,
        seed=int(cfg.get("seed", 0)),
        hidden_dim=128,
        n_hidden_layers=2,
    )
    bc_obj.load(bc_ckpt)
    env.fleet_expo.zero_()
    results["bc"] = _run_policy(
        env, policy=bc_obj, n_episodes=_N_EPISODES * env.B,
        seed=_EVAL_SEED, name="BC-only",
    )
    print(f"  BC mean_return={results['bc'].mean_return:.4f} r_voi={results['bc'].r_voi:.4f}", flush=True)

    # -- Max-Fisher --
    print("Running max-Fisher...", flush=True)
    env.fleet_expo.zero_()
    results["fisher"] = _run_policy(
        env, policy=None, n_episodes=_N_EPISODES * env.B,
        seed=_EVAL_SEED, name="max-Fisher", use_fisher=True,
    )
    print(f"  Fisher mean_return={results['fisher'].mean_return:.4f} r_voi={results['fisher'].r_voi:.4f}", flush=True)

    # -- Uniform random --
    print("Running uniform random...", flush=True)
    env.fleet_expo.zero_()
    results["random"] = _run_policy(
        env, policy=None, n_episodes=_N_EPISODES * env.B,
        seed=_EVAL_SEED, name="uniform random",
    )
    print(f"  Random mean_return={results['random'].mean_return:.4f} r_voi={results['random'].r_voi:.4f}", flush=True)

    return {k: asdict(v) for k, v in results.items()}


def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for cohort in ["staircase", "randomwalk"]:
        all_results[cohort] = run_cohort_eval(cohort)

    # Save JSON.
    out_json = _RESULTS_DIR / "E47_eval.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved to {out_json}", flush=True)

    # Print summary.
    print("\n" + "="*60)
    print("E4.7 FOUR-POLICY EVALUATION SUMMARY")
    print("="*60)
    for cohort, res in all_results.items():
        print(f"\nCohort: {cohort.upper()}")
        print(f"{'policy':<16} {'mean_return':>12} {'95% CI':>22} {'r_info':>8} {'r_voi':>8} {'voi_pos%':>9}")
        for key in ["ppo", "bc", "fisher", "random"]:
            r = res[key]
            print(
                f"{r['name']:<16} {r['mean_return']:>+12.4f} "
                f"({r['ci_low']:>+.4f},{r['ci_high']:>+.4f}) "
                f"{r['r_info']:>+8.4f} {r['r_voi']:>+8.4f} {r['r_voi_positive_frac']:>9.1%}"
            )

    print(f"\nResults JSON: {out_json}", flush=True)


if __name__ == "__main__":
    main()
