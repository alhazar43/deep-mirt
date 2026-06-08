"""Top-level PPO training script.

Reads a YAML config and runs the full training pipeline.

1. Frozen MA-IRT world model (random-init or checkpoint).
2. Adapter (synthetic, eedi, etc.) materialise + load.
3. Per-item ``(alpha, beta)`` cache.
4. ``OrdRecEnv`` + ``OrdinalRewardCompute``.
5. ``PPO`` algorithm.
6. Optional BC + static-MVE warm-start.
7. PPO loop, saves best by mean episode return to
   ``outputs/<experiment>/best.pt`` and the running metrics to
   ``outputs/<experiment>/metrics.csv``.

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \\
        python rl/scripts/train_ppo.py --config rl/configs/ppo_synth_smoke.yaml

See ``docs/ordrec_impl_guide.md`` Section 5.2 for the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from ordrec.bc_warmstart import bc_warmstart, mve_warmstart_critic
from ordrec.data import (
    AdapterConfig,
    EediAdapter,
    SyntheticAdapter,
)
from ordrec.envs import FrozenMAGPCM, OrdRecEnv, build_item_cache
from ordrec.reward import OrdinalRewardCompute, RewardConfig
from ordrec.training import PPO, pick_device, set_seed


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config at {path} must be a YAML mapping, got {type(cfg).__name__}"
        )
    return cfg


_ADAPTERS = {
    "synthetic": SyntheticAdapter,
    "eedi": EediAdapter,
}


def _build_adapter(adapter_cfg: Dict[str, Any]) -> Any:
    adapter_type = adapter_cfg.get("type", "synthetic")
    if adapter_type not in _ADAPTERS:
        raise KeyError(
            f"Unknown adapter type {adapter_type!r}. "
            f"Available: {sorted(_ADAPTERS)}"
        )
    AdapterCls = _ADAPTERS[adapter_type]
    raw_dir = Path(adapter_cfg["raw_dir"])
    out_dir = Path(adapter_cfg["out_dir"])
    cfg = AdapterConfig(
        name=adapter_cfg["name"],
        raw_dir=raw_dir,
        out_dir=out_dir,
        split_seed=int(adapter_cfg.get("split_seed", 0)),
        test_frac=float(adapter_cfg.get("test_frac", 0.1)),
        valid_frac=float(adapter_cfg.get("valid_frac", 0.1)),
        min_seq_len=int(adapter_cfg.get("min_seq_len", 5)),
        max_seq_len=int(adapter_cfg.get("max_seq_len", 0)),
        chunk_long_sequences=bool(adapter_cfg.get("chunk_long_sequences", False)),
    )
    adapter = AdapterCls(cfg)
    artefact = adapter.artefact_dir
    if not (artefact / "sequences.json").exists():
        adapter.materialise()
    adapter.load()
    return adapter


def _build_world_model(
    world_cfg: Dict[str, Any], adapter: Any, device: torch.device,
) -> FrozenMAGPCM:
    # Lazy import so the script's --help works without ma-irt on PYTHONPATH.
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    n_questions = int(world_cfg.get("n_questions", adapter.get_n_questions()))
    n_categories = int(world_cfg.get("n_categories", adapter.get_n_categories()))
    model = MAGPCM(
        n_questions=n_questions,
        n_categories=n_categories,
        n_traits=int(world_cfg.get("n_traits", 1)),
        memory_size=int(world_cfg.get("memory_size", 8)),
        key_dim=int(world_cfg.get("key_dim", 16)),
        value_dim=int(world_cfg.get("value_dim", 16)),
        summary_dim=int(world_cfg.get("summary_dim", 16)),
        embedding_type=world_cfg.get("embedding_type", "learned"),
        dropout_rate=float(world_cfg.get("dropout_rate", 0.0)),
        ability_scale=float(world_cfg.get("ability_scale", 1.0)),
        separate_theta=bool(world_cfg.get("separate_theta", True)),
        init_value_memory=bool(world_cfg.get("init_value_memory", False)),
    ).to(device)

    ckpt_path = world_cfg.get("checkpoint")
    if ckpt_path:
        ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    return FrozenMAGPCM(model, n_questions=n_questions, n_categories=n_categories)


def _build_env(
    cfg: Dict[str, Any], adapter: Any, world: FrozenMAGPCM, device: torch.device,
) -> OrdRecEnv:
    env_cfg = cfg.get("env", {})
    reward_cfg_dict = cfg.get("reward", {})
    reward_cfg = RewardConfig(
        w_info=float(reward_cfg_dict.get("w_info", 1.0)),
        w_cost=float(reward_cfg_dict.get("w_cost", 0.05)),
        w_expo=float(reward_cfg_dict.get("w_expo", 0.10)),
        w_voi=float(reward_cfg_dict.get("w_voi", 5.0)),
        probe_M=int(reward_cfg_dict.get("probe_M", 32)),
        probe_H=int(reward_cfg_dict.get("probe_H", 20)),
        r_max=float(reward_cfg_dict.get("r_max", 0.20)),
        c_expo=float(reward_cfg_dict.get("c_expo", 1.0)),
        expo_ema_decay=float(reward_cfg_dict.get("expo_ema_decay", 0.99)),
        K_B=int(env_cfg.get("K_B", reward_cfg_dict.get("K_B", 5))),
        T=int(env_cfg.get("T", reward_cfg_dict.get("T", 10))),
        n_difficulty_strata=int(
            reward_cfg_dict.get("n_difficulty_strata", 5)
        ),
        eps_prob=float(reward_cfg_dict.get("eps_prob", 1e-12)),
        prior_precision_jitter=float(
            reward_cfg_dict.get("prior_precision_jitter", 1e-6)
        ),
        running_norm_freeze_after=int(
            reward_cfg_dict.get("running_norm_freeze_after", 1000)
        ),
    )
    n_categories = adapter.get_n_categories()
    cache = build_item_cache(
        world,
        n_contexts=int(env_cfg.get("cache_n_contexts", 2)),
        context_seq_len=int(env_cfg.get("cache_context_seq_len", 1)),
        dataset_name=cfg["experiment_name"],
    )
    reward = OrdinalRewardCompute(reward_cfg, n_categories=n_categories)
    env = OrdRecEnv(
        world_model=world,
        adapter=adapter,
        item_cache=cache,
        reward_fn=reward,
        cfg=reward_cfg,
        batch_size=int(env_cfg.get("batch_size", 4)),
        warmup_len=int(env_cfg.get("warmup_len", 3)),
        split=str(env_cfg.get("split", "train")),
        device=device,
        seed=int(cfg.get("seed", 0)),
    )
    return env


def _build_ppo(
    cfg: Dict[str, Any], env: OrdRecEnv, device: torch.device,
) -> PPO:
    ppo_cfg = cfg.get("ppo", {})
    hyper = ppo_cfg.get("hyperparameters", {})
    return PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        device=device,
        seed=int(cfg.get("seed", 0)),
        hidden_dim=int(hyper.get("hidden_dim", 128)),
        n_hidden_layers=int(hyper.get("n_hidden_layers", 2)),
        learning_rate=float(hyper.get("learning_rate", 3e-4)),
        adam_eps=float(hyper.get("adam_eps", 1e-5)),
        clip_eps=float(hyper.get("clip_eps", 0.2)),
        gamma=float(hyper.get("gamma", 0.95)),
        gae_lambda=float(hyper.get("gae_lambda", 0.95)),
        entropy_coef_initial=float(hyper.get("entropy_coef_initial", 0.01)),
        entropy_coef_final=float(hyper.get("entropy_coef_final", 0.0)),
        entropy_anneal_fraction=float(
            hyper.get("entropy_anneal_fraction", 0.5)
        ),
        value_coef=float(hyper.get("value_coef", 0.5)),
        max_grad_norm=float(hyper.get("max_grad_norm", 0.5)),
        n_epochs=int(hyper.get("n_epochs", 4)),
        minibatch_size=int(hyper.get("minibatch_size", 32)),
        target_kl=float(hyper.get("target_kl", 0.02)),
        n_episodes_per_update=int(ppo_cfg.get("n_episodes_per_update", 32)),
        max_steps_per_episode=int(
            ppo_cfg.get("max_steps_per_episode", env.horizon_steps)
        ),
        total_updates=int(ppo_cfg.get("total_updates", 1000)),
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _log_step(
    csv_path: Path,
    row: Dict[str, Any],
) -> None:
    fields = list(row.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: Dict[str, Any], *, config_path: Optional[Path] = None) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = pick_device(cfg.get("device"))
    out_dir = Path(cfg.get("output_dir", "outputs")) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()
    best_path = out_dir / "best.pt"

    adapter = _build_adapter(cfg["adapter"])
    world = _build_world_model(cfg.get("world_model", {}), adapter, device)
    env = _build_env(cfg, adapter, world, device)
    ppo = _build_ppo(cfg, env, device)

    warmstart_cfg = cfg.get("warmstart", {})
    if warmstart_cfg.get("enabled", False):
        bc_n = int(warmstart_cfg.get("bc_updates", 5))
        mve_n = int(warmstart_cfg.get("mve_updates", 5))
        if bc_n > 0:
            bc_warmstart(
                ppo, env, n_updates=bc_n,
                n_episodes_per_update=int(
                    warmstart_cfg.get("bc_episodes_per_update", 2)
                ),
                seed=seed,
            )
        if mve_n > 0:
            mve_warmstart_critic(
                ppo, env, n_updates=mve_n,
                n_episodes_per_update=int(
                    warmstart_cfg.get("mve_episodes_per_update", 2)
                ),
                seed=seed,
            )

    total_updates = ppo.total_updates
    best_return = float("-inf")
    summary = {
        "experiment_name": cfg["experiment_name"],
        "config_path": str(config_path) if config_path is not None else None,
        "device": str(device),
        "updates": [],
    }
    t_start = time.time()
    for it in range(total_updates):
        rstats = ppo.rollout(env, n_episodes=ppo.n_episodes_per_update)
        ustats = ppo.update()
        elapsed = time.time() - t_start
        row = {
            "update": it,
            "mean_return": rstats.mean_episode_return,
            "mean_length": rstats.mean_episode_length,
            "policy_loss": ustats.policy_loss,
            "value_loss": ustats.value_loss,
            "entropy": ustats.entropy,
            "approx_kl": ustats.approx_kl,
            "clipfrac": ustats.clipfrac,
            "n_grad_steps": ustats.n_grad_steps,
            "early_stop": ustats.extras.get("early_stop", False),
            "entropy_coef": ustats.extras.get("entropy_coef", 0.0),
            "r_info": rstats.info.get("r_info", 0.0),
            "r_cost": rstats.info.get("r_cost", 0.0),
            "r_expo": rstats.info.get("r_expo", 0.0),
            "r_voi": rstats.info.get("r_voi", 0.0),
            "elapsed_s": elapsed,
        }
        _log_step(metrics_csv, row)
        summary["updates"].append(row)
        print(
            f"update={it:4d} mean_return={rstats.mean_episode_return:+.3f} "
            f"policy_loss={ustats.policy_loss:+.3f} "
            f"value_loss={ustats.value_loss:.3f} "
            f"entropy={ustats.entropy:.3f} "
            f"approx_kl={ustats.approx_kl:.4f} "
            f"r_info={rstats.info.get('r_info', 0.0):+.3f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        if rstats.mean_episode_return > best_return:
            best_return = rstats.mean_episode_return
            ppo.save(best_path)
    ppo.save(out_dir / "last.pt")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in summary.items()}, fh, indent=2, default=str)
    print(
        f"\nTraining complete. best_return={best_return:.3f} "
        f"checkpoint={best_path}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PPO on OrdRec.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _load_config(args.config)
    train(cfg, config_path=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
