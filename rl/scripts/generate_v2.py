"""CLI orchestrator for the DRL-MAIRT v2 synthetic dataset (M4-RL).

Reads a sim_v2 YAML preset, attaches the O*NET-derived continuous
``delta_j`` per job (Section M4-RL Step 1), samples N users with
(theta, lambda_u) priors (Step 3), samples per-user candidate sets
(clipped NegBinom), emits K=5 GPCM ordinal responses (Step 4) and
writes the v2 output bundle to disk.

Output files (under ``output_dir``)

    users.json              per-user latents and split assignment
    jobs.json               long-form job records with continuous delta_j
    responses.json          long-form (user_id, job_id, y, IsLiked, prob)
    splits.json             user-ID partitions
    oracle_metadata.json    aggregated diagnostics for downstream tools

Determinism: every random draw is seeded from explicit sub-seeds spawned
from the master ``seed`` and the per-stage seeds in the YAML, so two
runs of the same preset produce byte-identical outputs.

Usage::

    PYTHONPATH=rl/src python rl/scripts/generate_v2.py \\
        --config rl/configs/sim_v2_dev.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# Make ``irtrec`` importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "rl" / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from irtrec.datagen.onet_pool_attach import OnetPool, load_onet_pool  # noqa: E402
from irtrec.datagen.synth_likes import (  # noqa: E402
    GPCM_BETA,
    GPCM_K,
    GPCM_LIKED_CATEGORY,
    build_likes_table_v2,
)
from irtrec.datagen.synth_users import (  # noqa: E402
    LAMBDA_U_LOG_MEAN,
    LAMBDA_U_LOG_SIGMA,
    sample_users_v2,
    stratified_split,
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_compact_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"), sort_keys=False)


def _dump_pretty_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)


def _sample_candidate_set_sizes(
    *, n_users: int, r: float, p: float, clip: Tuple[int, int], seed: int
) -> np.ndarray:
    """Clipped NegBinom per-user candidate set sizes (v2 parity with v1)."""
    rng = np.random.default_rng(seed)
    draws = rng.negative_binomial(r, p, size=n_users)
    lo, hi = clip
    return np.clip(draws, lo, hi).astype(np.int64)


def _sample_candidate_jobs(
    *, n_jobs: int, candidate_sizes: np.ndarray, seed: int
) -> List[np.ndarray]:
    """Uniform without replacement per user (v2 parity with v1)."""
    rng = np.random.default_rng(seed)
    all_jobs = np.arange(n_jobs, dtype=np.int64)
    out: List[np.ndarray] = []
    for n_u in candidate_sizes.tolist():
        if n_u >= n_jobs:
            sample = rng.choice(all_jobs, size=n_jobs, replace=False)
        else:
            sample = rng.choice(all_jobs, size=int(n_u), replace=False)
        out.append(np.sort(sample.astype(np.int64)))
    return out


def _spawn_seeds(master_seed: int, labels: Tuple[str, ...]) -> Dict[str, int]:
    """Derive deterministic sub-seeds from a single master seed."""
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(len(labels))
    return {
        label: int(child.generate_state(1, dtype=np.uint32)[0])
        for label, child in zip(labels, children)
    }


_SEED_LABELS: Tuple[str, ...] = (
    "candidate_sizes",
    "candidate_jobs",
    "responses",
)


def build_dataset(config_path: Path, output_dir_override: Path | None = None) -> Path:
    """Run the full v2 pipeline for one preset.

    Returns the output directory that was written.
    """
    cfg = _load_config(config_path)
    master_seed = int(cfg["seed"])
    n_users = int(cfg["n_users"])
    output_dir = Path(output_dir_override or cfg["output_dir"]).resolve()

    parquet_path = Path(cfg.get("onet_parquet", "rl/artifacts/onet_v1.parquet"))
    if not parquet_path.is_absolute():
        parquet_path = (_REPO_ROOT / parquet_path).resolve()

    # ------------------------------------------------------------------
    # Seeds.
    # ------------------------------------------------------------------
    sub_seeds = _spawn_seeds(master_seed, _SEED_LABELS)
    users_cfg = cfg.get("users", {})
    theta_seed = int(users_cfg.get("theta_seed", 7))
    lambda_seed = int(users_cfg.get("lambda_seed", 11))

    # ------------------------------------------------------------------
    # Pool + continuous delta_j.
    # ------------------------------------------------------------------
    pool: OnetPool = load_onet_pool(parquet_path)

    # ------------------------------------------------------------------
    # User latents + split.
    # ------------------------------------------------------------------
    theta, lambda_u = sample_users_v2(
        n_users=n_users, theta_seed=theta_seed, lambda_seed=lambda_seed
    )
    split_cfg = cfg.get("split", {})
    splits = stratified_split(
        n_users=n_users,
        train_frac=float(split_cfg.get("train_frac", 0.80)),
        val_frac=float(split_cfg.get("val_frac", 0.10)),
        test_frac=float(split_cfg.get("test_frac", 0.10)),
        seed=int(split_cfg.get("split_seed", 13)),
    )

    # ------------------------------------------------------------------
    # Candidate sets + GPCM responses.
    # ------------------------------------------------------------------
    cand_cfg = cfg.get("candidate_set", {})
    clip = tuple(cand_cfg.get("negbinom_clip", [1, 200]))
    candidate_sizes = _sample_candidate_set_sizes(
        n_users=n_users,
        r=float(cand_cfg.get("negbinom_r", 2.0)),
        p=float(cand_cfg.get("negbinom_p", 0.05)),
        clip=clip,
        seed=sub_seeds["candidate_sizes"],
    )
    candidate_sets = _sample_candidate_jobs(
        n_jobs=pool.n_jobs,
        candidate_sizes=candidate_sizes,
        seed=sub_seeds["candidate_jobs"],
    )

    gpcm_cfg = cfg.get("gpcm", {})
    beta = tuple(float(b) for b in gpcm_cfg.get("beta", GPCM_BETA))
    liked_cat = int(gpcm_cfg.get("liked_category", GPCM_LIKED_CATEGORY))
    if liked_cat != GPCM_LIKED_CATEGORY:
        # If the user overrides, build_likes_table_v2 still uses the module
        # default. Warn loudly so the contract is visible.
        print(
            f"[generate_v2] warning: liked_category override {liked_cat} is "
            f"recorded in metadata but build_likes_table_v2 hardcodes "
            f"{GPCM_LIKED_CATEGORY}. To change, edit synth_likes.GPCM_LIKED_CATEGORY.",
            flush=True,
        )

    responses = build_likes_table_v2(
        theta=theta,
        lambda_u=lambda_u,
        candidate_sets=candidate_sets,
        delta_j=pool.delta_j,
        beta=beta,
        seed=sub_seeds["responses"],
    )

    # ------------------------------------------------------------------
    # Writeouts.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # users.json: keep a flat per-user record list so downstream loaders
    # can build a DataFrame in one line.
    split_lookup = np.empty(n_users, dtype=object)
    for partition_name, ids in splits.items():
        for uid in ids.tolist():
            split_lookup[uid] = partition_name
    users_records = [
        {
            "user_id": int(uid),
            "theta": float(theta[uid]),
            "lambda_u": float(lambda_u[uid]),
            "candidate_set_size": int(candidate_sizes[uid]),
            "split": str(split_lookup[uid]),
        }
        for uid in range(n_users)
    ]
    _dump_compact_json(output_dir / "users.json", users_records)

    jobs_records = pool.to_jobs_records()
    _dump_pretty_json(output_dir / "jobs.json", jobs_records)

    _dump_compact_json(output_dir / "responses.json", responses)

    splits_obj = {name: ids.tolist() for name, ids in splits.items()}
    _dump_compact_json(output_dir / "splits.json", splits_obj)

    # Aggregated diagnostics.
    y_arr = np.asarray([r["y"] for r in responses], dtype=np.int64)
    isliked_arr = np.asarray([r["IsLiked"] for r in responses], dtype=np.int64)
    prob_arr = np.asarray([r["prob"] for r in responses], dtype=np.float64)
    cat_counts = np.bincount(y_arr, minlength=GPCM_K).tolist() if y_arr.size else [0] * GPCM_K
    oracle_meta = {
        "schema_flag": cfg.get("schema_flag", "drlmairt_sim_v2"),
        "preset_name": cfg.get("preset_name", config_path.stem),
        "master_seed": master_seed,
        "sub_seeds": sub_seeds,
        "n_users": n_users,
        "n_jobs": pool.n_jobs,
        "n_responses": int(y_arr.size),
        "K": GPCM_K,
        "gpcm_beta": list(beta),
        "gpcm_liked_category": liked_cat,
        "user_prior": {
            "theta_seed": theta_seed,
            "lambda_seed": lambda_seed,
            "lambda_log_mean": LAMBDA_U_LOG_MEAN,
            "lambda_log_sigma": LAMBDA_U_LOG_SIGMA,
        },
        "candidate_set": {
            "negbinom_r": float(cand_cfg.get("negbinom_r", 2.0)),
            "negbinom_p": float(cand_cfg.get("negbinom_p", 0.05)),
            "negbinom_clip": list(clip),
            "mean_size": float(candidate_sizes.mean()),
            "median_size": int(np.median(candidate_sizes)),
        },
        "split_counts": {name: int(ids.size) for name, ids in splits.items()},
        "delta_j_stats": {
            "mean": float(pool.delta_j.mean()),
            "std": float(pool.delta_j.std()),
            "min": float(pool.delta_j.min()),
            "max": float(pool.delta_j.max()),
            "n_unique": int(np.unique(pool.delta_j).size),
        },
        "response_stats": {
            "category_counts": cat_counts,
            "is_liked_rate": float(isliked_arr.mean()) if isliked_arr.size else 0.0,
            "mean_oracle_prob": float(prob_arr.mean()) if prob_arr.size else 0.0,
        },
        "config_path": str(config_path),
    }
    _dump_pretty_json(output_dir / "oracle_metadata.json", oracle_meta)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a sim_v2 YAML config (dev or 100k).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the config's output_dir. Useful for ad-hoc runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = build_dataset(args.config, args.output_dir)
    print(f"[sim_v2] wrote dataset -> {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
