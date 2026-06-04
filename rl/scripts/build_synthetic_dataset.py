"""CLI orchestrator for the v1 synthetic dataset (M3).

Reads a YAML preset, draws ``N`` synthetic users + a mixed 2PL/GPCM
item bank, attaches the O*NET-derived ``delta_j`` per job, calibrates
the Option A preference model, samples engagement classes, candidate
sets, and binary likes, then writes the six output files defined in
Section 5.4 of the v1 plan.

Usage::

    PYTHONPATH=rl/src python rl/scripts/build_synthetic_dataset.py \\
        --config rl/configs/sim_v1_dev.yaml

The output directory in the YAML can be overridden with
``--output-dir``. All files are written with deterministic key
orderings so two runs at the same seed produce byte-identical outputs
(the M3 reproducibility sanity check).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# Make ``irtrec`` importable when run from the repo root without an
# install step. The package layout is rl/src/irtrec.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "rl" / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from irtrec.datagen.onet_pool_attach import OnetPool, load_onet_pool  # noqa: E402
from irtrec.datagen.synth_likes import (  # noqa: E402
    LikesGenSpec,
    assign_engagement,
    build_likes_table,
    sample_candidate_jobs,
    sample_candidate_set_sizes,
)
from irtrec.datagen.synth_users import (  # noqa: E402
    DEFAULT_BANK_COMPOSITION,
    ItemBank,
    build_item_bank,
    sample_responses,
    sample_users,
)


SEED_LABELS: Tuple[str, ...] = (
    "users_theta",
    "item_bank",
    "responses",
    "engagement",
    "candidate_sizes",
    "candidate_jobs",
    "likes_bernoulli",
)


def _spawn_seeds(master_seed: int) -> Dict[str, int]:
    """Derive deterministic sub-seeds for each draw category.

    Using ``SeedSequence`` keeps the master seed as the single knob
    while giving each RNG a separately-seeded stream. The integer seeds
    are taken as the high 32 bits of the spawned key state, which is
    stable across numpy versions.
    """
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(len(SEED_LABELS))
    return {
        label: int(child.generate_state(1, dtype=np.uint32)[0])
        for label, child in zip(SEED_LABELS, children)
    }


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _composition_from_config(cfg: dict) -> Tuple[Tuple[int, int], ...]:
    raw = cfg.get("user_bank", {}).get("composition")
    if raw is None:
        return tuple((int(K), int(c)) for K, c in DEFAULT_BANK_COMPOSITION)
    out = []
    for K, c in raw:
        out.append((int(K), int(c)))
    return tuple(out)


def _likes_spec_from_config(cfg: dict) -> LikesGenSpec:
    pref = cfg.get("preference", {})
    cand = cfg.get("candidate_set", {})
    return LikesGenSpec(
        target_like_rate=float(pref.get("target_like_rate", 0.20)),
        rejecter_share=float(pref.get("rejecter_share", 0.40)),
        engaged_share=float(pref.get("engaged_share", 0.60)),
        lambda_range=tuple(pref.get("lambda_range", [0.5, 3.0])),  # type: ignore[arg-type]
        bias_range=tuple(pref.get("bias_range", [-2.0, 2.0])),  # type: ignore[arg-type]
        negbinom_r=float(cand.get("negbinom_r", 2.0)),
        negbinom_p=float(cand.get("negbinom_p", 0.05)),
        negbinom_clip=tuple(cand.get("negbinom_clip", [1, 200])),  # type: ignore[arg-type]
    )


def _dump_json(path: Path, obj) -> None:
    """Write deterministic JSON (sorted keys, fixed separators)."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"), sort_keys=False)


def _dump_pretty_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)


def build_dataset(config_path: Path, output_dir_override: Path | None = None) -> Path:
    """Run the full M3 pipeline for one preset.

    Returns the output directory that was written.
    """
    cfg = _load_config(config_path)
    master_seed = int(cfg["seed"])
    n_users = int(cfg["n_users"])
    output_dir = Path(output_dir_override or cfg["output_dir"]).resolve()
    parquet_path = Path(cfg.get("onet_parquet", "rl/artifacts/onet_v1.parquet"))
    if not parquet_path.is_absolute():
        parquet_path = (_REPO_ROOT / parquet_path).resolve()

    seeds = _spawn_seeds(master_seed)

    # ------------------------------------------------------------------
    # Users + item bank + questionnaire responses.
    # ------------------------------------------------------------------
    theta = sample_users(n_users=n_users, seed=seeds["users_theta"])
    bank_cfg = cfg.get("user_bank", {})
    bank = build_item_bank(
        composition=_composition_from_config(cfg),
        alpha_sigma=float(bank_cfg.get("alpha_sigma", 0.4)),
        bmean_sigma=float(bank_cfg.get("bmean_sigma", 0.7)),
        threshold_sigma=float(bank_cfg.get("threshold_sigma", 0.5)),
        seed=seeds["item_bank"],
    )
    sequences = sample_responses(theta=theta, bank=bank, seed=seeds["responses"])

    # ------------------------------------------------------------------
    # O*NET pool + delta_j.
    # ------------------------------------------------------------------
    pool: OnetPool = load_onet_pool(parquet_path)

    # ------------------------------------------------------------------
    # Engagement, candidate sets, likes.
    # ------------------------------------------------------------------
    spec = _likes_spec_from_config(cfg)
    engagement = assign_engagement(
        n_users=n_users,
        rejecter_share=spec.rejecter_share,
        seed=seeds["engagement"],
    )
    candidate_sizes = sample_candidate_set_sizes(
        n_users=n_users,
        spec=spec,
        seed=seeds["candidate_sizes"],
    )
    candidate_sets = sample_candidate_jobs(
        n_jobs=pool.n_jobs,
        candidate_sizes=candidate_sizes,
        seed=seeds["candidate_jobs"],
    )
    likes_records, lam, bias = build_likes_table(
        theta=theta,
        engagement=engagement,
        candidate_sets=candidate_sets,
        delta_j=pool.delta_j,
        spec=spec,
        seed=seeds["likes_bernoulli"],
    )

    # ------------------------------------------------------------------
    # Compose output files.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    _dump_json(output_dir / "sequences.json", sequences)

    jobs_records = pool.to_jobs_records()
    _dump_pretty_json(output_dir / "jobs.json", jobs_records)

    _dump_json(output_dir / "likes.json", likes_records)

    irt_params = {
        "theta": theta.tolist(),
        "alpha": bank.alpha.tolist(),
        "beta": bank.beta_as_list(),
        "n_categories": bank.n_categories.tolist(),
        "n_students": int(n_users),
        "n_questions": int(bank.n_items),
        "theta_stats": {
            "mean": float(theta.mean()),
            "std": float(theta.std(ddof=0)),
        },
        "alpha_stats": {
            "mean": float(bank.alpha.mean()),
            "std": float(bank.alpha.std(ddof=0)),
        },
    }
    _dump_pretty_json(output_dir / "true_irt_parameters.json", irt_params)

    pref_params = {
        "delta_j": pool.delta_j.tolist(),
        "lambda": float(lam),
        "bias": float(bias),
        "engagement_class": engagement.tolist(),
        "candidate_set_sizes": candidate_sizes.tolist(),
        "delta_source": cfg.get("preference", {}).get("delta_source", "work_zone"),
    }
    _dump_pretty_json(output_dir / "true_preference_parameters.json", pref_params)

    overall_like_rate = (
        float(np.mean([r["IsLiked"] for r in likes_records]))
        if likes_records
        else 0.0
    )
    engaged_count = int(np.sum(engagement == 1))
    rejecter_count = int(np.sum(engagement == 0))

    metadata = {
        "schema_flag": cfg.get("schema_flag", "drlmairt_sim_v1"),
        "preset_name": cfg.get("preset_name", config_path.stem),
        "seed": master_seed,
        "n_users": n_users,
        "n_questions": int(bank.n_items),
        "n_jobs": int(pool.n_jobs),
        "k_distribution": bank.k_distribution,
        "engagement": {
            "rejecter_share_target": spec.rejecter_share,
            "engaged_share_target": spec.engaged_share,
            "rejecter_count": rejecter_count,
            "engaged_count": engaged_count,
        },
        "preference": {
            "target_like_rate": spec.target_like_rate,
            "lambda": float(lam),
            "bias": float(bias),
            "achieved_overall_like_rate": overall_like_rate,
            "delta_source": pref_params["delta_source"],
        },
        "candidate_set": {
            "negbinom_r": spec.negbinom_r,
            "negbinom_p": spec.negbinom_p,
            "negbinom_clip": list(spec.negbinom_clip),
            "median_size": int(np.median(candidate_sizes)),
            "mean_size": float(candidate_sizes.mean()),
        },
        "sub_seeds": seeds,
        "config_path": str(config_path),
    }
    _dump_pretty_json(output_dir / "metadata.json", metadata)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a sim_v1 YAML config (dev or recovery).",
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
    print(f"[sim_v1] wrote dataset -> {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
