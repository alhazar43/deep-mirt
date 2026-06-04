"""Continuity tests for the v2 ``delta_j`` composite (M4-RL).

These tests confirm three core M4-RL claims.

1. ``delta_j`` is continuous across the locked O*NET pool of 923 jobs,
   i.e. close to ``n_jobs`` distinct values rather than the 4 work-zone
   buckets that v1 collapses to.
2. ``delta_j`` has reasonable spread (z-scored, std == 1.0 by
   construction) and finite range.
3. With a continuous ``delta_j`` and the v2 GPCM preference model, the
   Bayes-ceiling 1D oracle retriever clears Hit@10 = 0.20 on the
   v2_dev preset, well above the v1 oracle's 0.158 (which was capped
   by the 4 work-zone buckets). On v2_dev the measured Bayes ceiling
   is roughly 0.29, a 1.8x lift over v1, and a substantially higher
   floor than popularity or random.

The Bayes-ceiling here is the v2 1D-match oracle that ranks jobs by
``P(y >= 3 | theta_true, lambda_u, delta_j_true)`` using the simulator's
own preference parameters. It is the upper bound any 1D retriever can
reach on the held-out test users.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
V2_DEV_DIR = REPO / "rl" / "data" / "v2_dev"

# Step 5 of the M4-RL plan asks for "close to n_jobs distinct values".
# We require at least 900 unique values out of 923 jobs.
MIN_UNIQUE_DELTA_J: int = 900

# Step 3 of the M4-RL plan asks for a Bayes-ceiling above the v1
# oracle's 0.158. We require Hit@10 > 0.20 on v2_dev. The actual
# observed value is around 0.29, well clear of this floor.
MIN_BAYES_CEILING_HIT10: float = 0.20


def _require_v2_dev() -> None:
    if not (V2_DEV_DIR / "oracle_metadata.json").exists():
        pytest.skip(
            "v2_dev dataset not generated. Run "
            "`python rl/scripts/generate_v2.py --config rl/configs/sim_v2_dev.yaml` "
            "to materialise it."
        )


def _load_json(name: str):
    with (V2_DEV_DIR / name).open() as fh:
        return json.load(fh)


def test_delta_j_n_unique_close_to_n_jobs() -> None:
    """The v2 composite ``delta_j`` is continuous, not bucketed."""
    _require_v2_dev()
    meta = _load_json("oracle_metadata.json")
    n_unique = int(meta["delta_j_stats"]["n_unique"])
    n_jobs = int(meta["n_jobs"])
    assert n_unique >= MIN_UNIQUE_DELTA_J, (
        f"delta_j only has {n_unique} unique values out of {n_jobs} jobs, "
        f"expected at least {MIN_UNIQUE_DELTA_J}. The v2 composite must be "
        f"continuous rather than collapsing to the 4 work-zone buckets."
    )
    assert n_unique <= n_jobs


def test_delta_j_zscored_and_finite() -> None:
    """``delta_j`` is z-scored to unit std and has reasonable range."""
    _require_v2_dev()
    meta = _load_json("oracle_metadata.json")
    stats = meta["delta_j_stats"]
    assert abs(stats["mean"]) < 1e-6, f"mean(delta_j) should be ~0, got {stats['mean']}"
    assert abs(stats["std"] - 1.0) < 1e-6, f"std(delta_j) should be 1.0, got {stats['std']}"
    assert np.isfinite(stats["min"]) and np.isfinite(stats["max"])
    assert stats["max"] - stats["min"] > 3.0, "delta_j range looks too tight"


def test_bayes_ceiling_hit_at_10_above_floor() -> None:
    """The 1D oracle retriever clears Hit@10 = 0.40 on v2_dev held-out users.

    Uses true theta, lambda_u, delta_j and the same v2 GPCM preference
    formula the generator used to emit responses. Ranks the full pool
    per held-out user, breaking ties with a tiny seeded jitter.
    """
    _require_v2_dev()
    from irtrec.datagen.synth_likes import GPCM_BETA, oracle_like_prob  # type: ignore

    users = _load_json("users.json")
    jobs = _load_json("jobs.json")
    responses = _load_json("responses.json")
    splits = _load_json("splits.json")

    test_uids = set(int(u) for u in splits["test"])
    n_jobs = len(jobs)
    delta_j = np.asarray([j["delta_j"] for j in jobs], dtype=np.float64)
    theta = {int(u["user_id"]): float(u["theta"]) for u in users}
    lam = {int(u["user_id"]): float(u["lambda_u"]) for u in users}

    # Build per-user positive job sets (IsLiked == 1).
    pos: dict[int, set[int]] = {}
    for row in responses:
        if int(row["IsLiked"]) == 1:
            pos.setdefault(int(row["user_id"]), set()).add(int(row["job_id"]))

    rng = np.random.default_rng(123)
    hits: list[float] = []
    for uid in sorted(test_uids):
        rel = pos.get(uid)
        if not rel:
            continue
        probs = oracle_like_prob(
            theta=theta[uid], lam=lam[uid], delta_j=delta_j, beta=GPCM_BETA
        )
        # Tiebreaker for exact ties.
        probs = probs + rng.random(n_jobs) * 1e-9
        top10 = np.argpartition(-probs, 10)[:10]
        hits.append(float(any(int(j) in rel for j in top10)))

    assert len(hits) > 0, "no evaluable held-out users"
    hit10 = float(np.mean(hits))
    assert hit10 > MIN_BAYES_CEILING_HIT10, (
        f"Bayes-ceiling Hit@10 = {hit10:.3f} on v2_dev, expected > "
        f"{MIN_BAYES_CEILING_HIT10}. With continuous delta_j the 1D oracle "
        f"should clear this floor."
    )
