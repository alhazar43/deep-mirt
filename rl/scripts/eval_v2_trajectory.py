"""Within-session recommendation trajectory on the v2 dev simulator.

CaRReL produced a "relevance over time" curve, Hit-style relevance
climbing as more user history is observed. Our M4-RL preliminary only
measured terminal Hit@10. This script computes the within-session
trajectory analog on v2 dev data, item by item.

Protocol

  Same 80/20 user split as ``eval_v2_baselines.py`` (split seed = 0,
  test users with at least one IsLiked positive). For each held-out
  test user u with T_u responses, walk through their response sequence
  in the order it appears in ``responses.json``. At each step
  t = 1, ..., T_u

    1. Incrementally update the log-posterior of theta on a 91-point
       grid in [-4.5, 4.5] with a unit Gaussian prior, conditioning on
       the first t responses. The EAP estimator matches
       ``eval_v2_baselines.eap_theta_v2`` exactly.
    2. Score every job j by P(y >= 3 | theta_hat_t, lambda_u, delta_j)
       under the true GPCM, same scoring rule as the terminal eval.
    3. Hit@10(u, t) = 1[any IsLiked = 1 job is in top 10].
    4. NDCG@10(u, t) uses the simulator's true
       p_sim_like = P(y >= 3 | theta_true_u, lambda_u, delta_j) as
       graded relevance.

  Aggregate across users by per-t nanmean and (p25, p75) on a
  rectangular grid t = 1, ..., T_max. User curves that end early are
  NaN-padded so they do not pull down later t.

  Two comparison curves

    theta_true_oracle, scored with the true theta from t = 1 (constant
      horizontal upper bound).
    random, per-user uniform scores re-drawn each user (constant
      horizontal lower bound).

Outputs

  rl/results/v2/data/m4rl_trajectory.json
  rl/results/v2/plots/m4rl_recommendation_over_time.png

Run

  PYTHONPATH=ma-irt KMP_DUPLICATE_LIB_OK=TRUE \\
    python rl/scripts/eval_v2_trajectory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
V2_DEV_DIR = REPO / "rl" / "data" / "v2_dev"
PLOT_DIR = REPO / "rl" / "results" / "v2" / "plots"
DATA_DIR = REPO / "rl" / "results" / "v2" / "data"

_SRC = REPO / "rl" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from irtrec.datagen.synth_likes import (  # noqa: E402
    GPCM_BETA,
    GPCM_K,
    GPCM_LIKED_CATEGORY,
    oracle_like_prob,
)

# Reuse loaders from the baselines script so the eval cohort matches.
sys.path.insert(0, str(REPO / "rl" / "scripts"))
from eval_v2_baselines import (  # noqa: E402
    SPLIT_SEED,
    build_user_positives,
    build_user_responses,
    gpcm_log_probs_grid,
    load_v2_dataset,
)

TRAJECTORY_SEED = 20260604
TOP_K = 10

# Minimum users required at a given t for the headline aggregation to be
# considered stable. Users whose T_u falls below t contribute NaN, so
# beyond the deepest t with this many surviving users the curve gets
# dominated by tail noise.
N_VALID_MIN = 50


# ---------------------------------------------------------------------------
# Per-user trajectory computation.
# ---------------------------------------------------------------------------


def _ndcg_at_k_graded(
    *,
    order: np.ndarray,
    rel_per_job: np.ndarray,
    k: int,
) -> float:
    """NDCG@k with continuous relevance scores per job.

    Parameters
    ----------
    order
        Ranked job IDs, descending by predicted score. Shape (n_jobs,).
    rel_per_job
        Per-job continuous relevance (the simulator's
        ``oracle_like_prob`` under the true theta). Shape (n_jobs,).
    k
        Cut-off.

    Returns
    -------
    float
        NDCG@k in [0, 1]. Returns 0.0 if ideal DCG is zero.
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = rel_per_job[order[:k]]
    dcg = float((gains * discounts).sum())
    ideal = np.sort(rel_per_job)[::-1][:k]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def trajectory_for_user(
    *,
    job_ids: np.ndarray,
    y: np.ndarray,
    positives_set: set,
    lam: float,
    theta_true_u: float,
    delta_j: np.ndarray,
    beta: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
    rng: np.random.Generator,
    item_log_probs: Dict[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run theta-hat EAP and competing policies for one user.

    Returns dict with arrays of length T_u

      hit_hat, ndcg_hat              theta-hat policy
      hit_true, ndcg_true            theta-true oracle policy (constant)
      hit_random, ndcg_random        random policy (constant)
    """
    T = int(y.shape[0])
    n_jobs = delta_j.shape[0]

    # Pre-compute per-job true relevance (graded NDCG gain).
    rel_per_job = oracle_like_prob(
        theta=theta_true_u,
        lam=lam,
        delta_j=delta_j,
        beta=tuple(beta.tolist()),
        liked_category=GPCM_LIKED_CATEGORY,
    )

    # theta-true oracle: constant ranking across t.
    scores_true = rel_per_job  # same monotone function of theta_true
    order_true = np.argsort(-scores_true, kind="stable")
    hit_true_const = float(
        any(int(j) in positives_set for j in order_true[:TOP_K])
    )
    ndcg_true_const = _ndcg_at_k_graded(
        order=order_true, rel_per_job=rel_per_job, k=TOP_K
    )

    # random: one fixed per-user permutation (CaRReL-style horizontal
    # lower bound). Drawn from the trajectory RNG so each user has its
    # own seeded baseline.
    rand_scores = rng.random(n_jobs)
    order_rand = np.argsort(-rand_scores, kind="stable")
    hit_rand_const = float(
        any(int(j) in positives_set for j in order_rand[:TOP_K])
    )
    ndcg_rand_const = _ndcg_at_k_graded(
        order=order_rand, rel_per_job=rel_per_job, k=TOP_K
    )

    hit_hat = np.empty(T, dtype=np.float64)
    ndcg_hat = np.empty(T, dtype=np.float64)
    hit_true = np.full(T, hit_true_const, dtype=np.float64)
    ndcg_true = np.full(T, ndcg_true_const, dtype=np.float64)
    hit_random = np.full(T, hit_rand_const, dtype=np.float64)
    ndcg_random = np.full(T, ndcg_rand_const, dtype=np.float64)

    # Incremental EAP loop.
    log_post = log_prior.copy()
    for t in range(T):
        q = int(job_ids[t])
        r = int(y[t])
        if q not in item_log_probs:
            item_log_probs[q] = gpcm_log_probs_grid(
                theta_grid=grid,
                lam=lam,
                delta_j=float(delta_j[q]),
                beta=beta,
            )
        # NOTE the item_log_probs cache is keyed only by q because
        # within a single user lam is fixed. The caller resets the
        # cache between users.
        log_post = log_post + item_log_probs[q][:, r]
        m = log_post.max()
        w = np.exp(log_post - m)
        w /= w.sum()
        theta_hat_t = float((w * grid).sum())

        scores_hat = oracle_like_prob(
            theta=theta_hat_t,
            lam=lam,
            delta_j=delta_j,
            beta=tuple(beta.tolist()),
            liked_category=GPCM_LIKED_CATEGORY,
        )
        order_hat = np.argsort(-scores_hat, kind="stable")
        hit_hat[t] = float(
            any(int(j) in positives_set for j in order_hat[:TOP_K])
        )
        ndcg_hat[t] = _ndcg_at_k_graded(
            order=order_hat, rel_per_job=rel_per_job, k=TOP_K
        )

    return {
        "hit_hat": hit_hat,
        "ndcg_hat": ndcg_hat,
        "hit_true": hit_true,
        "ndcg_true": ndcg_true,
        "hit_random": hit_random,
        "ndcg_random": ndcg_random,
        "T": T,
    }


# ---------------------------------------------------------------------------
# Aggregation across users.
# ---------------------------------------------------------------------------


def aggregate_trajectories(
    per_user: List[Dict[str, np.ndarray]],
    T_max: int,
    T_fixed: int,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """Aggregate per-t curves two ways.

    variable_cohort
      Stack user curves into a (n_users, T_max) matrix with NaN padding,
      take nanmean and nanpercentile per column. Cohort shrinks with t
      because short-session users drop out, so apparent trends mix
      learning and selection.

    fixed_cohort
      Restrict to users with T_u >= T_fixed. Every column from t=1 to
      t=T_fixed has the same denominator, so any movement is genuinely
      within-session learning rather than selection.
    """
    policies = ["hat", "true", "random"]
    metrics = ["hit", "ndcg"]

    n = len(per_user)

    def _agg(matrix: np.ndarray) -> Dict[str, List[float]]:
        mean = np.nanmean(matrix, axis=0)
        p25 = np.nanpercentile(matrix, 25, axis=0)
        p75 = np.nanpercentile(matrix, 75, axis=0)
        n_valid = np.sum(~np.isnan(matrix), axis=0).astype(int)
        return {
            "mean": mean.tolist(),
            "p25": p25.tolist(),
            "p75": p75.tolist(),
            "n_valid": n_valid.tolist(),
        }

    variable_cohort: Dict[str, Dict[str, List[float]]] = {}
    fixed_cohort: Dict[str, Dict[str, List[float]]] = {}

    fixed_idx = [i for i, rec in enumerate(per_user) if rec["T"] >= T_fixed]
    for pol in policies:
        for met in metrics:
            key = f"{met}_{pol}"
            mat = np.full((n, T_max), np.nan, dtype=np.float64)
            for i, rec in enumerate(per_user):
                T = rec["T"]
                mat[i, :T] = rec[key]
            variable_cohort[key] = _agg(mat)

            mat_fixed = mat[fixed_idx, :T_fixed]
            fixed_cohort[key] = _agg(mat_fixed)

    fixed_cohort["_n_users"] = {"value": [len(fixed_idx)]}
    fixed_cohort["_T_fixed"] = {"value": [T_fixed]}
    return variable_cohort, fixed_cohort


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def plot_trajectory(
    *,
    variable: Dict[str, Dict[str, List[float]]],
    fixed: Dict[str, Dict[str, List[float]]],
    T_max: int,
    T_fixed: int,
    n_eval: int,
    n_fixed_users: int,
    n_jobs: int,
    out_path: Path,
    n_valid_min: int = N_VALID_MIN,
) -> None:
    """Draw the recommendation-over-time figure for Hit@10.

    The figure has two panels.

    Left, variable cohort. Per-t mean and IQR using every user who is
      still in-session at t. This is the CaRReL-style curve. Its rising
      shape is dominated by selection (long-T users have more positives
      and more chances), not by learning.

    Right, fixed cohort. Users with T_u >= T_fixed only, so the
      denominator is constant across t. In a 1D IRT recommender the
      EAP-learned ranking is a monotone function of -delta_j and is
      therefore independent of theta_hat_t, so this panel is expected
      to be flat. A flat curve is the diagnostic, not a flaw.

    Both panels carry the theta-true oracle and random horizontals from
    their own cohort.
    """

    color_hat = "#ff7f0e"

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 5.0), sharey=True)

    # ---- Left panel, variable cohort.
    ax = axes[0]
    n_valid = np.asarray(variable["hit_hat"]["n_valid"])
    valid = n_valid >= n_valid_min
    t_axis = np.arange(1, T_max + 1)
    mean_hat = np.asarray(variable["hit_hat"]["mean"])
    p25_hat = np.asarray(variable["hit_hat"]["p25"])
    p75_hat = np.asarray(variable["hit_hat"]["p75"])
    ax.plot(
        t_axis[valid],
        mean_hat[valid],
        color=color_hat,
        linewidth=2.0,
        label=r"$\hat{\theta}$ EAP, per-t mean",
    )
    ax.fill_between(
        t_axis[valid],
        p25_hat[valid],
        p75_hat[valid],
        color=color_hat,
        alpha=0.20,
        label=r"$\hat{\theta}$ IQR (p25-p75)",
    )
    true_mean_var = float(np.nanmean(np.asarray(variable["hit_true"]["mean"])[valid]))
    rand_mean_var = float(np.nanmean(np.asarray(variable["hit_random"]["mean"])[valid]))
    ax.axhline(
        true_mean_var,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.6,
        label=rf"$\theta$ oracle (cohort mean) = {true_mean_var:.3f}",
    )
    ax.axhline(
        rand_mean_var,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.6,
        label=f"Random (cohort mean) = {rand_mean_var:.3f}",
    )
    ax.set_xlabel("items asked t")
    ax.set_ylabel("Hit@10")
    ax.set_title(
        f"Variable cohort, n(t=1)={n_eval}\n"
        f"(rise is selection, not learning)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    last_valid_t = int(np.where(valid)[0].max() + 1)
    ax.set_xlim(0, last_valid_t + 1)
    ax.legend(loc="lower right", fontsize=8)

    # ---- Right panel, fixed cohort.
    ax = axes[1]
    t_axis_f = np.arange(1, T_fixed + 1)
    mean_hat_f = np.asarray(fixed["hit_hat"]["mean"])
    p25_hat_f = np.asarray(fixed["hit_hat"]["p25"])
    p75_hat_f = np.asarray(fixed["hit_hat"]["p75"])
    ax.plot(
        t_axis_f,
        mean_hat_f,
        color=color_hat,
        linewidth=2.0,
        label=r"$\hat{\theta}$ EAP, per-t mean",
    )
    ax.fill_between(
        t_axis_f,
        p25_hat_f,
        p75_hat_f,
        color=color_hat,
        alpha=0.20,
        label=r"$\hat{\theta}$ IQR (p25-p75)",
    )
    true_mean_f = float(np.nanmean(np.asarray(fixed["hit_true"]["mean"])))
    rand_mean_f = float(np.nanmean(np.asarray(fixed["hit_random"]["mean"])))
    ax.axhline(
        true_mean_f,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.6,
        label=rf"$\theta$ oracle = {true_mean_f:.3f}",
    )
    ax.axhline(
        rand_mean_f,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.6,
        label=f"Random = {rand_mean_f:.3f}",
    )
    ax.set_xlabel("items asked t")
    ax.set_title(
        rf"Fixed cohort, $T_u \geq$ {T_fixed} (n={n_fixed_users})"
        "\n(flat: 1D ranking is "
        r"$\theta$-invariant)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.set_xlim(0, T_fixed + 1)
    ax.legend(loc="lower right", fontsize=8)

    ymax = max(
        0.10,
        float(np.nanmax(mean_hat[valid])),
        float(np.nanmax(mean_hat_f)),
        true_mean_var,
        true_mean_f,
    ) * 1.15
    for a in axes:
        a.set_ylim(0.0, ymax)

    fig.suptitle(
        f"v2 Recommendation Hit@10 Over Test Length, top-K = {TOP_K}, "
        f"n_jobs = {n_jobs}, EAP grid 91 points on [-4.5, 4.5]",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[m4rl-trajectory] loading v2_dev")
    dataset = load_v2_dataset()
    users = dataset["users"]
    jobs = dataset["jobs"]
    responses = dataset["responses"]
    meta = dataset["oracle_metadata"]

    n_users = int(meta["n_users"])
    n_jobs = int(meta["n_jobs"])
    theta_true = np.asarray([u["theta"] for u in users], dtype=np.float64)
    lambda_u = np.asarray([u["lambda_u"] for u in users], dtype=np.float64)
    delta_j = np.asarray([j["delta_j"] for j in jobs], dtype=np.float64)
    beta = np.asarray(GPCM_BETA, dtype=np.float64)

    positives = build_user_positives(responses, n_users)
    user_jobs, user_y = build_user_responses(responses, n_users)

    # 80/20 user split, mirror eval_v2_baselines.py.
    rng_split = np.random.default_rng(SPLIT_SEED)
    perm = rng_split.permutation(n_users)
    n_train = int(round(0.8 * n_users))
    test_users = perm[n_train:].astype(np.int64)
    eval_users = np.asarray(
        [int(u) for u in test_users if len(positives[int(u)]) > 0],
        dtype=np.int64,
    )
    n_eval = len(eval_users)
    print(f"  n_users={n_users}, n_jobs={n_jobs}, n_eval={n_eval}")

    grid = np.linspace(-4.5, 4.5, 91)
    log_prior = -0.5 * grid**2

    # Walk each eval user.
    rng_traj = np.random.default_rng(TRAJECTORY_SEED)
    per_user: List[Dict[str, np.ndarray]] = []
    T_max = 0
    for idx, uid in enumerate(eval_users.tolist()):
        y = user_y[uid]
        jobs_u = user_jobs[uid]
        if y.size == 0:
            continue
        # Deterministic per-user RNG keyed on uid so that re-runs are
        # idempotent regardless of evaluation order.
        rng_u = np.random.default_rng(TRAJECTORY_SEED + 7919 * (uid + 1))
        item_cache: Dict[int, np.ndarray] = {}
        rec = trajectory_for_user(
            job_ids=jobs_u,
            y=y,
            positives_set=positives[uid],
            lam=float(lambda_u[uid]),
            theta_true_u=float(theta_true[uid]),
            delta_j=delta_j,
            beta=beta,
            grid=grid,
            log_prior=log_prior,
            rng=rng_u,
            item_log_probs=item_cache,
        )
        per_user.append(rec)
        T_max = max(T_max, rec["T"])
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx + 1}/{n_eval} users, T_max so far = {T_max}")

    _ = rng_traj.random()  # silence unused rng

    # Pick T_fixed as the largest t where the variable-cohort cell still
    # holds at least N_VALID_MIN users. This caps the headline at a t
    # the data can actually support and lets us also report a fixed-
    # cohort curve on the same horizon.
    Ts = np.asarray([rec["T"] for rec in per_user], dtype=np.int64)
    survival = np.zeros(T_max + 1, dtype=np.int64)
    for T in Ts:
        survival[1 : T + 1] += 1
    valid_ts = np.where(survival >= N_VALID_MIN)[0]
    T_fixed = int(valid_ts.max()) if valid_ts.size else int(np.median(Ts))
    n_fixed_users = int((Ts >= T_fixed).sum())
    print(
        f"[m4rl-trajectory] T_max = {T_max}, headline T_fixed = {T_fixed} "
        f"(n_fixed_users = {n_fixed_users}, min users = {N_VALID_MIN})"
    )

    variable_cohort, fixed_cohort = aggregate_trajectories(
        per_user, T_max=T_max, T_fixed=T_fixed
    )

    def at(metric_key: str, t: int, table: Dict[str, Dict[str, List[float]]]) -> float:
        # t is 1-indexed.
        arr = np.asarray(table[metric_key]["mean"])
        idx = max(0, min(len(arr) - 1, t - 1))
        return float(arr[idx])

    def iqr_width(
        metric_key: str, t: int, table: Dict[str, Dict[str, List[float]]]
    ) -> float:
        p25 = np.asarray(table[metric_key]["p25"])
        p75 = np.asarray(table[metric_key]["p75"])
        idx = max(0, min(len(p25) - 1, t - 1))
        return float(p75[idx] - p25[idx])

    headline_t_marks = [1, 5, 10, 25]
    headline_t_marks = [t for t in headline_t_marks if t <= T_fixed]

    hit_hat_at: Dict[str, float] = {}
    ndcg_hat_at: Dict[str, float] = {}
    for t in headline_t_marks:
        hit_hat_at[f"t={t}"] = at("hit_hat", t, fixed_cohort)
        ndcg_hat_at[f"t={t}"] = at("ndcg_hat", t, fixed_cohort)
    hit_hat_at[f"t={T_fixed}"] = at("hit_hat", T_fixed, fixed_cohort)
    ndcg_hat_at[f"t={T_fixed}"] = at("ndcg_hat", T_fixed, fixed_cohort)

    var_hit_at: Dict[str, float] = {}
    var_ndcg_at: Dict[str, float] = {}
    for t in headline_t_marks:
        var_hit_at[f"t={t}"] = at("hit_hat", t, variable_cohort)
        var_ndcg_at[f"t={t}"] = at("ndcg_hat", t, variable_cohort)
    var_hit_at[f"t={T_fixed}"] = at("hit_hat", T_fixed, variable_cohort)
    var_ndcg_at[f"t={T_fixed}"] = at("ndcg_hat", T_fixed, variable_cohort)

    headline = {
        "T_fixed": T_fixed,
        "n_fixed_users": n_fixed_users,
        "n_eval_at_t1": n_eval,
        # Fixed-cohort headline (the honest within-user trajectory).
        "fixed_cohort": {
            "hit_hat": hit_hat_at,
            "ndcg_hat": ndcg_hat_at,
            "slope_hit_t1_to_t10": at("hit_hat", 10, fixed_cohort)
            - at("hit_hat", 1, fixed_cohort),
            "slope_hit_t10_to_Tfixed": at("hit_hat", T_fixed, fixed_cohort)
            - at("hit_hat", 10, fixed_cohort),
            "slope_ndcg_t1_to_t10": at("ndcg_hat", 10, fixed_cohort)
            - at("ndcg_hat", 1, fixed_cohort),
            "slope_ndcg_t10_to_Tfixed": at("ndcg_hat", T_fixed, fixed_cohort)
            - at("ndcg_hat", 10, fixed_cohort),
            "iqr_hit_t10": iqr_width("hit_hat", 10, fixed_cohort),
            f"iqr_hit_t{T_fixed}": iqr_width("hit_hat", T_fixed, fixed_cohort),
            "iqr_ndcg_t10": iqr_width("ndcg_hat", 10, fixed_cohort),
            f"iqr_ndcg_t{T_fixed}": iqr_width("ndcg_hat", T_fixed, fixed_cohort),
            "oracle_hit_const": float(
                np.nanmean(np.asarray(fixed_cohort["hit_true"]["mean"]))
            ),
            "random_hit_const": float(
                np.nanmean(np.asarray(fixed_cohort["hit_random"]["mean"]))
            ),
        },
        # Variable-cohort headline (CaRReL-style shape, contaminated by
        # selection). Provided for narrative comparison.
        "variable_cohort": {
            "hit_hat": var_hit_at,
            "ndcg_hat": var_ndcg_at,
            "slope_hit_t1_to_t10": at("hit_hat", 10, variable_cohort)
            - at("hit_hat", 1, variable_cohort),
            "slope_hit_t10_to_Tfixed": at("hit_hat", T_fixed, variable_cohort)
            - at("hit_hat", 10, variable_cohort),
            "max_hit_in_window": float(
                np.nanmax(np.asarray(variable_cohort["hit_hat"]["mean"])[:T_fixed])
            ),
            "iqr_hit_t10": iqr_width("hit_hat", 10, variable_cohort),
            f"iqr_hit_t{T_fixed}": iqr_width("hit_hat", T_fixed, variable_cohort),
        },
        # Top-level convenience aliases for the schema return.
        "hit_hat": hit_hat_at,
        "ndcg_hat": ndcg_hat_at,
        "slope_hit_t1_to_t10": at("hit_hat", 10, fixed_cohort)
        - at("hit_hat", 1, fixed_cohort),
        "slope_hit_t10_to_Tfixed": at("hit_hat", T_fixed, fixed_cohort)
        - at("hit_hat", 10, fixed_cohort),
        "iqr_hit_t10": iqr_width("hit_hat", 10, fixed_cohort),
        f"iqr_hit_t{T_fixed}": iqr_width("hit_hat", T_fixed, fixed_cohort),
    }

    summary = {
        "preset": meta["preset_name"],
        "n_users": n_users,
        "n_jobs": n_jobs,
        "n_eval_users": n_eval,
        "split_seed": SPLIT_SEED,
        "trajectory_seed": TRAJECTORY_SEED,
        "top_k": TOP_K,
        "T_max": T_max,
        "T_fixed": T_fixed,
        "n_fixed_users": n_fixed_users,
        "n_valid_min": N_VALID_MIN,
        "grid": {"low": -4.5, "high": 4.5, "n_points": int(grid.shape[0])},
        "K": GPCM_K,
        "gpcm_beta": list(GPCM_BETA),
        "trajectories_variable_cohort": variable_cohort,
        "trajectories_fixed_cohort": fixed_cohort,
        "headline": headline,
    }

    out_json = DATA_DIR / "m4rl_trajectory.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[m4rl-trajectory] wrote {out_json}")

    out_png = PLOT_DIR / "m4rl_recommendation_over_time.png"
    plot_trajectory(
        variable=variable_cohort,
        fixed=fixed_cohort,
        T_max=T_max,
        T_fixed=T_fixed,
        n_eval=n_eval,
        n_fixed_users=n_fixed_users,
        n_jobs=n_jobs,
        out_path=out_png,
    )
    print(f"[m4rl-trajectory] wrote {out_png}")

    # Console summary.
    print(f"\n[m4rl-trajectory] headline Hit@10 (fixed cohort, T_fixed = {T_fixed})")
    for k, v in hit_hat_at.items():
        print(f"  hit_hat {k:>8s} = {v:.4f}")
    fc = headline["fixed_cohort"]
    vc = headline["variable_cohort"]
    print(
        f"  slope t=1 -> t=10      : {fc['slope_hit_t1_to_t10']:+.4f}\n"
        f"  slope t=10 -> t={T_fixed}    : {fc['slope_hit_t10_to_Tfixed']:+.4f}\n"
        f"  IQR width  t=10        : {fc['iqr_hit_t10']:.4f}\n"
        f"  IQR width  t={T_fixed}       : {fc[f'iqr_hit_t{T_fixed}']:.4f}\n"
        f"  oracle const (fixed)   : {fc['oracle_hit_const']:.4f}\n"
        f"  random const (fixed)   : {fc['random_hit_const']:.4f}"
    )
    print(f"\n[m4rl-trajectory] variable-cohort Hit@10 (selection-mixed)")
    for k, v in vc["hit_hat"].items():
        print(f"  hit_hat {k:>8s} = {v:.4f}")
    print(
        f"  slope t=1 -> t=10      : {vc['slope_hit_t1_to_t10']:+.4f}\n"
        f"  slope t=10 -> t={T_fixed}    : {vc['slope_hit_t10_to_Tfixed']:+.4f}\n"
        f"  max in window          : {vc['max_hit_in_window']:.4f}"
    )


if __name__ == "__main__":
    main()
