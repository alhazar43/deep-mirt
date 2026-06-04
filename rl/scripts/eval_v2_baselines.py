"""M4-RL recommender baselines on the v2 dev simulator.

Mirrors :mod:`rl/scripts/run_m23_baselines.py` for v2, adapting the
scoring rules to the K=5 GPCM preference model. Four baselines.

  random         Per-user seeded random permutation.
  popularity     Rank by training-set IsLiked counts (jitter for ties).
  theta_true     Oracle. Rank by P(y >= 3 | theta_true, lambda_u,
                 delta_j_true) using the same GPCM the generator used.
  theta_hat      Realistic. Same scoring rule with theta_hat recovered
                 by Bayesian EAP from the user's own K=5 ordinal
                 responses on their candidate set, using the TRUE
                 lambda_u and TRUE delta_j as item parameters.

Eval protocol matches v1.
  80/20 user split, seed=0. Hit@5, Hit@10, Hit@20, NDCG@10, MRR with
  95 percent bootstrap CIs over 500 resamples. Held-out users with at
  least one IsLiked positive are the evaluable set.

Outputs.
  rl/results/v2/data/m4rl_baselines.json
  rl/results/v2/data/m4rl_theta_recovery.json
  rl/results/v2/plots/m4rl_baselines.png

Run.
  PYTHONPATH=ma-irt KMP_DUPLICATE_LIB_OK=TRUE \\
    python rl/scripts/eval_v2_baselines.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
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
    gpcm_category_probs,
    oracle_like_prob,
)

SPLIT_SEED = 0
BASELINE_RANDOM_SEED = 12345
N_BOOTSTRAP = 500
K_LIST = (5, 10, 20)


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------


def load_v2_dataset() -> Dict:
    out: Dict = {}
    for name in ["users", "jobs", "responses", "splits", "oracle_metadata"]:
        with open(V2_DEV_DIR / f"{name}.json") as fh:
            out[name] = json.load(fh)
    return out


def build_user_positives(responses: List[Dict], n_users: int) -> List[set]:
    pos: List[set] = [set() for _ in range(n_users)]
    for row in responses:
        if int(row["IsLiked"]) == 1:
            pos[int(row["user_id"])].add(int(row["job_id"]))
    return pos


def build_user_responses(
    responses: List[Dict], n_users: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Group long-form responses by user. Returns per-user (job_ids, y)."""
    per_user_jobs: List[List[int]] = [[] for _ in range(n_users)]
    per_user_y: List[List[int]] = [[] for _ in range(n_users)]
    for row in responses:
        uid = int(row["user_id"])
        per_user_jobs[uid].append(int(row["job_id"]))
        per_user_y[uid].append(int(row["y"]))
    out_jobs = [np.asarray(j, dtype=np.int64) for j in per_user_jobs]
    out_y = [np.asarray(y, dtype=np.int64) for y in per_user_y]
    return out_jobs, out_y


# ---------------------------------------------------------------------------
# Theta-hat recovery via EAP on the user's own GPCM responses.
# Uses TRUE lambda_u and TRUE delta_j as item parameters.
# ---------------------------------------------------------------------------


def gpcm_log_probs_grid(
    *,
    theta_grid: np.ndarray,
    lam: float,
    delta_j: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Log P(y = k | theta) for one item across a theta grid.

    Shapes
    ------
    theta_grid : (G,)
    Returns    : (G, K)
    """
    K = beta.shape[0] + 1
    G = theta_grid.shape[0]
    psi = np.zeros((G, K), dtype=np.float64)
    cumsum = np.zeros(G, dtype=np.float64)
    for k in range(1, K):
        cumsum = cumsum + lam * theta_grid - (beta[k - 1] + delta_j)
        psi[:, k] = cumsum
    Umax = psi.max(axis=1, keepdims=True)
    Z = Umax.squeeze(1) + np.log(np.exp(psi - Umax).sum(axis=1))
    return psi - Z[:, None]


def eap_theta_v2(
    *,
    y: np.ndarray,
    job_ids: np.ndarray,
    lam: float,
    delta_j: np.ndarray,
    beta: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> float:
    """EAP theta given one user's K=5 responses on their candidate set."""
    log_post = log_prior.copy()
    for q, r in zip(job_ids.tolist(), y.tolist()):
        lp = gpcm_log_probs_grid(theta_grid=grid, lam=lam, delta_j=float(delta_j[q]), beta=beta)
        log_post += lp[:, int(r)]
    m = log_post.max()
    w = np.exp(log_post - m)
    w /= w.sum()
    return float((w * grid).sum())


def recover_theta_hat_v2(
    *,
    user_y: List[np.ndarray],
    user_jobs: List[np.ndarray],
    lambda_u: np.ndarray,
    delta_j: np.ndarray,
    beta: np.ndarray = np.asarray(GPCM_BETA, dtype=np.float64),
) -> np.ndarray:
    grid = np.linspace(-4.5, 4.5, 91)
    log_prior = -0.5 * grid**2
    n_users = len(user_y)
    theta_hat = np.zeros(n_users, dtype=np.float64)
    for uid in range(n_users):
        y = user_y[uid]
        jobs = user_jobs[uid]
        if y.size == 0:
            # No responses, fall back to prior mean.
            theta_hat[uid] = 0.0
            continue
        theta_hat[uid] = eap_theta_v2(
            y=y,
            job_ids=jobs,
            lam=float(lambda_u[uid]),
            delta_j=delta_j,
            beta=beta,
            grid=grid,
            log_prior=log_prior,
        )
    return theta_hat


# ---------------------------------------------------------------------------
# Ranking baselines.
# ---------------------------------------------------------------------------


def score_random(n_users: int, n_jobs: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n_users, n_jobs))


def score_popularity(
    responses: List[Dict],
    train_user_set: set,
    n_users: int,
    n_jobs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.zeros(n_jobs, dtype=np.float64)
    for row in responses:
        if int(row["user_id"]) in train_user_set and int(row["IsLiked"]) == 1:
            counts[int(row["job_id"])] += 1.0
    jitter = rng.random((n_users, n_jobs)) * 1e-6
    return counts[None, :] + jitter


def score_v2_1d_match(
    theta: np.ndarray,
    lambda_u: np.ndarray,
    delta_j: np.ndarray,
    beta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-user P(y >= 3 | theta, lambda_u, delta_j) for every job."""
    n_users = theta.shape[0]
    n_jobs = delta_j.shape[0]
    scores = np.empty((n_users, n_jobs), dtype=np.float64)
    for uid in range(n_users):
        scores[uid] = oracle_like_prob(
            theta=float(theta[uid]),
            lam=float(lambda_u[uid]),
            delta_j=delta_j,
            beta=tuple(beta.tolist()),
            liked_category=GPCM_LIKED_CATEGORY,
        )
    jitter = rng.random(scores.shape) * 1e-9
    return scores + jitter


# ---------------------------------------------------------------------------
# Metrics with bootstrap CIs.
# ---------------------------------------------------------------------------


def per_user_metrics(
    scores: np.ndarray,
    positives: List[set],
    user_ids: np.ndarray,
    k_list=K_LIST,
) -> Dict[str, np.ndarray]:
    hits: Dict[int, List[float]] = {k: [] for k in k_list}
    ndcgs: List[float] = []
    mrrs: List[float] = []
    for u in user_ids:
        pos = positives[int(u)]
        if not pos:
            continue
        s = scores[int(u)]
        order = np.argsort(-s, kind="stable")
        rels = np.fromiter(
            (1.0 if int(j) in pos else 0.0 for j in order),
            dtype=np.float64,
            count=order.shape[0],
        )
        for k in k_list:
            hits[k].append(float(rels[:k].sum() > 0))
        k = 10
        gains = rels[:k]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((gains * discounts).sum())
        n_pos = len(pos)
        ideal_rels = np.zeros(k, dtype=np.float64)
        ideal_rels[: min(k, n_pos)] = 1.0
        idcg = float((ideal_rels * discounts).sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        first = np.argmax(rels > 0)
        mrr = float(1.0 / (first + 1)) if rels[first] > 0 else 0.0
        mrrs.append(mrr)
    out = {f"hit@{k}": np.asarray(hits[k]) for k in k_list}
    out["ndcg@10"] = np.asarray(ndcgs)
    out["mrr"] = np.asarray(mrrs)
    return out


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def plot_hit_at_10(summary: Dict, out_path: Path) -> None:
    name_order = ["random", "popularity", "theta_hat", "theta_true"]
    pretty = {
        "random": "Random",
        "popularity": "Popularity",
        "theta_hat": r"1D match ($\hat{\theta}$)",
        "theta_true": r"1D match ($\theta$ oracle)",
    }
    means, los, his = [], [], []
    for name in name_order:
        m = summary["baselines"][name]["hit@10"]
        means.append(m["mean"])
        los.append(m["mean"] - m["ci_lo"])
        his.append(m["ci_hi"] - m["mean"])
    yerr = np.asarray([los, his])
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(name_order))
    bars = ax.bar(
        x, means, yerr=yerr, capsize=4, color=colors, alpha=0.85,
        edgecolor="black", linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[n] for n in name_order])
    ax.set_ylabel("Hit@10")
    ax.set_ylim(0, max(0.05, max(np.asarray(means) + np.asarray(his)) * 1.15))
    ax.set_title(
        f"v2 Hit@10 over O*NET pool (n_jobs={summary['n_jobs']}), "
        f"held-out users (n_eval={summary['n_eval_users']})\n"
        f"95% bootstrap CI, {N_BOOTSTRAP} resamples"
    )
    ax.grid(alpha=0.3, axis="y")
    for b, m in zip(bars, means):
        ax.annotate(
            f"{m:.3f}",
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[m4rl-baselines] loading v2_dev")
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

    # 80/20 user split.
    rng_split = np.random.default_rng(SPLIT_SEED)
    perm = rng_split.permutation(n_users)
    n_train = int(round(0.8 * n_users))
    train_users = set(int(u) for u in perm[:n_train])
    test_users = perm[n_train:].astype(np.int64)
    n_test = len(test_users)
    eval_users = np.asarray(
        [int(u) for u in test_users if len(positives[int(u)]) > 0],
        dtype=np.int64,
    )
    n_eval = len(eval_users)
    print(f"  n_users={n_users}, n_jobs={n_jobs}")
    print(f"  n_train={n_train}, n_test={n_test}, n_eval={n_eval}")

    # Theta-hat from EAP on each user's GPCM responses.
    print("[m4rl-baselines] running EAP theta recovery on responses")
    user_jobs, user_y = build_user_responses(responses, n_users)
    theta_hat = recover_theta_hat_v2(
        user_y=user_y,
        user_jobs=user_jobs,
        lambda_u=lambda_u,
        delta_j=delta_j,
        beta=beta,
    )
    r_hat = float(np.corrcoef(theta_true, theta_hat)[0, 1])
    rmse_hat = float(np.sqrt(np.mean((theta_hat - theta_true) ** 2)))
    print(f"  Pearson r (theta_hat, theta_true) = {r_hat:.4f}, RMSE = {rmse_hat:.4f}")

    # Score matrices.
    print("[m4rl-baselines] scoring baselines")
    scores = {
        "random": score_random(
            n_users, n_jobs, np.random.default_rng(BASELINE_RANDOM_SEED + 1)
        ),
        "popularity": score_popularity(
            responses, train_users, n_users, n_jobs,
            np.random.default_rng(BASELINE_RANDOM_SEED + 2),
        ),
        "theta_true": score_v2_1d_match(
            theta_true, lambda_u, delta_j, beta,
            np.random.default_rng(BASELINE_RANDOM_SEED + 3),
        ),
        "theta_hat": score_v2_1d_match(
            theta_hat, lambda_u, delta_j, beta,
            np.random.default_rng(BASELINE_RANDOM_SEED + 3),
        ),
    }

    summary: Dict = {
        "preset": meta["preset_name"],
        "n_users": n_users,
        "n_jobs": n_jobs,
        "n_train_users": int(n_train),
        "n_test_users": int(n_test),
        "n_eval_users": n_eval,
        "split_seed": SPLIT_SEED,
        "bootstrap_iters": N_BOOTSTRAP,
        "K": GPCM_K,
        "gpcm_beta": GPCM_BETA,
        "theta_hat_recovery": {"pearson_r": r_hat, "rmse": rmse_hat},
        "baselines": {},
    }
    metric_names = [f"hit@{k}" for k in K_LIST] + ["ndcg@10", "mrr"]
    for name, S in scores.items():
        per_user = per_user_metrics(S, positives, eval_users)
        rec: Dict = {}
        for m_name in metric_names:
            vals = per_user[m_name]
            seed = BASELINE_RANDOM_SEED + hash(m_name) % 10_000
            mean, lo, hi = bootstrap_mean_ci(vals, seed=seed)
            rec[m_name] = {
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_eval": int(len(vals)),
            }
        summary["baselines"][name] = rec
        print(
            f"  {name:>11s}  Hit@5={rec['hit@5']['mean']:.3f}  "
            f"Hit@10={rec['hit@10']['mean']:.3f}  "
            f"Hit@20={rec['hit@20']['mean']:.3f}  "
            f"NDCG@10={rec['ndcg@10']['mean']:.3f}  "
            f"MRR={rec['mrr']['mean']:.3f}"
        )

    candidates = ["random", "popularity", "theta_hat"]
    best_name = max(candidates, key=lambda n: summary["baselines"][n]["hit@10"]["mean"])
    summary["best_baseline_hit10"] = {
        "name": best_name,
        "value": summary["baselines"][best_name]["hit@10"]["mean"],
    }

    out_json = DATA_DIR / "m4rl_baselines.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[m4rl-baselines] wrote {out_json}")

    recovery_json = DATA_DIR / "m4rl_theta_recovery.json"
    with open(recovery_json, "w") as f:
        json.dump(
            {
                "preset": meta["preset_name"],
                "n_users": n_users,
                "pearson_r": r_hat,
                "rmse": rmse_hat,
                "theta_true": theta_true.tolist(),
                "theta_hat": theta_hat.tolist(),
                "lambda_u": lambda_u.tolist(),
            },
            f,
        )
    print(f"[m4rl-baselines] wrote {recovery_json}")

    out_png = PLOT_DIR / "m4rl_baselines.png"
    plot_hit_at_10(summary, out_png)
    print(f"[m4rl-baselines] wrote {out_png}")


if __name__ == "__main__":
    main()
