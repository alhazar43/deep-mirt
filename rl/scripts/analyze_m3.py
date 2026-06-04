"""M3 preliminary analysis.

Produces five plots and two JSON metric files under
rl/results/v1/{plots,data}/ to validate the synthetic data generator.

The headline metric is theta recovery (Pearson r, RMSE) computed via
Bayesian EAP (expected a posteriori) using the true item parameters
on a 41-point Gauss-Hermite-like grid. EAP on true items gives the
identifiability ceiling for the response model. If recovery here is
below the > 0.85 target from Plan Section 5.5, that is a property
of the simulator (item bank composition, sequence length), not of
any downstream encoder.

Run:
    PYTHONPATH=ma-irt KMP_DUPLICATE_LIB_OK=TRUE python rl/scripts/analyze_m3.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEV_DIR = REPO / "rl" / "data" / "sim_v1_dev"
RECOVERY_DIR = REPO / "rl" / "data" / "sim_v1_recovery"
PLOT_DIR = REPO / "rl" / "results" / "v1" / "plots"
DATA_DIR = REPO / "rl" / "results" / "v1" / "data"
RESULTS_MD = REPO / "rl" / "results" / "v1" / "RESULTS.md"


def load_dataset(path: Path) -> Dict:
    out = {}
    for name in [
        "sequences",
        "true_irt_parameters",
        "true_preference_parameters",
        "jobs",
        "metadata",
    ]:
        with open(path / f"{name}.json") as f:
            out[name] = json.load(f)
    return out


def gpcm_log_probs(theta: np.ndarray, alpha: float, beta: np.ndarray) -> np.ndarray:
    """GPCM log P(Y=k|theta) for k=0..K-1, vectorised over theta grid.

    Args:
        theta: shape (G,) ability grid.
        alpha: scalar discrimination.
        beta: shape (K-1,) step thresholds.

    Returns:
        log_p: shape (G, K) log probabilities.
    """
    K = len(beta) + 1
    # Cumulative sums: U_k = sum_{m=0}^{k-1} alpha*(theta - beta_m), with U_0 = 0.
    # log P(k) = U_k - logsumexp(U_0..U_{K-1}).
    G = theta.shape[0]
    U = np.zeros((G, K), dtype=np.float64)
    # U[:, 0] = 0
    for k in range(1, K):
        U[:, k] = U[:, k - 1] + alpha * (theta - beta[k - 1])
    # Stable log-sum-exp
    Umax = U.max(axis=1, keepdims=True)
    Z = Umax.squeeze(1) + np.log(np.exp(U - Umax).sum(axis=1))
    log_p = U - Z[:, None]
    return log_p


def eap_theta(
    responses: np.ndarray,
    questions: np.ndarray,
    alpha: np.ndarray,
    beta_list: List[np.ndarray],
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> Tuple[float, float]:
    """Compute EAP estimate and posterior SD for one user.

    Args:
        responses: (T,) int observed responses.
        questions: (T,) int 1-indexed question ids.
        alpha: (Q,) discriminations.
        beta_list: per-item step thresholds, length Q.
        grid: (G,) theta grid.
        log_prior: (G,) log N(0,1) prior on grid.

    Returns:
        (theta_hat, posterior_sd).
    """
    log_post = log_prior.copy()
    for q, r in zip(questions, responses):
        j = int(q) - 1
        lp = gpcm_log_probs(grid, float(alpha[j]), beta_list[j])
        log_post += lp[:, int(r)]
    # Normalise.
    m = log_post.max()
    w = np.exp(log_post - m)
    w /= w.sum()
    mean = float((w * grid).sum())
    var = float((w * (grid - mean) ** 2).sum())
    return mean, np.sqrt(var)


def run_theta_recovery(dataset: Dict, label: str) -> Dict:
    """EAP theta recovery using true item parameters."""
    seqs = dataset["sequences"]
    irt = dataset["true_irt_parameters"]
    theta_true = np.asarray(irt["theta"], dtype=np.float64)
    alpha = np.asarray(irt["alpha"], dtype=np.float64)
    beta_list = [np.asarray(b, dtype=np.float64) for b in irt["beta"]]
    n_users = len(seqs)

    # Theta grid for EAP. Wide enough to cover N(0,1) prior tails.
    grid = np.linspace(-4.5, 4.5, 91)
    log_prior = -0.5 * grid**2  # unnormalised log N(0,1); constant drops in normalisation

    theta_hat = np.empty(n_users)
    for i, seq in enumerate(seqs):
        q = np.asarray(seq["questions"], dtype=np.int64)
        r = np.asarray(seq["responses"], dtype=np.int64)
        mean, _ = eap_theta(r, q, alpha, beta_list, grid, log_prior)
        theta_hat[i] = mean

    # Metrics.
    pearson_r = float(np.corrcoef(theta_true, theta_hat)[0, 1])
    rmse = float(np.sqrt(np.mean((theta_hat - theta_true) ** 2)))

    out = {
        "label": label,
        "method": "EAP_true_items",
        "n": int(n_users),
        "pearson_r": pearson_r,
        "rmse": rmse,
        "theta_true_min": float(theta_true.min()),
        "theta_true_max": float(theta_true.max()),
        "theta_hat_min": float(theta_hat.min()),
        "theta_hat_max": float(theta_hat.max()),
        "theta_true": theta_true.tolist(),
        "theta_hat": theta_hat.tolist(),
    }
    return out


def plot_theta_recovery(recovery: Dict, out_path: Path) -> None:
    theta_true = np.asarray(recovery["theta_true"])
    theta_hat = np.asarray(recovery["theta_hat"])
    r = recovery["pearson_r"]
    rmse = recovery["rmse"]
    n = recovery["n"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(theta_true, theta_hat, s=8, alpha=0.35, color="#1f77b4", edgecolors="none")
    lo = min(theta_true.min(), theta_hat.min()) - 0.2
    hi = max(theta_true.max(), theta_hat.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"True $\theta$")
    ax.set_ylabel(r"$\hat{\theta}$ (EAP on true items)")
    ax.set_title(
        f"Theta recovery, {recovery['label']}\n"
        f"N={n}, Pearson r = {r:.3f}, RMSE = {rmse:.3f}"
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_like_distribution(dataset: Dict, out_path: Path, out_json: Path) -> None:
    likes = json.load(open(DEV_DIR / "likes.json"))
    pref = dataset["true_preference_parameters"]
    eng = np.asarray(pref["engagement_class"], dtype=np.int64)
    n_users = len(eng)

    # Per-user like rate.
    n_shown = np.zeros(n_users, dtype=np.int64)
    n_liked = np.zeros(n_users, dtype=np.int64)
    for row in likes:
        u = int(row["user_id"])
        n_shown[u] += 1
        n_liked[u] += int(row["IsLiked"])
    rate = np.where(n_shown > 0, n_liked / np.maximum(n_shown, 1), np.nan)

    mask_valid = n_shown > 0
    rate_rej = rate[(eng == 0) & mask_valid]
    rate_eng = rate[(eng == 1) & mask_valid]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(
        [rate_rej, rate_eng],
        bins=bins,
        stacked=False,
        label=[
            f"rejecter (n={len(rate_rej)})",
            f"engaged (n={len(rate_eng)})",
        ],
        color=["#d62728", "#2ca02c"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("Per-user like rate")
    ax.set_ylabel("Number of users")
    ax.set_title("Like rate distribution by engagement class (sim_v1_dev)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    overall = float(np.mean(rate[mask_valid]))
    out = {
        "overall_like_rate": overall,
        "rejecter_mean_rate": float(rate_rej.mean()) if len(rate_rej) else None,
        "rejecter_median_rate": float(np.median(rate_rej)) if len(rate_rej) else None,
        "engaged_mean_rate": float(rate_eng.mean()) if len(rate_eng) else None,
        "engaged_median_rate": float(np.median(rate_eng)) if len(rate_eng) else None,
        "n_rejecters": int(len(rate_rej)),
        "n_engaged": int(len(rate_eng)),
        "n_likes_rows": len(likes),
        "n_likes_positive": int(sum(row["IsLiked"] for row in likes)),
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)


def plot_engagement_split(dataset: Dict, out_path: Path) -> None:
    pref = dataset["true_preference_parameters"]
    eng = np.asarray(pref["engagement_class"], dtype=np.int64)
    meta = dataset["metadata"]
    target_rej = meta["engagement"]["rejecter_share_target"]
    target_eng = meta["engagement"]["engaged_share_target"]
    n = len(eng)
    actual_rej = float((eng == 0).mean())
    actual_eng = float((eng == 1).mean())

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    x = np.arange(2)
    width = 0.38
    bar_target = ax.bar(
        x - width / 2,
        [target_rej, target_eng],
        width,
        label="target",
        color="#7f7f7f",
        alpha=0.7,
        edgecolor="black",
    )
    bar_actual = ax.bar(
        x + width / 2,
        [actual_rej, actual_eng],
        width,
        label="actual",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["rejecter", "engaged"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of users")
    ax.set_title(f"Engagement class split (N={n})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    for bars in [bar_target, bar_actual]:
        for b in bars:
            ax.annotate(
                f"{b.get_height():.2f}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_k_distribution(dataset: Dict, out_path: Path) -> None:
    nc = dataset["true_irt_parameters"]["n_categories"]
    cfg_dist = dataset["metadata"]["k_distribution"]
    actual = {k: nc.count(k) for k in sorted(set(nc))}
    target = {int(k): int(v) for k, v in cfg_dist.items()}
    all_k = sorted(set(actual) | set(target))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    x = np.arange(len(all_k))
    width = 0.38
    target_vals = [target.get(k, 0) for k in all_k]
    actual_vals = [actual.get(k, 0) for k in all_k]
    bar_t = ax.bar(x - width / 2, target_vals, width, label="configured", color="#7f7f7f", alpha=0.7, edgecolor="black")
    bar_a = ax.bar(x + width / 2, actual_vals, width, label="actual", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in all_k])
    ax.set_ylabel("Number of items")
    ax.set_title("Item K (category count) distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    for bars in [bar_t, bar_a]:
        for b in bars:
            ax.annotate(
                f"{int(b.get_height())}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_j_distribution(dataset: Dict, out_path: Path) -> None:
    pref = dataset["true_preference_parameters"]
    delta = np.asarray(pref["delta_j"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    uniq, counts = np.unique(delta, return_counts=True)
    # Bar chart since delta_j is from work_zone, only ~5 distinct values.
    if len(uniq) <= 10:
        ax.bar(np.arange(len(uniq)), counts, color="#9467bd", alpha=0.85, edgecolor="black")
        ax.set_xticks(np.arange(len(uniq)))
        ax.set_xticklabels([f"{v:.3f}" for v in uniq], rotation=20)
        ax.set_xlabel(r"$\delta_j$ (work_zone z-score)")
        ax.set_ylabel("Number of jobs")
        for i, c in enumerate(counts):
            ax.annotate(
                f"{c}",
                xy=(i, c),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    else:
        ax.hist(delta, bins=30, color="#9467bd", alpha=0.85, edgecolor="black")
        ax.set_xlabel(r"$\delta_j$")
        ax.set_ylabel("Number of jobs")
    ax.set_title(
        f"delta_j distribution (work_zone, n_jobs={len(delta)})\n"
        f"mean={delta.mean():.3f}, sd={delta.std():.3f}, "
        f"unique={len(uniq)}"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def append_results_md(
    recovery_dev: Dict,
    recovery_rec: Dict,
    like_stats_path: Path,
) -> None:
    with open(like_stats_path) as f:
        like_stats = json.load(f)
    lines = []
    lines.append("\n## M3 preliminary analysis (2026-06-04)\n")
    lines.append("Theta recovery via Bayesian EAP on true item parameters.\n")
    lines.append("")
    lines.append("| preset | N | Pearson r | RMSE |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| sim_v1_dev | {recovery_dev['n']} | {recovery_dev['pearson_r']:.3f} | {recovery_dev['rmse']:.3f} |"
    )
    lines.append(
        f"| sim_v1_recovery | {recovery_rec['n']} | {recovery_rec['pearson_r']:.3f} | {recovery_rec['rmse']:.3f} |"
    )
    lines.append("")
    lines.append("Like generation diagnostics on sim_v1_dev.")
    lines.append("")
    lines.append("| stat | value |")
    lines.append("|---|---|")
    lines.append(f"| overall like rate | {like_stats['overall_like_rate']:.3f} |")
    lines.append(f"| rejecter mean rate | {like_stats['rejecter_mean_rate']:.3f} |")
    lines.append(f"| engaged mean rate | {like_stats['engaged_mean_rate']:.3f} |")
    lines.append(f"| n_likes_rows | {like_stats['n_likes_rows']} |")
    lines.append(f"| n_likes_positive | {like_stats['n_likes_positive']} |")
    lines.append("")
    lines.append("Plots saved under rl/results/v1/plots/.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    if RESULTS_MD.exists():
        with open(RESULTS_MD, "a") as f:
            f.write(text)
    else:
        with open(RESULTS_MD, "w") as f:
            f.write("# DRL-MAIRT v1 preliminary results\n")
            f.write(text)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dev = load_dataset(DEV_DIR)
    rec = load_dataset(RECOVERY_DIR)

    # Theta recovery on both, but headline uses recovery (N=5000).
    print("[m3-analyze] EAP theta recovery on sim_v1_dev (N=500)")
    rec_dev = run_theta_recovery(dev, "sim_v1_dev (N=500)")
    print(f"  Pearson r = {rec_dev['pearson_r']:.4f}, RMSE = {rec_dev['rmse']:.4f}")

    print("[m3-analyze] EAP theta recovery on sim_v1_recovery (N=5000)")
    rec_rec = run_theta_recovery(rec, "sim_v1_recovery (N=5000)")
    print(f"  Pearson r = {rec_rec['pearson_r']:.4f}, RMSE = {rec_rec['rmse']:.4f}")

    # Headline plot uses recovery preset.
    plot_theta_recovery(rec_rec, PLOT_DIR / "m3_theta_recovery.png")
    with open(DATA_DIR / "m3_theta_recovery.json", "w") as f:
        out = {k: v for k, v in rec_rec.items() if k not in {"theta_true", "theta_hat"}}
        out["sim_v1_dev"] = {
            "n": rec_dev["n"],
            "pearson_r": rec_dev["pearson_r"],
            "rmse": rec_dev["rmse"],
        }
        json.dump(out, f, indent=2)

    # Like distribution + engagement + K + delta_j on dev preset.
    print("[m3-analyze] like rate distribution + engagement split + K + delta_j")
    like_json = DATA_DIR / "m3_like_distribution.json"
    plot_like_distribution(dev, PLOT_DIR / "m3_like_distribution.png", like_json)
    plot_engagement_split(dev, PLOT_DIR / "m3_engagement_split.png")
    plot_k_distribution(dev, PLOT_DIR / "m3_k_distribution.png")
    plot_delta_j_distribution(dev, PLOT_DIR / "m3_delta_j_distribution.png")

    append_results_md(rec_dev, rec_rec, like_json)
    print("[m3-analyze] done. Plots in", PLOT_DIR)


if __name__ == "__main__":
    main()
