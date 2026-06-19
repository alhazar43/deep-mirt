"""validate.py -- Four validity checks for E2 recovered learning rates.

All correlations use Spearman rho with a bootstrap 95% CI.

(a) Predictive validity    -- r_hat (first half) vs delta_acc + negative control
(b) AFM concurrent         -- r_hat vs AFM/iAFM slope; part-stratified
(c) Split-half reliability -- r_hat_odd vs r_hat_even
(d) Convergent             -- aligned theta r_hat vs responsive theta r_hat
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from deep_irt.core.model import DeepIRTModel
from deep_irt.traj_synth.metrics import fit_rate


# ---------------------------------------------------------------------------
# Bootstrap Spearman CI
# ---------------------------------------------------------------------------

def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2_000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Return (rho, ci_lo, ci_hi) for Spearman(x, y) with bootstrap 95% CI."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = spearmanr(x[idx], y[idx]).statistic
        boot[b] = r if np.isfinite(r) else rho
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return rho, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# (a) Predictive validity
# ---------------------------------------------------------------------------

def predictive_validity(
    seqs_item: list[np.ndarray],
    seqs_resp: list[np.ndarray],
    seqs_corr: list[np.ndarray],
    model: DeepIRTModel,
    batch_size: int = 128,
    rng_seed: int = 1,
) -> dict:
    """Split each sequence at temporal midpoint.

    Recover aligned theta on the first half, fit r_hat_first.
    delta_acc = mean(correct 2nd half) - mean(correct 1st half).
    Compute Spearman(r_hat_first, delta_acc).
    Negative control: randomly permute each learner's response ORDER and redo.
    """
    import torch

    N = len(seqs_item)
    r_hat_first  = np.full(N, np.nan)
    delta_acc    = np.full(N, np.nan)

    model.encoder.eval()
    device = model.device

    for i in range(N):
        items_i = seqs_item[i]
        corr_i  = seqs_corr[i]
        resp_i  = seqs_resp[i]
        L = len(items_i)
        if L < 40:
            continue
        mid = L // 2

        # First half only
        items_h1 = torch.tensor(items_i[:mid], dtype=torch.long).unsqueeze(0).to(device)
        resp_h1  = torch.tensor(resp_i[:mid].astype(np.int64),
                                dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th_h1, _ = model.encoder.aligned_theta_and_state(items_h1, resp_h1)
        th_h1_np = th_h1.squeeze(0).cpu().numpy().astype(float)
        t_h1     = np.arange(mid, dtype=float)
        r, _, _  = fit_rate(th_h1_np, robust=True, smooth=1, t=t_h1)
        r_hat_first[i] = r

        # delta_acc
        delta_acc[i] = (corr_i[mid:].mean() - corr_i[:mid].mean())

    rho, ci_lo, ci_hi = _bootstrap_spearman(r_hat_first, delta_acc, rng_seed=rng_seed)

    # --- Negative control: permute response ORDER per learner ---------------
    rng = np.random.default_rng(rng_seed + 100)
    r_hat_perm = np.full(N, np.nan)
    for i in range(N):
        items_i = seqs_item[i]
        resp_i  = seqs_resp[i]
        L = len(items_i)
        if L < 40:
            continue
        mid = L // 2
        perm_idx = rng.permutation(L)
        items_perm = items_i[perm_idx]
        resp_perm  = resp_i[perm_idx]
        items_h1 = torch.tensor(items_perm[:mid], dtype=torch.long).unsqueeze(0).to(device)
        resp_h1  = torch.tensor(resp_perm[:mid].astype(np.int64),
                                dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th_h1, _ = model.encoder.aligned_theta_and_state(items_h1, resp_h1)
        th_h1_np = th_h1.squeeze(0).cpu().numpy().astype(float)
        t_h1     = np.arange(mid, dtype=float)
        r, _, _  = fit_rate(th_h1_np, robust=True, smooth=1, t=t_h1)
        r_hat_perm[i] = r

    # Use the same delta_acc (real), only r_hat changes
    rho_neg, ci_lo_neg, ci_hi_neg = _bootstrap_spearman(
        r_hat_perm, delta_acc, rng_seed=rng_seed + 200
    )

    return {
        "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n": int((np.isfinite(r_hat_first) & np.isfinite(delta_acc)).sum()),
        "neg_control_rho": rho_neg,
        "neg_control_ci_lo": ci_lo_neg,
        "neg_control_ci_hi": ci_hi_neg,
    }


# ---------------------------------------------------------------------------
# (b) AFM concurrent validity
# ---------------------------------------------------------------------------

def _afm_slope_per_learner(
    items:  np.ndarray,   # (L,) int
    corr:   np.ndarray,   # (L,) int {0,1}
    parts:  np.ndarray,   # (L,) int 1..7
) -> float:
    """Fit AFM: logistic(correct ~ opp_count_per_KC), return aggregate slope.

    Opportunity count per KC resets to 0 at the start of each learner's
    sequence.  Returns the mean slope across all KCs (parts) that had
    >= 3 observations.
    """
    part_counts: dict[int, int] = {}
    X = np.empty(len(items), dtype=float)
    for t in range(len(items)):
        kc = int(parts[t])
        part_counts[kc] = part_counts.get(kc, 0) + 1
        X[t] = part_counts[kc] - 1   # 0-indexed opportunity

    y = corr.astype(float)
    slopes = []
    for kc in np.unique(parts):
        mask = parts == kc
        if mask.sum() < 3:
            continue
        x_kc = X[mask].reshape(-1, 1)
        y_kc = y[mask]
        if y_kc.std() < 1e-6:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lr = LogisticRegression(max_iter=200, C=1e6, solver="lbfgs")
                lr.fit(x_kc, y_kc)
                slopes.append(float(lr.coef_[0, 0]))
            except Exception:
                pass
    return float(np.mean(slopes)) if slopes else np.nan


def afm_concurrent(
    seqs_item: list[np.ndarray],
    seqs_corr: list[np.ndarray],
    seqs_part: list[np.ndarray],
    r_hat: np.ndarray,
    rng_seed: int = 2,
) -> dict:
    """Spearman(AFM_slope, r_hat) with a part-stratified breakdown."""
    N = len(seqs_item)
    afm_slopes = np.full(N, np.nan)

    for i in range(N):
        if not np.isfinite(r_hat[i]):
            continue
        afm_slopes[i] = _afm_slope_per_learner(
            seqs_item[i], seqs_corr[i], seqs_part[i]
        )

    rho, ci_lo, ci_hi = _bootstrap_spearman(afm_slopes, r_hat, rng_seed=rng_seed)

    # Part-stratified: for each part, compute Spearman of part-specific AFM
    # slope (from learners who visited that part >= 5 times) vs r_hat
    part_results = {}
    for part_id in range(1, 8):
        slopes_p = np.full(N, np.nan)
        for i in range(N):
            if not np.isfinite(r_hat[i]):
                continue
            mask = seqs_part[i] == part_id
            if mask.sum() < 5:
                continue
            slopes_p[i] = _afm_slope_per_learner(
                seqs_item[i][mask],
                seqs_corr[i][mask],
                seqs_part[i][mask],
            )
        rho_p, lo_p, hi_p = _bootstrap_spearman(slopes_p, r_hat, rng_seed=rng_seed+part_id)
        n_p = int((np.isfinite(slopes_p) & np.isfinite(r_hat)).sum())
        part_results[f"part_{part_id}"] = {
            "rho": rho_p, "ci_lo": lo_p, "ci_hi": hi_p, "n": n_p
        }

    return {
        "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n": int((np.isfinite(afm_slopes) & np.isfinite(r_hat)).sum()),
        "part_stratified": part_results,
    }


# ---------------------------------------------------------------------------
# (c) Split-half reliability
# ---------------------------------------------------------------------------

def split_half_reliability(
    seqs_item: list[np.ndarray],
    seqs_resp: list[np.ndarray],
    model: DeepIRTModel,
    rng_seed: int = 3,
) -> dict:
    """Correlate r_hat from odd vs even interactions."""
    import torch

    N = len(seqs_item)
    r_odd  = np.full(N, np.nan)
    r_even = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    for i in range(N):
        items_i = seqs_item[i]
        resp_i  = seqs_resp[i]
        L = len(items_i)
        if L < 40:
            continue

        for half_name, idx in [("odd", np.arange(0, L, 2)),
                                 ("even", np.arange(1, L, 2))]:
            items_h = torch.tensor(items_i[idx], dtype=torch.long).unsqueeze(0).to(device)
            resp_h  = torch.tensor(resp_i[idx].astype(np.int64),
                                   dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                th_h, _ = model.encoder.aligned_theta_and_state(items_h, resp_h)
            th_np = th_h.squeeze(0).cpu().numpy().astype(float)
            r, _, _ = fit_rate(th_np, robust=True, smooth=1,
                               t=np.arange(len(idx), dtype=float))
            if half_name == "odd":
                r_odd[i] = r
            else:
                r_even[i] = r

    rho, ci_lo, ci_hi = _bootstrap_spearman(r_odd, r_even, rng_seed=rng_seed)
    return {
        "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n": int((np.isfinite(r_odd) & np.isfinite(r_even)).sum()),
    }


# ---------------------------------------------------------------------------
# (d) Convergent: aligned vs responsive theta
# ---------------------------------------------------------------------------

def convergent_aligned_vs_responsive(
    seqs_item: list[np.ndarray],
    seqs_resp: list[np.ndarray],
    model: DeepIRTModel,
    r_hat_aligned: np.ndarray,
    rng_seed: int = 4,
) -> dict:
    """Correlate r_hat from aligned theta vs r_hat from raw responsive theta."""
    import torch

    N = len(seqs_item)
    r_responsive = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    for i in range(N):
        items_i = seqs_item[i]
        resp_i  = seqs_resp[i]
        L = len(items_i)
        if L < 20:
            continue
        items_t = torch.tensor(items_i, dtype=torch.long).unsqueeze(0).to(device)
        resp_t  = torch.tensor(resp_i.astype(np.int64),
                               dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th_resp = model.encoder.encode(items_t, resp_t)
        th_np = th_resp.squeeze(0).cpu().numpy().astype(float)
        r, _, _ = fit_rate(th_np, robust=True, smooth=1,
                           t=np.arange(L, dtype=float))
        r_responsive[i] = r

    rho, ci_lo, ci_hi = _bootstrap_spearman(
        r_hat_aligned, r_responsive, rng_seed=rng_seed
    )
    return {
        "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n": int((np.isfinite(r_hat_aligned) & np.isfinite(r_responsive)).sum()),
    }
