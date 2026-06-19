"""metrics.py -- trajectory and rate recovery scoring for E0.

Two recovery targets. The per-step ability correlation asks whether the
recovered trajectory tracks the true one (within each respondent, scale
free). The rate recovery asks whether the curve fit to the recovered
trajectory returns the true rate r across respondents. An oracle that
knows the item parameters and the curve family and fits the rate by
maximum likelihood gives the statistical ceiling, which separates a
recovery failure from a fundamental identifiability limit.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Curve fit to a recovered trajectory (encoder side)
# ---------------------------------------------------------------------------

def _curve(t, A, B, r):
    return A - B * np.exp(-r * t)


def _smooth(y, k):
    """Edge-replicated centered moving average, window 2k+1 (k=0 is a no-op)."""
    if k <= 0:
        return y
    yp = np.pad(y, k, mode="edge")
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(yp, kernel, mode="valid")


def fit_rate(theta_seq, robust: bool = True, smooth: int = 0,
             r_max: float = 1.5, maxfev: int = 10000, t=None):
    """Fit theta(x) = A - B exp(-r x), return (r, A, B).

    The rate is affine-invariant in theta, so an arbitrary theta scale does
    not bias r. ``t`` are the abscissae (the evidence axis); when None they
    default to the step index 0..T-1, which is right for a contiguous
    sequence but not for non-uniform shot counts, where the actual k values
    must be passed. The encoder's per-step theta is noisy, so two options
    harden the readout: ``smooth`` applies a short moving average before
    fitting, and ``robust`` uses the soft-L1 loss so a few noisy steps
    cannot drag the rate. ``r_max`` caps the rate. Returns nan on failure.
    """
    y = np.asarray(theta_seq, dtype=float)
    y = _smooth(y, smooth)
    T = len(y)
    t = np.arange(T, dtype=float) if t is None else np.asarray(t, dtype=float)
    A0, B0 = y[-1], (y[-1] - y[0])
    kw = dict(p0=[A0, B0, 0.2],
              bounds=([-10.0, -20.0, 1e-3], [10.0, 20.0, r_max]), maxfev=maxfev)
    if robust:
        kw.update(method="trf", loss="soft_l1", f_scale=0.3)
    try:
        popt, _ = curve_fit(_curve, t, y, **kw)
        return float(popt[2]), float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan, np.nan


def recover_rates(theta_hat: np.ndarray, **kw) -> np.ndarray:
    """Per-respondent recovered rate from a (N,T) trajectory array."""
    return np.array([fit_rate(theta_hat[i], **kw)[0] for i in range(theta_hat.shape[0])])


def perstep_ability_corr(theta_hat: np.ndarray, theta_true: np.ndarray):
    """Mean within-respondent Pearson corr of recovered vs true theta(t)."""
    cors = []
    for i in range(theta_hat.shape[0]):
        if np.std(theta_hat[i]) > 1e-6 and np.std(theta_true[i]) > 1e-6:
            cors.append(pearsonr(theta_hat[i], theta_true[i])[0])
    cors = np.array(cors)
    return float(np.nanmean(cors)), cors


# ---------------------------------------------------------------------------
# Oracle: rate MLE with known item params and known curve family (ceiling)
# ---------------------------------------------------------------------------

def _gpcm_logprob_np(resp, theta_t, a_i, betas_i):
    """Sum log P(resp_t) under the GPCM for one respondent (numpy)."""
    step = a_i[:, None] * (theta_t[:, None] - betas_i)        # (T, K-1)
    cum = np.cumsum(step, axis=1)
    lognum = np.concatenate([np.zeros((len(theta_t), 1)), cum], axis=1)  # (T, K)
    lognum = lognum - lognum.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(lognum).sum(axis=1))
    ll = lognum[np.arange(len(resp)), resp] - logZ
    return float(ll.sum())


def oracle_rate_mle(resp, a_i, betas_i):
    """Best-case rate, fitting (theta_0, theta_inf, r) by ML with known items."""
    T = len(resp)
    t = np.arange(T, dtype=float)

    def negll(x):
        th0, thinf, r = x
        theta_t = thinf - (thinf - th0) * np.exp(-r * t)
        return -_gpcm_logprob_np(resp, theta_t, a_i, betas_i)

    best = None
    for r0 in (0.1, 0.3):
        res = minimize(
            negll, x0=[-0.5, 1.0, r0],
            bounds=[(-5.0, 5.0), (-5.0, 5.0), (1e-3, 3.0)],
        )
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[2])


def oracle_rates(seq_responses, seq_item_ids, a, betas, max_resp=None):
    """Oracle rate per respondent. Subsample with max_resp to bound time."""
    resp = np.asarray(seq_responses)
    items = np.asarray(seq_item_ids)
    a_np = np.asarray(a)
    betas_np = np.asarray(betas)
    n = resp.shape[0] if max_resp is None else min(max_resp, resp.shape[0])
    out = np.full(n, np.nan)
    for i in range(n):
        idx = items[i]
        out[i] = oracle_rate_mle(resp[i], a_np[idx], betas_np[idx])
    return out


# ---------------------------------------------------------------------------
# Aggregate rate-recovery summary
# ---------------------------------------------------------------------------

def rate_recovery_summary(r_hat: np.ndarray, r_true: np.ndarray) -> dict:
    """Pearson, Spearman, and median absolute error of recovered rates."""
    r_hat = np.asarray(r_hat, dtype=float)
    r_true = np.asarray(r_true, dtype=float)[: len(r_hat)]
    ok = np.isfinite(r_hat) & np.isfinite(r_true)
    if ok.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan"),
                "median_abs_err": float("nan"), "n": int(ok.sum()),
                "frac_fit": float(ok.mean())}
    return {
        "pearson": float(pearsonr(r_hat[ok], r_true[ok])[0]),
        "spearman": float(spearmanr(r_hat[ok], r_true[ok])[0]),
        "median_abs_err": float(np.median(np.abs(r_hat[ok] - r_true[ok]))),
        "n": int(ok.sum()),
        "frac_fit": float(ok.mean()),
    }
