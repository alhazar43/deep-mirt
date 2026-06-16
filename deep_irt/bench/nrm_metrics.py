"""nrm_metrics.py -- prediction + recovery metrics for the NRM engine bench.

The SAME functions score both engines (ma-irt-NRM and deep_irt-NRM).  An engine
hands back, for its held-out positions, the predicted OPTION probabilities and
the true option labels; and for recovery, its per-item per-option slopes a_k and
intercepts c_k, the correct-option location beta, and per-learner (or per-step)
theta.  Nothing here knows which engine produced the numbers.

Prediction (nominal -- QWK is meaningless for unordered options)
----------------------------------------------------------------
- acc      : top-1 categorical accuracy on held-out positions.
- nll      : multiclass negative log-likelihood (mean cross-entropy).
- macro_auc: macro-averaged one-vs-rest ROC AUC over the K options.

Recovery (Spearman + Pearson, sign/scale aligned)
-------------------------------------------------
- theta : recovered vs true ability.  Static -> per-learner final-step theta vs
          constant true theta.  Dynamic -> honest within-learner net drift.
- beta  : correct-option location (-c_corr/a_corr), the NRM analogue of 2PL
          difficulty.
- a_k   : per-option slopes, mean-centered (both sides) before scoring so the
          Bock representatives are comparable; sign aligned to truth.
- c_k   : per-option intercepts, mean-centered before scoring; sign aligned.

Both a_k and c_k are flattened over (item, option) after centering.  Sign of
each recovered vector is aligned to its truth first (the latent scale
orientation is not identified).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spearman(x, y) -> float:
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(sp_stats.spearmanr(x, y).correlation)


def _pearson(x, y) -> float:
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(sp_stats.pearsonr(x, y)[0])


def _align_sign(v, ref):
    """Flip v if it is negatively correlated with ref (latent orientation)."""
    v = np.asarray(v, dtype=float); ref = np.asarray(ref, dtype=float)
    r = _pearson(v, ref)
    return -v if (r == r and r < 0) else v


def _center_options(x: np.ndarray) -> np.ndarray:
    """Mean-center across the option axis (last dim), the Bock convention."""
    return x - x.mean(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# prediction metrics
# ---------------------------------------------------------------------------

def prediction_metrics(probs: np.ndarray, targets: np.ndarray, n_options: int) -> dict:
    """probs: (M, K) predicted option probabilities; targets: (M,) option labels."""
    M = len(targets)
    preds = probs.argmax(axis=1)
    acc = float((preds == targets).mean())

    # Multiclass NLL (mean cross-entropy) with a clamp for numerical safety.
    p_true = probs[np.arange(M), targets]
    nll = float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())

    # Macro-averaged one-vs-rest AUC over the K options.  Only options that
    # appear (and do not appear in every position) contribute a defined AUC;
    # average over the defined ones.
    aucs = []
    for k in range(n_options):
        y = (targets == k).astype(int)
        if 0 < y.sum() < M:                     # both classes present
            try:
                aucs.append(roc_auc_score(y, probs[:, k]))
            except Exception:
                pass
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    return {"acc": acc, "nll": nll, "macro_auc": macro_auc}


# ---------------------------------------------------------------------------
# recovery metrics
# ---------------------------------------------------------------------------

def theta_recovery_static(theta_hat_final, theta_true) -> dict:
    return {
        "theta_spearman": _spearman(theta_hat_final, theta_true),
        "theta_pearson": _pearson(theta_hat_final, theta_true),
    }


def theta_recovery_dynamic(theta_hat_traj, theta_true_traj) -> dict:
    """Honest within-learner net-drift recovery for dynamic ability."""
    hat = np.asarray(theta_hat_traj, dtype=float)
    tru = np.asarray(theta_true_traj, dtype=float)
    hat_drift = hat[:, -1] - hat[:, 0]
    tru_drift = tru[:, -1] - tru[:, 0]
    return {
        "theta_netdrift_spearman": _spearman(hat_drift, tru_drift),
        "theta_netdrift_pearson": _pearson(hat_drift, tru_drift),
        "theta_pooled_pearson": _pearson(hat.ravel(), tru.ravel()),
    }


def item_recovery(a_hat, c_hat, beta_hat, a_true, c_true, beta_true,
                  seen=None) -> dict:
    """Per-option slope / intercept and correct-option location recovery.

    a_hat, a_true : (Q, K)   per-option slopes
    c_hat, c_true : (Q, K)   per-option intercepts
    beta_hat, beta_true : (Q,)  correct-option location
    seen          : optional boolean index of items the model actually saw.

    a_k and c_k are mean-centered across options on BOTH sides before scoring
    (the Bock representative), then flattened over (item, option).  Sign of each
    recovered vector is aligned to its truth.
    """
    a_hat = np.asarray(a_hat, dtype=float); a_true = np.asarray(a_true, dtype=float)
    c_hat = np.asarray(c_hat, dtype=float); c_true = np.asarray(c_true, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float); beta_true = np.asarray(beta_true, dtype=float)

    if seen is not None:
        a_hat, a_true = a_hat[seen], a_true[seen]
        c_hat, c_true = c_hat[seen], c_true[seen]
        beta_hat, beta_true = beta_hat[seen], beta_true[seen]

    # Mean-center both sides across the option axis (Bock representative).
    a_hat_c = _center_options(a_hat).ravel()
    a_true_c = _center_options(a_true).ravel()
    c_hat_c = _center_options(c_hat).ravel()
    c_true_c = _center_options(c_true).ravel()

    a_hat_s = _align_sign(a_hat_c, a_true_c)
    c_hat_s = _align_sign(c_hat_c, c_true_c)
    beta_hat_s = _align_sign(beta_hat, beta_true)

    return {
        "a_spearman": _spearman(a_hat_s, a_true_c),
        "a_pearson": _pearson(a_hat_s, a_true_c),
        "c_spearman": _spearman(c_hat_s, c_true_c),
        "c_pearson": _pearson(c_hat_s, c_true_c),
        "beta_spearman": _spearman(beta_hat_s, beta_true),
        "beta_pearson": _pearson(beta_hat_s, beta_true),
    }
