"""
metrics_bench.py -- write-once prediction + recovery metrics for the bench.

The SAME functions score both engines.  An engine hands back, for its held-out
positions, the predicted category probabilities and the true labels; and for
recovery, its per-item alpha/beta and per-learner (or per-step) theta.  Nothing
here knows which engine produced the numbers.

Prediction
----------
- accuracy        : argmax categorical accuracy on held-out positions.
- qwk             : quadratic weighted kappa (ordinal agreement).
- auc             : binary AUC.  For K>2 we collapse "top half" vs "bottom half"
                    of the ordinal scale into a binary correctness proxy so a
                    single AUC column is comparable across datasets; for K=2 it
                    is the exact 2PL AUC.

Recovery (the metric that matters for measurement)
--------------------------------------------------
- theta : Spearman + Pearson of recovered vs true ability.  Static datasets use
          the per-learner final-step theta vs the constant true theta.  Dynamic
          datasets use the honest within-learner NET DRIFT (theta_end -
          theta_start) recovered vs true, never the pooled level (which is
          inflated by between-learner spread).
- a / b : Spearman + Pearson of recovered vs true item discrimination and the
          flattened step-thresholds.  Sign is aligned to the truth first
          (the latent scale orientation is not identified).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score


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


# ---------------------------------------------------------------------------
# prediction metrics
# ---------------------------------------------------------------------------

def prediction_metrics(probs: np.ndarray, targets: np.ndarray, n_cats: int) -> dict:
    """probs: (M, K) predicted category probabilities; targets: (M,) labels."""
    preds = probs.argmax(axis=1)
    acc = float((preds == targets).mean())

    # QWK over the full ordinal label set 0..K-1.
    labels = list(range(n_cats))
    try:
        qwk = float(cohen_kappa_score(targets, preds, weights="quadratic", labels=labels))
    except Exception:
        qwk = float("nan")

    # AUC.  K=2 -> exact 2PL AUC on P(class 1).  K>2 -> collapse to top-half vs
    # bottom-half correctness proxy so one AUC column is comparable across sets.
    try:
        if n_cats == 2:
            y = targets
            score = probs[:, 1]
        else:
            half = n_cats / 2.0
            y = (targets >= half).astype(int)
            score = probs[:, int(np.ceil(half)):].sum(axis=1)
        auc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    return {"acc": acc, "qwk": qwk, "auc": auc}


# ---------------------------------------------------------------------------
# recovery metrics
# ---------------------------------------------------------------------------

def theta_recovery_static(theta_hat_final, theta_true) -> dict:
    """Per-learner final theta vs constant true theta."""
    return {
        "theta_spearman": _spearman(theta_hat_final, theta_true),
        "theta_pearson": _pearson(theta_hat_final, theta_true),
    }


def theta_recovery_dynamic(theta_hat_traj, theta_true_traj) -> dict:
    """Honest within-learner net-drift recovery for dynamic ability.

    theta_hat_traj, theta_true_traj : (N, T) arrays.
    Net drift = theta[:, -1] - theta[:, 0] per learner; correlate recovered vs
    true. Also reports pooled (inflated) corr for reference.
    """
    hat = np.asarray(theta_hat_traj, dtype=float)
    tru = np.asarray(theta_true_traj, dtype=float)
    hat_drift = hat[:, -1] - hat[:, 0]
    tru_drift = tru[:, -1] - tru[:, 0]
    return {
        "theta_netdrift_spearman": _spearman(hat_drift, tru_drift),
        "theta_netdrift_pearson": _pearson(hat_drift, tru_drift),
        "theta_pooled_pearson": _pearson(hat.ravel(), tru.ravel()),
    }


def item_recovery(a_hat, b_hat, a_true, b_true, seen=None) -> dict:
    """Discrimination and step-threshold recovery.

    a_hat, a_true : (Q,)
    b_hat, b_true : (Q, K-1)
    seen          : optional boolean/int index of items that actually appeared
                    (recovery is only meaningful for items the model saw).
    Sign of a_hat / b_hat is aligned to truth before scoring.
    """
    a_hat = np.asarray(a_hat, dtype=float); a_true = np.asarray(a_true, dtype=float)
    b_hat = np.asarray(b_hat, dtype=float); b_true = np.asarray(b_true, dtype=float)
    if seen is not None:
        a_hat, a_true = a_hat[seen], a_true[seen]
        b_hat, b_true = b_hat[seen], b_true[seen]

    a_hat_s = _align_sign(a_hat, a_true)
    b_hat_s = _align_sign(b_hat.ravel(), b_true.ravel())
    return {
        "a_spearman": _spearman(a_hat_s, a_true),
        "a_pearson": _pearson(a_hat_s, a_true),
        "b_spearman": _spearman(b_hat_s, b_true.ravel()),
        "b_pearson": _pearson(b_hat_s, b_true.ravel()),
    }
