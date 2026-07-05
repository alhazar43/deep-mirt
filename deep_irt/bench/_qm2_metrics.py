"""_qm2_metrics.py -- estimands and aggregation for the paper-2 campaign.

Growth scores are the observable dual (plan v1.1): expected/observed score on
fixed reference items, reported in PROPORTION-OF-MAXIMUM units (score/(K-1)),
per learner first, population second. Latent recoveries are internal gates.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from _qm2_datagen import gpcm_probs


# ------------------------------------------------------------- basic stats
def spearman(a, b):
    r = spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel()).statistic
    return float(r)


def seed_agg(values):
    """mean, sd, positive-count over seed cells (non-finite dropped, e.g.
    within-learner correlation on a flat null-B trajectory)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "sd": None, "n": 0, "n_pos": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1
            else 0.0, "n": int(len(v)), "n_pos": int((v > 0).sum())}


# --------------------------------------------------------------- recovery
# Under simple structure each concept carries its own location/scale gauge
# (z0[:,c]+a_c vs d+a_c is likelihood-invariant), so all recovery metrics
# center WITHIN concept (2026-07-05 review fix); global centering lets gauge
# masquerade as item pathology.
def _center_wc(x, concept):
    x = np.asarray(x, dtype=float).copy()
    for c in np.unique(concept):
        x[concept == c] -= x[concept == c].mean()
    return x


def item_recovery(log_alpha_hat, items):
    """Within-concept-centered rank recovery + mean per-concept slope."""
    con = items["concept"]
    la_hat = _center_wc(log_alpha_hat, con)
    la_true = _center_wc(np.log(items["alpha"]), con)
    slopes = [float(np.polyfit(la_true[con == c], la_hat[con == c], 1)[0])
              for c in np.unique(con)]
    return {"alpha_spearman": spearman(la_hat, la_true),
            "alpha_slope": float(np.mean(slopes))}


def d_recovery(d_hat, items):
    con = items["concept"]
    loc_hat = _center_wc(np.asarray(d_hat).mean(axis=1), con)
    loc_true = _center_wc(items["d"].mean(axis=1), con)
    return {"d_loc_spearman": spearman(loc_hat, loc_true),
            "d_loc_bias_sd": float((loc_hat - loc_true).std())}


def position_bias(d_hat, items, seq_eff):
    """Debeer-Janssen detector: correlate an item's REALIZED mean
    administration position (pooled over learners from seq_eff, so the
    informative-schedule twin is visible) with its within-concept location
    bias. Motion laundered into items = later-administered items look
    easier."""
    seq_eff = np.atleast_2d(np.asarray(seq_eff))
    J = items["J"]
    T = seq_eff.shape[1]
    pos_sum = np.zeros(J)
    pos_cnt = np.zeros(J)
    tgrid = np.tile(np.arange(T), (seq_eff.shape[0], 1))
    np.add.at(pos_sum, seq_eff.ravel(), tgrid.ravel())
    np.add.at(pos_cnt, seq_eff.ravel(), 1.0)
    ok = pos_cnt > 0
    pos = np.where(ok, pos_sum / np.maximum(pos_cnt, 1), np.nan)
    con = items["concept"]
    bias = _center_wc(np.asarray(d_hat).mean(axis=1), con) \
        - _center_wc(items["d"].mean(axis=1), con)
    return {"pos_bias_r": spearman(pos[ok], bias[ok])}


def within_learner(z_hat, theta_true, concept):
    """Mean over learners of Spearman(z_hat[i,:,c], theta[i,:,c])."""
    N = z_hat.shape[0]
    rs = [spearman(z_hat[i, :, concept], theta_true[i, :, concept])
          for i in range(N)]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs))


# ------------------------------------------------------------ growth score
def expected_score(items, q_ids, z_c):
    """E[X]/(K-1) for items q_ids at ability values z_c (broadcast)."""
    K = items["K"]
    es = []
    for q in np.asarray(q_ids).ravel():
        p = gpcm_probs(items["alpha"][q], items["d"][q], z_c)
        es.append((p * np.arange(K)).sum(-1))
    return np.stack(es, axis=0).mean(axis=0) / (K - 1)


def growth_scores(ds, z_hat, concept, early, late):
    """Observed vs model-predicted growth score on concept's reference items.

    early/late: step-index arrays of measurement steps for this concept.
    Observed: mean response (prop-of-max) on those steps. Predicted: expected
    score under FROZEN fitted items at the fitted state, same steps.
    Returns per-learner arrays and group means.
    """
    items, resp, seq_eff = ds["items"], ds["resp"], ds["seq_eff"]
    K = items["K"]

    def obs(idx):
        vals = np.array([resp[:, t] for t in idx], dtype=float)  # (n, N)
        return vals.mean(axis=0) / (K - 1)

    def pred(idx, alpha_hat, d_hat):
        out = np.zeros((len(idx), resp.shape[0]))
        for j, t in enumerate(idx):
            for i in range(resp.shape[0]):
                q = seq_eff[i, t]
                p = gpcm_probs(alpha_hat[q], d_hat[q], z_hat[i, t, concept])
                out[j, i] = (p * np.arange(K)).sum() / (K - 1)
        return out.mean(axis=0)

    return obs, pred


def obs_pred_growth(ds, z_hat, alpha_hat, d_hat, concept, early, late):
    obs, pred = growth_scores(ds, z_hat, concept, early, late)
    o_e, o_l = obs(early), obs(late)
    p_e, p_l = pred(early, alpha_hat, d_hat), pred(late, alpha_hat, d_hat)
    return {"obs_rise": float((o_l - o_e).mean()),
            "pred_rise": float((p_l - p_e).mean()),
            "gap": float(abs((o_l - o_e).mean() - (p_l - p_e).mean())),
            "per_learner_obs": (o_l - o_e), "per_learner_pred": (p_l - p_e)}


# ----------------------------------------------------------- forecast gaps
def forecast_gaps(per_noG, per_withG, marks):
    """per_*: per-step NLL arrays (N, T). active gap = noG - withG NLL,
    positive = with-G wins. PARAMETER ORDER: (noG, withG) -- matches the
    p1b call site; a swapped order silently sign-flips every gap (caught by
    the 2026-07-05 adversarial review; the earlier pilots' NLL gaps were
    inverted)."""
    fb, fu = marks["fc_B"], marks["fc_U"]
    gB = float(per_noG[:, fb].mean() - per_withG[:, fb].mean())
    gU = float(per_noG[:, fu].mean() - per_withG[:, fu].mean())
    return {"active_gap_B": gB, "active_gap_U": gU}
