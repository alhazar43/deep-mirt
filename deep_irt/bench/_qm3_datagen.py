"""_qm3_datagen.py -- blueprint-era generator: multidimensional GPCM bank
with Q-matrix cross-loadings and correlated-ability cohorts.

Emission (review-specified, docs/qmirt_blueprint.md v1.1):
    eta_j(z) = sum_c Q[j,c] * A[j,c] * z_c        (compensatory composite)
    category-k logit = k * eta_j - sum_{i<k} b[j,i]   (thresholds on the
    composite scale; K-1 per item, centered per item). Nests the Chapter 0
    unidimensional GPCM at C=1 with d_i = b_i / a.

Bank composition: per concept, n_pure PURE items (single nonzero loading;
the anchors the confirmatory-MIRT identification condition requires); the
remainder are TWO-concept cross-loaders cycling pairs (0,1), (1,2), (2,0).
Loadings lognormal(0, 0.3) on the Q-support. Item location loc_j ~ N(0,1);
thresholds b[j,i] = loc_j * a_mean_j + centered offsets (a_mean_j = the
item's summed loadings, so locations live on each item's composite scale).

Static calibration cohort (Gate C): theta ~ MVN(0, R(r)) with unit
variances (identification constraint), no learning, mixed random
administration over the whole bank.
"""
from __future__ import annotations

import numpy as np


def make_bank(rng, C=3, j_per_concept=12, n_pure=3, K=5,
              load_sigma=0.3, loc_sigma=1.0, spread=1.6):
    """Returns dict with A (J,C) loadings, Q (J,C) 0/1, b (J,K-1),
    primary (J,) primary-concept tag, pure (J,) bool."""
    J = C * j_per_concept
    Q = np.zeros((J, C), dtype=float)
    primary = np.repeat(np.arange(C), j_per_concept)
    pure = np.zeros(J, dtype=bool)
    pairs = [(c, (c + 1) % C) for c in range(C)]
    for c in range(C):
        block = np.arange(c * j_per_concept, (c + 1) * j_per_concept)
        for idx, j in enumerate(block):
            if idx < n_pure:
                Q[j, c] = 1.0
                pure[j] = True
            else:
                c2 = pairs[c][1]
                Q[j, c] = 1.0
                Q[j, c2] = 1.0
    A = Q * np.exp(rng.normal(0.0, load_sigma, size=(J, C)))
    # MDISC comparability: scale multi-concept items so the composite
    # discrimination norm matches pure items (else cross-loaders are
    # twice as steep and category usage goes bimodal).
    n_load = Q.sum(axis=1, keepdims=True)
    A = A / np.sqrt(np.maximum(n_load, 1.0))
    loc = rng.normal(0.0, loc_sigma, size=J)
    offsets = np.linspace(-spread / 2, spread / 2, K - 1)
    # thresholds on the composite scale; per-item mean equals loc * <a>
    a_mean = A.sum(axis=1)                     # scale locs to composite
    b = (loc * a_mean)[:, None] + offsets[None, :]
    return {"A": A, "Q": Q, "b": b, "K": K, "J": J, "C": C,
            "primary": primary, "pure": pure,
            "j_per_concept": j_per_concept, "n_pure": n_pure}


def mgpcm_probs(a_vec, b_row, z):
    """Category probabilities for one item. a_vec (C,), b_row (K-1,),
    z (..., C). Returns (..., K)."""
    z = np.asarray(z, dtype=float)
    eta = z @ np.asarray(a_vec, dtype=float)           # (...,)
    steps = eta[..., None] - np.asarray(b_row)          # (..., K-1)
    csum = np.cumsum(steps, axis=-1)
    logits = np.concatenate([np.zeros_like(csum[..., :1]), csum], axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def corr_matrix(C, r):
    R = np.full((C, C), r, dtype=float)
    np.fill_diagonal(R, 1.0)
    return R


def generate_static_cohort(data_seed, C=3, j_per_concept=12, n_pure=3,
                           K=5, N=400, T=60, r=0.3):
    """Static (no-learning) calibration cohort on a cross-loading bank.
    theta ~ MVN(0, R(r)), unit variances. Returns dict."""
    rng = np.random.default_rng(data_seed)
    bank = make_bank(rng, C=C, j_per_concept=j_per_concept, n_pure=n_pure,
                     K=K)
    L = np.linalg.cholesky(corr_matrix(C, r))
    theta = rng.standard_normal(size=(N, C)) @ L.T
    seq = rng.integers(0, bank["J"], size=(N, T))
    resp = np.empty((N, T), dtype=int)
    for i in range(N):
        for t in range(T):
            j = seq[i, t]
            p = mgpcm_probs(bank["A"][j], bank["b"][j], theta[i])
            resp[i, t] = rng.choice(K, p=p)
    return {"bank": bank, "theta": theta, "seq": seq, "resp": resp,
            "config": {"data_seed": data_seed, "C": C, "N": N, "T": T,
                       "r": r, "n_pure": n_pure,
                       "j_per_concept": j_per_concept, "K": K}}


if __name__ == "__main__":
    ds = generate_static_cohort(42, n_pure=3, r=0.3, N=200, T=40)
    bank = ds["bank"]
    cats, cnt = np.unique(ds["resp"], return_counts=True)
    print("J", bank["J"], "pure/concept",
          int(bank["pure"].sum() / bank["C"]),
          "cross-loaders", int((~bank["pure"]).sum()))
    print("category usage",
          dict(zip(cats.tolist(), (cnt / cnt.sum()).round(3).tolist())))
    print("theta corr\n", np.corrcoef(ds["theta"].T).round(2))
    cross = ~bank["pure"]
    ratios = []
    for j in np.where(cross)[0]:
        c_on = np.where(bank["Q"][j] > 0)[0]
        ratios.append(bank["A"][j, c_on[0]] / bank["A"][j, c_on[1]])
    print("cross-loading ratio spread (log sd):",
          float(np.std(np.log(ratios))).__round__(3))
