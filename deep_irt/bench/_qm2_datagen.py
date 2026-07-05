"""_qm2_datagen.py -- paper-2 generator family (simple structure, ordinal).

Fresh build per docs/qmirt_paper_plan.md v1.1 (P0/P1). Recovered _qmirt_* are
reference only. Every item is tagged to exactly ONE concept (between-item
multidimensionality); responses are GPCM with K categories.

True person process (per learner i, concept c), the MATCHED form:

    theta[t+1,c] = theta[t,c]
                   + rate[i,c] * gap_c * prac[t,c]                 (own gain)
                   + gap_c * sum_{c'!=c} g_true[c,c'] * prac[t,c'] (transfer)
                   + eps,  eps ~ N(0, sigma_theta^2)               (noise)

    gap_c = clip(ceil - theta_c, 0, inf): GROWTH routes fill the remaining
    mastery gap (matched form with the model; doses ~0.025). INTERFERENCE
    (negative transfer) is gated by the mirror quantity
    gap_dn = clip(theta_c - floor, 0, inf): you can only lose what you have
    built, so decline decelerates toward the floor instead of accelerating
    (2026-07-05 pilot fix -- symmetric ceiling-gating made the interference
    twin saturate at the response floor where both forecast arms tie).

    MEASUREMENT IS STATE-INERT by default (ref_inert=True): reference items
    (the first n_ref per concept, used in measurement blocks and for growth
    scores) do NOT fire practice, in generator and model alike, so "B's only
    route to move is transfer" holds literally and the null twin's B is
    genuinely flat (2026-07-05 review fix; measurement-as-practice becomes a
    later robustness twin).

MISMATCH twins for C1 certification (generate under B, calibrate under the
model's form A):
    'nonmono'  : + forgetting decay toward floor on unpracticed concepts,
                 sigma_theta > 0 (dips; the venue-3 R1 regime)
    'hetrate'  : rate[i,c] ~ LogNormal (heterogeneous learners)
    'forget'   : strong forgetting, weak noise
    'infosched': item administration targets |b_q - theta| (informative
                 schedule; the Debeer-Janssen position confound)

Schedules:
    forecast_schedule(): venue-1 pattern. Conditioning [M0 P0 M1 P1 M2 P2 M3]
    then forecast [P3 M4 P4 M5]. M blocks measure B and U on reference items;
    P blocks practice A. B's only route to move in the forecast window is
    transfer from A.
    mixed_schedule(): random practice over all concepts with a decoupling
    fraction (positivity control) for calibration studies.

Reference items: the first N_REF items of each concept (administered in M
blocks and used for growth scores). Single exposure is NOT enforced within a
schedule (items repeat across blocks); growth scores handle this by windowed
means on reference items -- the R10 single-exposure rule binds REAL data, and
synthetic re-exposure is exact by construction (no retest effect generated).
"""
from __future__ import annotations

import numpy as np

C_DEFAULT = 3  # A=0 (practiced), B=1 (transfer target), U=2 (control)
A, B, U = 0, 1, 2


# ---------------------------------------------------------------- item bank
def make_items(rng, C=C_DEFAULT, j_per_concept=12, K=5,
               alpha_sigma=0.3, b_sigma=1.0, spread=1.6):
    """Single-tagged bank. Returns dict of arrays, length J = C*j_per_concept.

    thresholds d[q, k] (K-1 per item): item location b_q + centered, ordered
    offsets spanning `spread` logits. Ordered by construction (synthetic).
    """
    J = C * j_per_concept
    concept = np.repeat(np.arange(C), j_per_concept)
    alpha = np.exp(rng.normal(0.0, alpha_sigma, size=J))
    b = rng.normal(0.0, b_sigma, size=J)
    offsets = np.linspace(-spread / 2, spread / 2, K - 1)
    d = b[:, None] + offsets[None, :]
    return {"concept": concept, "alpha": alpha, "d": d, "K": K, "J": J,
            "j_per_concept": j_per_concept, "b": b}


def gpcm_probs(alpha_q, d_q, theta):
    """GPCM category probabilities. theta scalar/array, d_q shape (K-1,)."""
    theta = np.asarray(theta, dtype=float)
    steps = alpha_q * (theta[..., None] - d_q)          # (..., K-1)
    csum = np.cumsum(steps, axis=-1)
    logits = np.concatenate([np.zeros_like(csum[..., :1]), csum], axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- schedules
def forecast_schedule(items, n_ref=4, prac_lens=(12, 4, 20),
                      fc_prac1=12, fc_prac2=6, rng=None):
    """Venue-1 pattern with POSITIVITY variation. Returns (seq, marks).

    Conditioning P-block lengths VARY across cycles (prac_lens) so the
    per-cycle A-practice dose decorrelates from B's constant per-cycle
    measurement dose; without this, own-gain, OU pull, and G are collinear
    and the null twin fabricates G (found in the 2026-07-04 pilot; the
    venue-2 co-scheduling confound in schedule form).

    seq: 1-D array of item ids (shared by all learners; rng shuffles within
    blocks if given). marks: dict with T_cond, T_total, index arrays for
    forecast B/U measurement steps and forecast A-practice steps.
    """
    jc = items["j_per_concept"]
    ids = lambda c: np.arange(c * jc, c * jc + n_ref)      # reference items
    prac_pool = np.arange(A * jc + n_ref, A * jc + jc)     # A non-ref items

    def m_block():
        blk = np.concatenate([ids(B), ids(U)])
        if rng is not None:
            rng.shuffle(blk)
        return blk

    def p_block(n):
        reps = np.resize(prac_pool, n)
        if rng is not None:
            rng.shuffle(reps)
        return reps

    cond = []
    for k in range(3):
        cond += [m_block(), p_block(prac_lens[k])]
    cond.append(m_block())
    cond = np.concatenate(cond)
    T_cond = len(cond)

    fc = [p_block(fc_prac1), m_block(), p_block(fc_prac2), m_block()]
    fc = np.concatenate(fc)
    seq = np.concatenate([cond, fc])

    fc_concept = items["concept"][fc]
    fc_idx = np.arange(T_cond, len(seq))
    marks = {
        "T_cond": T_cond, "T_total": len(seq),
        "fc_B": fc_idx[fc_concept == B], "fc_U": fc_idx[fc_concept == U],
        "fc_A_prac": fc_idx[fc_concept == A],
        "ref_ids": {c: ids(c) for c in (A, B, U)},
    }
    return seq, marks


def mixed_schedule(items, T, rng, decouple_frac=1.0):
    """Random practice over all concepts. decouple_frac in [0,1]: fraction of
    steps drawn from single-concept runs (A-only blocks etc.); the remainder
    interleaves uniformly. Used by C1 calibration beds."""
    J = items["J"]
    seq = np.empty(T, dtype=int)
    t = 0
    C = int(items["concept"].max()) + 1
    while t < T:
        if rng.random() < decouple_frac:
            c = rng.integers(C)
            run = min(int(rng.integers(4, 9)), T - t)
            pool = np.where(items["concept"] == c)[0]
            seq[t:t + run] = rng.choice(pool, size=run)
            t += run
        else:
            seq[t] = rng.integers(J)
            t += 1
    return seq


# ---------------------------------------------------------------- generator
def generate(data_seed, kind="matched", C=C_DEFAULT, N=400, K=5,
             j_per_concept=12, g_true=None, schedule="forecast",
             T_mixed=90, decouple_frac=1.0, n_ref=4, ref_inert=True,
             rate_mean=0.035, ceil=2.5, theta0_mu=-1.5, theta0_sd=0.5,
             sigma_theta=0.0, forget=0.0, floor=-2.0, theta0_mu_B=None,
             per_learner_items=False, model_seed_shuffle=True):
    """One dataset. kind selects the generating twin (see module docstring).

    g_true: (C,C) zero-diagonal array or None (null twin). For the canonical
    sparse edge pass g_mat with only [B,A] set.
    Returns dict: items, seq (T,), resp (N,T), theta (N,T,C), marks, config.
    """
    rng = np.random.default_rng(data_seed)
    items = make_items(rng, C=C, j_per_concept=j_per_concept, K=K)
    inert_items = np.zeros(items["J"], dtype=bool)
    if ref_inert:
        for c in range(C):
            inert_items[c * j_per_concept: c * j_per_concept + n_ref] = True

    if schedule == "forecast":
        seq, marks = forecast_schedule(items, n_ref=n_ref, rng=rng)
    else:
        seq = mixed_schedule(items, T_mixed, rng, decouple_frac)
        marks = {"T_cond": T_mixed, "T_total": T_mixed,
                 "ref_ids": {c: np.arange(c * j_per_concept,
                                          c * j_per_concept + 4)
                             for c in range(C)}}
    T = len(seq)

    g = np.zeros((C, C)) if g_true is None else np.asarray(g_true, float)
    assert np.allclose(np.diag(g), 0.0), "g_true must be zero-diagonal"

    if kind == "hetrate":
        rate = np.exp(rng.normal(np.log(rate_mean), 0.5, size=(N, C)))
    else:
        rate = np.full((N, C), rate_mean)
    if kind in ("nonmono", "forget"):
        forget = max(forget, 0.03 if kind == "nonmono" else 0.06)
        sigma_theta = max(sigma_theta, 0.20 if kind == "nonmono" else 0.05)

    theta = np.empty((N, T, C))
    theta[:, 0, :] = rng.normal(theta0_mu, theta0_sd, size=(N, C))
    if theta0_mu_B is not None:            # interference beds start B high
        theta[:, 0, B] = rng.normal(theta0_mu_B, theta0_sd, size=N)

    seq_eff_init = np.tile(seq, (N, 1))
    if per_learner_items and schedule == "forecast":
        # variable sequences over a large bank: the SLOT structure (which
        # step measures B/U vs practices A) is shared, so marks stay valid;
        # each learner draws WHICH A-item fills each practice slot from A's
        # non-reference pool (bank size scales, administered count fixed).
        jc = j_per_concept
        a_pool = np.arange(A * jc + n_ref, A * jc + jc)
        a_slots = np.where((items["concept"][seq] == A)
                           & ~inert_items[seq])[0]
        for i in range(N):
            seq_eff_init[i, a_slots] = rng.choice(a_pool, size=len(a_slots))
    resp = np.empty((N, T), dtype=int)

    seq_eff = seq_eff_init
    if kind == "infosched":
        # informative administration: at each step pick, within the scheduled
        # item's concept, the item whose location is nearest current theta
        concept_of = items["concept"]
        b = items["b"]

    for t in range(T):
        q_t = seq_eff[:, t]
        if kind == "infosched":
            c_t = concept_of[q_t]
            for i in range(N):
                pool = np.where(concept_of == c_t[i])[0]
                q_t[i] = pool[np.argmin(np.abs(b[pool]
                                               - theta[i, t, c_t[i]]))]
            seq_eff[:, t] = q_t
        # responses from current theta (emit BEFORE update)
        for i in range(N):
            q = q_t[i]
            p = gpcm_probs(items["alpha"][q], items["d"][q],
                           theta[i, t, items["concept"][q]])
            resp[i, t] = rng.choice(items["K"], p=p)
        if t + 1 < T:
            prac = np.zeros((N, C))
            live = ~inert_items[q_t]
            prac[np.arange(N)[live], items["concept"][q_t][live]] = 1.0
            gap_up = np.clip(ceil - theta[:, t, :], 0.0, None)
            gap_dn = np.clip(theta[:, t, :] - floor, 0.0, None)
            gain = rate * gap_up * prac
            x = prac @ g.T
            trans = gap_up * np.clip(x, 0.0, None) \
                - gap_dn * np.clip(-x, 0.0, None)
            dec = -forget * (theta[:, t, :] - floor) * (1 - prac) \
                if forget > 0 else 0.0
            noise = rng.normal(0, sigma_theta, size=(N, C)) \
                if sigma_theta > 0 else 0.0
            theta[:, t + 1, :] = theta[:, t, :] + gain + trans + dec + noise

    return {"items": items, "seq": seq, "seq_eff": seq_eff, "resp": resp,
            "theta": theta, "marks": marks, "inert_items": inert_items,
            "config": {"kind": kind, "data_seed": data_seed, "N": N, "C": C,
                       "K": K, "g_true": g.tolist(), "rate_mean": rate_mean,
                       "ceil": ceil, "sigma_theta": sigma_theta,
                       "forget": forget, "schedule": schedule,
                       "decouple_frac": decouple_frac}}


def sparse_g(C=C_DEFAULT, val=0.10, src=A, dst=B):
    g = np.zeros((C, C))
    g[dst, src] = val
    return g


if __name__ == "__main__":
    # calibration self-check: matched twin, forecast schedule
    ds = generate(42, kind="matched", N=50, g_true=sparse_g(val=0.025))
    th = ds["theta"]
    m = ds["marks"]
    tc = m["T_cond"]
    print("T_cond", tc, "T_total", m["T_total"])
    print("theta A rise", th[:, -1, A].mean() - th[:, 0, A].mean())
    print("theta B rise", th[:, -1, B].mean() - th[:, 0, B].mean())
    print("theta U rise", th[:, -1, U].mean() - th[:, 0, U].mean())
    print("theta at T_cond (A,B,U):", th[:, tc, A].mean().round(2),
          th[:, tc, B].mean().round(2), th[:, tc, U].mean().round(2),
          " ceil", ds["config"]["ceil"], " (B must be mid-range)")
    cats, cnt = np.unique(ds["resp"], return_counts=True)
    print("category usage", dict(zip(cats.tolist(),
                                     (cnt / cnt.sum()).round(3).tolist())))
