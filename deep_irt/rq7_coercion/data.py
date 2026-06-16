"""data.py -- RQ7 COERCION-ROBUSTNESS data layer.

The research question (real_data_agenda.md:79): does the learned ability/item
scale survive HOW raw behaviour is coerced into a response format?  We take the
SAME raw EdNet behaviour (correctness + response time) for the SAME learners and
items, and coerce it three ways into ordered response formats of growing
resolution:

    K=2  binary correctness                                       (2PL)
    K=3  wrong / correct-slow / correct-fast                      (GPCM K=3)
    K=4  wrong-slow / wrong-fast / correct-slow / correct-fast    (GPCM K=4)

Every level is ordered so that LEVEL 0 is the least ability shown (wrong, slow)
and the top level is the most ability shown (correct, fast).  This keeps the
direction of the latent consistent across coercions, so a Spearman of per-item
GPCM locations (or per-learner abilities) is directly comparable K2 vs K3 vs K4
without any explicit linking (Spearman is monotone-invariant).

Defensible claim under test (NOT "coercion is free").  The collapsing-categories
IRT literature (Andrich; DiStefano et al.; the dichotomization-loss results) says
coarsening bins LOSES scale information and precision and can shift trait
estimates -- coercion is not lossless.  So we test the WEAKER, defensible claim:
the learned scale stays RANK-invariant across coercion schemes even as absolute
precision drops on the coarser bins.  We therefore also report a precision side
(per-coercion split-half reliability of item locations) to show the expected
info-loss honestly.

No leakage.  The RT median thresholds that define the K=3 / K=4 bins are fit on
the TRAIN learner subset ONLY.  Time cuts are global by default (one median over
all train corrects, and a wrong/correct-specific median for K=4) so every item
sees a comparable cut; a per-learner option is provided for robustness.

Validation generators (the "validate after", built in).

    make_synthetic_positive(...)  -- one TRUE 1D IRT latent (known theta, item b),
        emit a graded latent response then coerce it K2/K3/K4 the SAME way.  The
        latent really is shared, so the pipeline SHOULD recover HIGH cross-coercion
        Spearman -> the achievable CEILING.

    scramble_coercion(...)        -- a NEGATIVE control: re-bin on a
        latent-IRRELEVANT signal (a per-interaction random draw), destroying the
        link between the format and the latent.  The pipeline SHOULD recover LOW
        cross-coercion Spearman -> the FLOOR (proves the metric can DETECT
        non-invariance and is not trivially high).

Public API
----------
``CoercedData``                       -- per-coercion sequences + item/learner index.
``load_raw_ednet(...)``               -- raw (correct, rt) records on a shared item/learner set.
``coerce(raw, K, ...)``               -- coerce raw -> CoercedData (K in {2,3,4}).
``scramble_coercion(raw, K, ...)``    -- negative-control coercion on random signal.
``make_synthetic_positive(...)``      -- synthetic raw from one true IRT latent (+ truth).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class RawBehaviour:
    """Raw per-interaction behaviour on a SHARED item/learner set.

    All coercions read from one ``RawBehaviour`` so the item set, learner set,
    and chronological order are identical across K.  Item ids are 1-based LOCAL
    (dense in [1, n_items]); learner ids are the original EdNet student ids.

    Fields
    ------
    records : list of per-learner dicts, each
        student_id : int
        questions  : (T,) int64  1-based local item id
        correct    : (T,) int64  0/1
        rt         : (T,) float64 raw response time (ms, > 0)
        rt_rank    : (T,) float64 per-interaction random signal in [0,1)
                     (latent-irrelevant; used only by the negative control)
    n_items     : int
    train_sids  : set[int]   learner ids whose interactions define the bin cuts
    """
    records: List[Dict]
    n_items: int
    train_sids: set
    meta: Dict = field(default_factory=dict)


@dataclass
class CoercedData:
    """One coercion of the raw behaviour into an ordered K-category format.

    Fields
    ------
    records : list of per-learner dicts, each
        student_id : int
        questions  : (T,) int64  1-based local item id  (SAME as raw)
        responses  : (T,) int64  ordinal response in [0, K-1]
    n_items : int
    K       : int
    label   : str   e.g. "K2", "K3", "K4", "K3_scram"
    cuts    : dict  the exact threshold(s) used (for the audit/report)
    """
    records: List[Dict]
    n_items: int
    K: int
    label: str
    cuts: Dict = field(default_factory=dict)


def _default_csv() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo, "rl", "data", "ednet_raw", "ednet_nrm.csv")


# ---------------------------------------------------------------------------
# Raw loader (shared item/learner set across ALL coercions)
# ---------------------------------------------------------------------------

def load_raw_ednet(
    csv_path: str | None = None,
    n_learners: int = 3000,
    min_item_responses: int = 50,
    min_seq_len: int = 10,
    max_seq_len: int = 200,
    train_frac: float = 0.5,
    seed: int = 0,
) -> RawBehaviour:
    """Load a bounded EdNet subset as raw (correct, rt) behaviour.

    The item set is fixed up front (items with ``>= min_item_responses`` total
    responses among the sampled learners) so that EVERY coercion is fit on the
    same items and learners -- the design requirement that makes item difficulty
    directly comparable across K.

    ``train_frac`` of learners are flagged as the TRAIN subset whose interactions
    define the RT-median bin cuts (no leakage from the held-out learners into the
    thresholds).  All learners are kept in the sequences -- the cuts are simply
    estimated on the train half.

    Returns
    -------
    RawBehaviour
    """
    if csv_path is None:
        csv_path = _default_csv()
    rng = np.random.default_rng(seed)

    df = pd.read_csv(
        csv_path,
        usecols=["student_id", "question_id", "correct", "response_time",
                 "timestamp"],
    )
    df = df[df["response_time"] > 0].copy()

    # --- sample learners (bounded) from those with enough interactions ---
    per_learner = df.groupby("student_id").size()
    eligible = np.array(sorted(per_learner.index[per_learner >= min_seq_len]))
    if n_learners and n_learners < len(eligible):
        pick = rng.permutation(len(eligible))[:n_learners]
        keep_learners = set(eligible[np.sort(pick)].tolist())
    else:
        keep_learners = set(eligible.tolist())
    df = df[df["student_id"].isin(keep_learners)].copy()

    # --- fix the item set: enough total responses among sampled learners ---
    item_counts = df.groupby("question_id").size()
    keep_items = item_counts.index[item_counts >= min_item_responses]
    df = df[df["question_id"].isin(keep_items)].copy()

    # Re-filter learners (some lost items) and assign 1-based LOCAL item ids.
    present_items = np.array(sorted(df["question_id"].unique().tolist()))
    item2local = {q: i + 1 for i, q in enumerate(present_items)}
    n_items = len(present_items)

    # --- train/test learner split for leak-free bin cuts ---
    all_sids = np.array(sorted(df["student_id"].unique().tolist()))
    perm = rng.permutation(len(all_sids))
    n_train = int(round(train_frac * len(all_sids)))
    train_sids = set(all_sids[perm[:n_train]].tolist())

    # --- per-interaction random signal (negative-control bin source) ---
    df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)
    df["rt_rank"] = rng.random(len(df))

    records: List[Dict] = []
    for sid, grp in df.groupby("student_id", sort=False):
        q_ids = grp["question_id"].to_numpy()
        local = np.array([item2local[q] for q in q_ids], dtype=np.int64)
        corr = grp["correct"].to_numpy().astype(np.int64)
        rt = grp["response_time"].to_numpy().astype(np.float64)
        rrank = grp["rt_rank"].to_numpy().astype(np.float64)
        if local.shape[0] > max_seq_len:
            local = local[:max_seq_len]
            corr = corr[:max_seq_len]
            rt = rt[:max_seq_len]
            rrank = rrank[:max_seq_len]
        if local.shape[0] < min_seq_len:
            continue
        records.append({
            "student_id": int(sid),
            "questions": local,
            "correct": corr,
            "rt": rt,
            "rt_rank": rrank,
        })

    return RawBehaviour(
        records=records,
        n_items=n_items,
        train_sids=train_sids,
        meta={
            "n_learners": len(records),
            "n_items": n_items,
            "n_train_learners": len(train_sids & {r["student_id"] for r in records}),
            "min_item_responses": min_item_responses,
            "max_seq_len": max_seq_len,
        },
    )


# ---------------------------------------------------------------------------
# RT-median cut estimation (TRAIN ONLY)
# ---------------------------------------------------------------------------

def _train_rt_arrays(raw: RawBehaviour) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate (correct, rt) over TRAIN learners only."""
    cs, rs = [], []
    for r in raw.records:
        if r["student_id"] in raw.train_sids:
            cs.append(r["correct"])
            rs.append(r["rt"])
    if not cs:
        raise ValueError("no train learners present in records")
    return np.concatenate(cs), np.concatenate(rs)


def _fit_cuts(raw: RawBehaviour) -> Dict[str, float]:
    """Fit the RT-median thresholds on the TRAIN subset only.

    Returns
    -------
    dict with
        median_correct : median RT among TRAIN corrects   (K=3 fast/slow cut)
        median_wrong   : median RT among TRAIN wrongs      (K=4 wrong fast/slow cut)
    """
    corr, rt = _train_rt_arrays(raw)
    m_correct = float(np.median(rt[corr == 1])) if (corr == 1).any() else float(np.median(rt))
    m_wrong = float(np.median(rt[corr == 0])) if (corr == 0).any() else float(np.median(rt))
    return {"median_correct": m_correct, "median_wrong": m_wrong}


# ---------------------------------------------------------------------------
# The coercions (the SAME raw behaviour -> ordered K-category formats)
# ---------------------------------------------------------------------------

def coerce(raw: RawBehaviour, K: int, cuts: Optional[Dict] = None) -> CoercedData:
    """Coerce raw behaviour into an ordered K-category response format.

    K=2 : level = correct                                   (0 wrong, 1 right).
    K=3 : 0 wrong, 1 correct-slow, 2 correct-fast.  Correct split at the TRAIN
          median RT among corrects (fast == rt < median, the more ability-showing
          level).
    K=4 : 0 wrong-slow, 1 wrong-fast, 2 correct-slow, 3 correct-fast.  Ordered by
          empirical ability shown (wrong < correct; slow < fast within each).
          Wrong/correct split each at their own TRAIN median RT.

    Tie handling.  EdNet RT is quantized to ~1s, so ``rt < median`` is used (strict)
    for the FAST level; ties at exactly the median go to the SLOW level.  This is a
    fixed, leak-free rule.

    Returns a ``CoercedData`` with the same item/learner set as ``raw``.
    """
    if K not in (2, 3, 4):
        raise ValueError(f"K must be 2, 3 or 4, got {K}")
    if cuts is None:
        cuts = _fit_cuts(raw)
    m_c = cuts["median_correct"]
    m_w = cuts["median_wrong"]

    records: List[Dict] = []
    for r in raw.records:
        corr = r["correct"]
        rt = r["rt"]
        fast_c = rt < m_c       # fast among (any) -- used for the correct cut
        fast_w = rt < m_w       # fast among (any) -- used for the wrong cut
        if K == 2:
            resp = corr.copy()
        elif K == 3:
            # 0 wrong; 1 correct-slow; 2 correct-fast
            resp = np.zeros_like(corr)
            is_c = corr == 1
            resp[is_c & ~fast_c] = 1
            resp[is_c & fast_c] = 2
        else:  # K == 4
            # 0 wrong-slow, 1 wrong-fast, 2 correct-slow, 3 correct-fast
            resp = np.zeros_like(corr)
            is_w = corr == 0
            is_c = corr == 1
            resp[is_w & ~fast_w] = 0
            resp[is_w & fast_w] = 1
            resp[is_c & ~fast_c] = 2
            resp[is_c & fast_c] = 3
        records.append({
            "student_id": r["student_id"],
            "questions": r["questions"].copy(),
            "responses": resp.astype(np.int64),
        })
    return CoercedData(
        records=records, n_items=raw.n_items, K=K, label=f"K{K}",
        cuts=dict(cuts),
    )


def scramble_coercion(raw: RawBehaviour, K: int, seed: int = 0) -> CoercedData:
    """NEGATIVE control: bin on a latent-IRRELEVANT signal.

    Produces a K-category ordinal response by binning a per-interaction RANDOM
    draw into equal-frequency levels, independent of correctness and RT.  The
    response format therefore carries no information about the latent, so its
    item locations should be NOISE and the cross-coercion Spearman against the
    real coercions (and against a second scramble) should collapse to ~0.  This
    establishes the FLOOR and proves the invariance metric can DETECT
    non-invariance.

    The bin EDGES are fit on the TRAIN subset's random signal only (leak-free,
    mirrors the real cuts).
    """
    if K not in (2, 3, 4):
        raise ValueError(f"K must be 2, 3 or 4, got {K}")
    rng = np.random.default_rng(1000 + seed)
    # Re-draw a fresh random signal so each scramble seed is independent.
    train_vals = []
    redraw: Dict[int, np.ndarray] = {}
    for r in raw.records:
        v = rng.random(r["questions"].shape[0])
        redraw[r["student_id"]] = v
        if r["student_id"] in raw.train_sids:
            train_vals.append(v)
    tv = np.concatenate(train_vals)
    edges = [float(np.quantile(tv, j / K)) for j in range(1, K)]

    records: List[Dict] = []
    for r in raw.records:
        v = redraw[r["student_id"]]
        resp = np.zeros(v.shape[0], dtype=np.int64)
        for e in edges:
            resp += (v >= e).astype(np.int64)
        resp = np.clip(resp, 0, K - 1)
        records.append({
            "student_id": r["student_id"],
            "questions": r["questions"].copy(),
            "responses": resp,
        })
    return CoercedData(
        records=records, n_items=raw.n_items, K=K, label=f"K{K}_scram{seed}",
        cuts={"edges": edges},
    )


# ---------------------------------------------------------------------------
# POSITIVE control: synthetic raw from ONE true 1D IRT latent
# ---------------------------------------------------------------------------

def make_synthetic_positive(
    n_learners: int = 3000,
    n_items: int = 800,
    seq_len: int = 120,
    train_frac: float = 0.5,
    seed: int = 0,
) -> Tuple[RawBehaviour, np.ndarray, np.ndarray]:
    """Generate raw behaviour from ONE true 1D IRT latent, then return it as a
    ``RawBehaviour`` ready to be coerced K2/K3/K4 the SAME way.

    The generative story makes correctness AND response time both functions of a
    single shared per-interaction latent margin ``z = theta - b``:

        P(correct) = sigmoid(1.7 * z)              (2PL-like, a=1.7)
        speed      = sigmoid(z) + noise            (high margin -> fast)
        rt         = exp(mu - kappa * speed)       (fast == small rt)

    so wrong/slow are the low-margin tail and correct/fast the high-margin tail --
    exactly the ordering the coercions assume.  Because every observable is driven
    by the SAME ``z``, coercing at K=2/3/4 yields formats of one shared latent, and
    the pipeline SHOULD recover HIGH cross-coercion Spearman.  This is the CEILING
    and proves the metric is not trivially low.

    Returns
    -------
    (raw, true_theta (n_learners,), true_b (n_items,))
    """
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_learners)              # true ability
    b = rng.standard_normal(n_items)                     # true difficulty
    a_disc = 1.7

    all_sids = np.arange(1, n_learners + 1)
    perm = rng.permutation(n_learners)
    n_train = int(round(train_frac * n_learners))
    train_sids = set(all_sids[perm[:n_train]].tolist())

    records: List[Dict] = []
    for li in range(n_learners):
        q_local = rng.integers(1, n_items + 1, size=seq_len)   # 1-based local
        z = theta[li] - b[q_local - 1]                          # margin
        p_correct = 1.0 / (1.0 + np.exp(-a_disc * z))
        corr = (rng.random(seq_len) < p_correct).astype(np.int64)
        speed = 1.0 / (1.0 + np.exp(-z)) + 0.25 * rng.standard_normal(seq_len)
        # rt: high margin -> high speed -> small rt.  Keep strictly positive.
        rt = np.exp(9.8 - 1.2 * speed)                          # ms-ish scale
        rrank = rng.random(seq_len)
        records.append({
            "student_id": int(all_sids[li]),
            "questions": q_local.astype(np.int64),
            "correct": corr,
            "rt": rt.astype(np.float64),
            "rt_rank": rrank.astype(np.float64),
        })

    raw = RawBehaviour(
        records=records, n_items=n_items, train_sids=train_sids,
        meta={"n_learners": n_learners, "n_items": n_items, "synthetic": True},
    )
    return raw, theta, b
