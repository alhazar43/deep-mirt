"""data.py -- Real EdNet THREE-format data layer for RQ1 format-agnosticism.

Extends ``deep_irt/rq1_ednet_nrm/data.py`` from two response views (binary
accuracy + nominal choice) to THREE, by adding RESPONSE TIME as a genuinely
independent third observable.  The same interleaved per-learner sequences over
the same 4-option EdNet items now carry three labels per interaction:

    accuracy : ``correct`` (1 = right, 0 = wrong)             -- the 2PL view.
    choice   : chosen option a/b/c/d remapped so the CORRECT option is index 0
               and distractors fill 1..K-1 alphabetically.    -- the NRM view.
    log_rt   : log of the response time in milliseconds, cleaned (see below).
                                                              -- the log-normal RT view.

Why response time is the strong test
-------------------------------------
Accuracy and choice are TIED: the nominal choice coarsens to the binary correct
flag, so a model that fits one already constrains the other.  Response time is a
SEPARATE channel of the same interaction: harder items tend to take longer
(a positive difficulty <-> time-intensity link), but speed-accuracy tradeoffs
(fast-wrong vs slow-right) inject real, substantive noise.  Whether item
time-intensity lands on the SAME shared item scale as accuracy difficulty,
above the independent-baseline noise floor, is the headline question.

RT cleaning (documented exactly, it matters)
--------------------------------------------
EdNet ``response_time`` is in milliseconds with a heavy long tail (max ~7.7 h,
clearly idle/abandoned) and a sub-second left tail (reflexive / accidental
clicks).  We:

    1. DROP rows with response_time <= 0 (5112 of 1.86M raw; missing-equivalent).
    2. Take log(response_time).
    3. WINSORIZE log-RT to robust percentiles [lo_q, hi_q] (defaults 0.5% /
       99.5%) computed on the KEPT subset.  Clipping (not dropping) keeps the
       interaction in the accuracy/choice sequence while bounding the absurd
       tails the log-normal model is sensitive to.
    4. STANDARDIZE log-RT to zero mean, unit variance on the kept subset.  The
       log-normal RT decoder expects a roughly Normal log-time scale; the person
       speed latent tau is centered by the model (van der Linden anchor) and the
       item time-intensity beta_i is read on this standardized scale.

The exact winsor bounds and standardization stats are returned in ``rt_stats``
for the report, so the cleaning is fully auditable.

Format partition (FOUR disjoint item groups)
---------------------------------------------
With three formats the held-out-format design needs every single-format group
plus a full bridge:

    ACC-ONLY    : the model sees ONLY accuracy (2PL head).
    CHOICE-ONLY : the model sees ONLY the chosen option (NRM head).
    RT-ONLY     : the model sees ONLY the log response time (RT head).
    ALL-THREE   : the bridge -- all three heads see these positions.

Each single-format group is a held-out test for the OTHER two scales: e.g. an
RT-ONLY item's shared location was shaped only by the RT head, so reading it on
the accuracy/choice scale is a pure cross-format transfer.  The partition is a
per-position FORMAT MASK on the single interleaved sequence; every position
carries all three labels plus three boolean visibility flags.  The encoder runs
over the full sequence (it needs the history); each head's loss is taken only at
its visible positions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

OPTIONS = ("a", "b", "c", "d")
OPT2IDX = {o: i for i, o in enumerate(OPTIONS)}

# Item groups.
G_ALL = 0
G_ACC_ONLY = 1
G_CHOICE_ONLY = 2
G_RT_ONLY = 3


@dataclass
class EdNet3Data:
    """Prepared real-EdNet three-format sequences and the item partition.

    Fields
    ------
    records : list of per-learner dicts, each with
        student_id : int
        questions  : np.ndarray (T,) long  1-based LOCAL item ids
        accuracy   : np.ndarray (T,) long  0/1 correctness
        choice     : np.ndarray (T,) long  0..K-1 chosen option (correct == 0)
        log_rt     : np.ndarray (T,) float standardized log response time
        acc_vis    : np.ndarray (T,) bool  item is accuracy-visible
        cho_vis    : np.ndarray (T,) bool  item is choice-visible
        rt_vis     : np.ndarray (T,) bool  item is rt-visible
    n_items : int
    K       : int
    group   : np.ndarray (n_items,)  G_* per LOCAL item id (index 0 == item 1)
    acc_only_idx / choice_only_idx / rt_only_idx / all_idx : 0-based local idx.
    rt_stats : dict with the exact RT cleaning constants.
    """

    records: List[Dict]
    n_items: int
    K: int
    group: np.ndarray
    acc_only_idx: np.ndarray
    choice_only_idx: np.ndarray
    rt_only_idx: np.ndarray
    all_idx: np.ndarray
    rt_stats: Dict = field(default_factory=dict)


def _default_csv() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo, "rl", "data", "ednet_raw", "ednet_nrm.csv")


def load_ednet3(
    csv_path: str | None = None,
    n_learners: int = 3000,
    min_opt_count: int = 5,
    min_item_responses: int = 30,
    min_seq_len: int = 10,
    max_seq_len: int = 200,
    rt_winsor_lo: float = 0.005,
    rt_winsor_hi: float = 0.995,
    frac_acc_only: float = 0.25,
    frac_choice_only: float = 0.25,
    frac_rt_only: float = 0.25,
    seed: int = 0,
) -> EdNet3Data:
    """Load and prepare a bounded EdNet three-format dataset.

    Parameters mirror ``rq1_ednet_nrm.load_ednet`` plus RT cleaning and a
    four-way item partition.  ``frac_acc_only`` / ``frac_choice_only`` /
    ``frac_rt_only`` are the single-format group fractions; the remainder is
    ALL-THREE (the bridge).

    Returns
    -------
    EdNet3Data
    """
    if csv_path is None:
        csv_path = _default_csv()
    rng = np.random.default_rng(seed)

    df = pd.read_csv(
        csv_path,
        usecols=["student_id", "question_id", "user_answer",
                 "correct_answer", "correct", "response_time", "timestamp"],
    )

    # --- keep only valid a/b/c/d answers/keys and positive response times ---
    df = df[df["user_answer"].isin(OPTIONS) & df["correct_answer"].isin(OPTIONS)]
    df = df[df["response_time"] > 0].copy()

    # --- item filter: all four options observed, adequate per-option counts ---
    opt_counts = (
        df.groupby(["question_id", "user_answer"]).size()
        .unstack(fill_value=0)
    )
    for o in OPTIONS:
        if o not in opt_counts.columns:
            opt_counts[o] = 0
    has_all4 = (opt_counts[list(OPTIONS)] >= min_opt_count).all(axis=1)
    tot = opt_counts[list(OPTIONS)].sum(axis=1)
    keep_items = opt_counts.index[has_all4 & (tot >= min_item_responses)]
    df = df[df["question_id"].isin(keep_items)].copy()

    # --- per-item correct option (a/b/c/d) fixed; remap so correct == 0 ---
    item_key = df.groupby("question_id")["correct_answer"].first().to_dict()

    def _remap(q_id: str, chosen: str) -> int:
        corr = item_key[q_id]
        order = [corr] + [o for o in OPTIONS if o != corr]
        return order.index(chosen)

    # --- sample learners (bounded) from those with enough kept interactions ---
    per_learner = df.groupby("student_id").size()
    eligible = per_learner.index[per_learner >= min_seq_len]
    eligible = np.array(sorted(eligible.tolist()))
    if n_learners and n_learners < len(eligible):
        pick = rng.permutation(len(eligible))[:n_learners]
        keep_learners = set(eligible[np.sort(pick)].tolist())
    else:
        keep_learners = set(eligible.tolist())
    df = df[df["student_id"].isin(keep_learners)].copy()

    # --- RT cleaning on the kept subset: log -> winsorize -> standardize ---
    log_rt_all = np.log(df["response_time"].to_numpy().astype(np.float64))
    lo = float(np.quantile(log_rt_all, rt_winsor_lo))
    hi = float(np.quantile(log_rt_all, rt_winsor_hi))
    log_rt_w = np.clip(log_rt_all, lo, hi)
    mu = float(log_rt_w.mean())
    sd = float(log_rt_w.std())
    sd = sd if sd > 1e-8 else 1.0
    df["log_rt_std"] = (log_rt_w - mu) / sd
    rt_stats = {
        "winsor_lo_q": rt_winsor_lo,
        "winsor_hi_q": rt_winsor_hi,
        "winsor_lo_logrt": lo,
        "winsor_hi_logrt": hi,
        "mean_logrt": mu,
        "std_logrt": sd,
        "n_clipped_low": int((log_rt_all < lo).sum()),
        "n_clipped_high": int((log_rt_all > hi).sum()),
        "n_obs": int(len(df)),
    }

    # --- re-filter items to those still present, assign LOCAL 1-based ids ---
    present_items = np.array(sorted(df["question_id"].unique().tolist()))
    item2local = {q: i + 1 for i, q in enumerate(present_items)}  # 1-based
    n_items = len(present_items)

    # --- item-group assignment (disjoint four-way partition) ---
    perm = rng.permutation(n_items)
    n_acc = int(round(frac_acc_only * n_items))
    n_cho = int(round(frac_choice_only * n_items))
    n_rt = int(round(frac_rt_only * n_items))
    group = np.full(n_items, G_ALL, dtype=np.int64)  # index by local_id - 1
    group[perm[:n_acc]] = G_ACC_ONLY
    group[perm[n_acc:n_acc + n_cho]] = G_CHOICE_ONLY
    group[perm[n_acc + n_cho:n_acc + n_cho + n_rt]] = G_RT_ONLY

    acc_only_idx = np.where(group == G_ACC_ONLY)[0]
    choice_only_idx = np.where(group == G_CHOICE_ONLY)[0]
    rt_only_idx = np.where(group == G_RT_ONLY)[0]
    all_idx = np.where(group == G_ALL)[0]

    # --- build per-learner chronological sequences ---
    df = df.sort_values(["student_id", "timestamp"])
    records: List[Dict] = []
    for sid, grp in df.groupby("student_id", sort=False):
        q_ids = grp["question_id"].to_numpy()
        chosen = grp["user_answer"].to_numpy()
        corr = grp["correct"].to_numpy().astype(np.int64)
        lrt = grp["log_rt_std"].to_numpy().astype(np.float32)
        local = np.array([item2local[q] for q in q_ids], dtype=np.int64)
        choice = np.array([_remap(q, c) for q, c in zip(q_ids, chosen)],
                          dtype=np.int64)
        if local.shape[0] > max_seq_len:
            local = local[:max_seq_len]
            corr = corr[:max_seq_len]
            choice = choice[:max_seq_len]
            lrt = lrt[:max_seq_len]
        if local.shape[0] < min_seq_len:
            continue
        g = group[local - 1]
        acc_vis = (g == G_ACC_ONLY) | (g == G_ALL)
        cho_vis = (g == G_CHOICE_ONLY) | (g == G_ALL)
        rt_vis = (g == G_RT_ONLY) | (g == G_ALL)
        records.append({
            "student_id": int(sid),
            "questions": local,        # 1-based local
            "accuracy": corr,          # 0/1
            "choice": choice,          # 0..K-1 (correct == 0)
            "log_rt": lrt,             # standardized log RT
            "acc_vis": acc_vis,
            "cho_vis": cho_vis,
            "rt_vis": rt_vis,
        })

    return EdNet3Data(
        records=records,
        n_items=n_items,
        K=len(OPTIONS),
        group=group,
        acc_only_idx=acc_only_idx,
        choice_only_idx=choice_only_idx,
        rt_only_idx=rt_only_idx,
        all_idx=all_idx,
        rt_stats=rt_stats,
    )
