"""data.py -- Real EdNet binary-vs-NRM data layer for RQ1 format-agnosticism.

Loads ``rl/data/ednet_raw/ednet_nrm.csv`` and prepares the interleaved
per-learner sequences plus the BINARY-ONLY / NRM-ONLY / BOTH item partition
needed by the cross-format transfer experiment.

Two response VIEWS of the same interaction, both 0-based:

    binary  : ``correct`` (1 = right, 0 = wrong)          -- the 2PL view.
    nominal : chosen option a/b/c/d remapped so the CORRECT option is index 0
              and distractors fill 1..K-1 in their alphabetical order.        -- the NRM view.

Remapping the correct option to a fixed index 0 makes ``correct_option`` uniform
across the whole item bank, so the NRM head's correct-option LOCATION is exactly
the shared difficulty d_i and is directly comparable to the 2PL difficulty.

Item filtering
--------------
Keep items with all four options observed and an adequate per-option count, so
the NRM head has signal on every option.  Keep learners with a minimum sequence
length so the encoder has history.

Format partition
----------------
Named (kept) items are split into three disjoint groups by a fixed seed:

    BINARY-ONLY : the model sees ONLY their right/wrong (2PL head); the NRM head
                  never sees these positions.
    NRM-ONLY    : the model sees ONLY their chosen option (NRM head); the 2PL
                  head never sees these positions.
    BOTH        : the bridge -- both heads see these positions.

The partition is applied as a per-position FORMAT MASK on the single interleaved
sequence: every position carries its (binary, nominal) labels, plus two boolean
flags ``bin_visible`` / ``nrm_visible`` derived from the item's group.  The
encoder always runs over the full sequence (it needs the history), but each
head's loss is taken only at its visible positions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

OPTIONS = ("a", "b", "c", "d")
OPT2IDX = {o: i for i, o in enumerate(OPTIONS)}

# Item groups.
G_BINARY_ONLY = 1
G_NRM_ONLY = 2
G_BOTH = 0


@dataclass
class EdNetData:
    """Prepared real-EdNet sequences and the item partition.

    Fields
    ------
    records : list of per-learner dicts, each with
        student_id : int (1-based, arbitrary)
        questions  : LongTensor-ready np.ndarray (T,)  1-based LOCAL item ids
        binary     : np.ndarray (T,)  0/1 correctness
        nominal    : np.ndarray (T,)  0..K-1 chosen option (correct remapped to 0)
        bin_vis    : np.ndarray (T,)  bool, item is binary-visible
        nrm_vis    : np.ndarray (T,)  bool, item is nrm-visible
    n_items : int       -- LOCAL item bank size (ids 1..n_items)
    K       : int       -- number of options (4)
    group   : np.ndarray (n_items,)  G_* per LOCAL item id (index 0 == item 1)
    binary_only_idx / nrm_only_idx / both_idx : np.ndarray of 0-based local item
        indices (i.e. item_id - 1) for each group.
    """

    records: List[Dict]
    n_items: int
    K: int
    group: np.ndarray
    binary_only_idx: np.ndarray
    nrm_only_idx: np.ndarray
    both_idx: np.ndarray


def _default_csv() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo, "rl", "data", "ednet_raw", "ednet_nrm.csv")


def load_ednet(
    csv_path: str | None = None,
    n_learners: int = 3000,
    min_opt_count: int = 5,
    min_item_responses: int = 30,
    min_seq_len: int = 10,
    max_seq_len: int = 200,
    frac_binary_only: float = 1.0 / 3.0,
    frac_nrm_only: float = 1.0 / 3.0,
    seed: int = 0,
) -> EdNetData:
    """Load and prepare a bounded EdNet binary-vs-NRM dataset.

    Parameters
    ----------
    csv_path : path to ``ednet_nrm.csv`` (default: rl/data/ednet_raw/).
    n_learners : number of learners to keep (bounded run).  Learners are
        sampled by a fixed seed AFTER item filtering.
    min_opt_count : minimum observations of each option for an item to be kept
        (ensures the NRM head sees every option).
    min_item_responses : minimum total responses for an item to be kept.
    min_seq_len : minimum kept-item interactions for a learner to be kept.
    max_seq_len : truncate each learner to their FIRST ``max_seq_len`` kept
        interactions (chronological) to bound the encoder.
    frac_binary_only / frac_nrm_only : item-group fractions; the remainder is
        BOTH (the bridge).
    seed : RNG seed for learner sampling and item-group assignment.

    Returns
    -------
    EdNetData
    """
    if csv_path is None:
        csv_path = _default_csv()
    rng = np.random.default_rng(seed)

    df = pd.read_csv(
        csv_path,
        usecols=["student_id", "question_id", "user_answer",
                 "correct_answer", "correct", "timestamp"],
    )

    # --- keep only valid a/b/c/d answers and keys ---
    df = df[df["user_answer"].isin(OPTIONS) & df["correct_answer"].isin(OPTIONS)]

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

    # --- per-item correct option (a/b/c/d) is fixed; build option remap ---
    # nominal index = position of chosen option in an item-specific ordering
    # where the correct option is FIRST (index 0) and distractors follow in
    # alphabetical order.  This makes correct_option == 0 for every item.
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

    # --- re-filter items to those still present, assign LOCAL 1-based ids ---
    present_items = np.array(sorted(df["question_id"].unique().tolist()))
    item2local = {q: i + 1 for i, q in enumerate(present_items)}  # 1-based
    n_items = len(present_items)

    # --- item-group assignment (disjoint partition) ---
    perm = rng.permutation(n_items)
    n_bin = int(round(frac_binary_only * n_items))
    n_nrm = int(round(frac_nrm_only * n_items))
    group = np.full(n_items, G_BOTH, dtype=np.int64)  # index by local_id - 1
    group[perm[:n_bin]] = G_BINARY_ONLY
    group[perm[n_bin:n_bin + n_nrm]] = G_NRM_ONLY

    binary_only_idx = np.where(group == G_BINARY_ONLY)[0]
    nrm_only_idx = np.where(group == G_NRM_ONLY)[0]
    both_idx = np.where(group == G_BOTH)[0]

    # --- build per-learner chronological sequences ---
    df = df.sort_values(["student_id", "timestamp"])
    records: List[Dict] = []
    for sid, grp in df.groupby("student_id", sort=False):
        q_ids = grp["question_id"].to_numpy()
        chosen = grp["user_answer"].to_numpy()
        corr = grp["correct"].to_numpy().astype(np.int64)
        local = np.array([item2local[q] for q in q_ids], dtype=np.int64)
        nominal = np.array([_remap(q, c) for q, c in zip(q_ids, chosen)],
                           dtype=np.int64)
        if local.shape[0] > max_seq_len:
            local = local[:max_seq_len]
            chosen = chosen[:max_seq_len]
            corr = corr[:max_seq_len]
            nominal = nominal[:max_seq_len]
        if local.shape[0] < min_seq_len:
            continue
        g = group[local - 1]
        bin_vis = (g == G_BINARY_ONLY) | (g == G_BOTH)
        nrm_vis = (g == G_NRM_ONLY) | (g == G_BOTH)
        records.append({
            "student_id": int(sid),
            "questions": local,         # 1-based local
            "binary": corr,             # 0/1
            "nominal": nominal,         # 0..K-1 (correct == 0)
            "bin_vis": bin_vis,
            "nrm_vis": nrm_vis,
        })

    return EdNetData(
        records=records,
        n_items=n_items,
        K=len(OPTIONS),
        group=group,
        binary_only_idx=binary_only_idx,
        nrm_only_idx=nrm_only_idx,
        both_idx=both_idx,
    )
