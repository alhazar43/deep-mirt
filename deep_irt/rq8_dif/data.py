"""data.py -- RQ8 INVARIANCE / DIF data layer.

Research question (real_data_agenda.md:82): do anchor-linked item parameters
REPLICATE across learner subpopulations?  I.e. is the learned item scale
SUBPOPULATION-INVARIANT (no differential item functioning, DIF), or are item
difficulties population-specific artifacts?

The whole design rests on ONE raw EdNet behaviour (binary correctness, K=2 / 2PL
-- the simplest and most standard format for a DIF check) on a SHARED item set.
We then split the LEARNERS three ways and fit item difficulty INDEPENDENTLY per
subgroup.  Because both subgroups answer the SAME items (restricted to a common
anchor set answered by enough learners in BOTH groups), per-item difficulties are
directly comparable AFTER a mean/sigma linking onto one metric.

The three splits
----------------
1. RANDOM-HALF PLACEBO (the control, run FIRST): learners split 50/50 at random.
   Item difficulty SHOULD replicate -- this is the NO-DIF baseline and the
   reliability floor.  Any divergence here is pure sampling + estimation noise.

2. ABILITY: top vs bottom tercile by per-learner mean correctness (drop the
   middle tercile for contrast).  Classic DIF axis -- does an item rank harder
   for low-ability learners than for high-ability learners?

3. ACTIVITY: long- vs short-history learners (top vs bottom tercile by sequence
   length).  Does an item's difficulty depend on how much practice the learner
   has?  (A PART-engagement alternative is provided too.)

Linking lives in the runner; this layer only produces the per-group record lists
and the shared item index so the runner can fit + link + compare.

Containers
----------
``RawBinary``   -- per-learner binary records on a shared item set (+ per-learner
                   summaries used by the meaningful splits).
``GroupSplit``  -- one named split into (group_a_records, group_b_records) with
                   labels and the learner-level summary that defined it.

Public API
----------
``load_raw_binary(...)``           -- bounded EdNet subset as binary records.
``split_random_half(raw, seed)``   -- the placebo control.
``split_ability(raw)``             -- top vs bottom tercile by mean correctness.
``split_activity(raw)``            -- long vs short history (seq-len terciles).
``split_part(raw, part_hi, part_lo)`` -- dominant-part engagement alternative.
``make_synthetic_nodif(...)``      -- positive control: one true 2PL latent, NO
                                      DIF by construction (achievable ceiling).
``make_synthetic_dif(...)``        -- negative control: KNOWN DIF injected on a
                                      subset of items (the metric must detect it).
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
class RawBinary:
    """Raw per-learner binary behaviour on a SHARED item set.

    Item ids are 1-based LOCAL (dense in [1, n_items]); learner ids are the
    original EdNet student ids.

    Fields
    ------
    records : list of per-learner dicts, each
        student_id : int
        questions  : (T,) int64  1-based local item id
        responses  : (T,) int64  0/1 correctness (the K=2 response)
    n_items : int
    summary : dict[int -> dict]  per-learner summary keyed by student_id:
        mean_correct : float   per-learner accuracy (ability proxy)
        seq_len      : int     history length (activity proxy)
        dom_part     : int     most-answered TOEIC part (1..7)
    """
    records: List[Dict]
    n_items: int
    summary: Dict[int, Dict] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


@dataclass
class GroupSplit:
    """One named split of the learners into two subpopulations.

    Fields
    ------
    name        : str   "placebo" | "ability" | "activity" | "part"
    label_a     : str   human label for group A (e.g. "rand-A", "high-ability")
    label_b     : str   human label for group B
    records_a   : list of binary records (same shape as RawBinary.records)
    records_b   : list of binary records
    note        : str   how the split was made (for the audit/report)
    """
    name: str
    label_a: str
    label_b: str
    records_a: List[Dict]
    records_b: List[Dict]
    note: str = ""


def _default_csv() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo, "rl", "data", "ednet_raw", "ednet_nrm.csv")


# ---------------------------------------------------------------------------
# Raw loader (shared item set; per-learner summaries for the splits)
# ---------------------------------------------------------------------------

def load_raw_binary(
    csv_path: str | None = None,
    n_learners: int = 4000,
    min_item_responses: int = 50,
    min_seq_len: int = 20,
    max_seq_len: int = 200,
    seed: int = 0,
) -> RawBinary:
    """Load a bounded EdNet subset as binary (correct) behaviour + summaries.

    The item set is fixed up front (items with ``>= min_item_responses`` total
    responses among the sampled learners) so every subgroup is fit on the SAME
    items -- the requirement that makes item difficulty comparable across groups.
    ``min_seq_len`` is set higher than RQ7 (default 20) because each DIF subgroup
    is only HALF (or a third) of the learners, so per-item counts must survive
    the split; longer minimum histories keep the anchor set populated in both
    subgroups.

    Per-learner summaries (mean correctness, sequence length, dominant part) are
    computed on the FULL kept history and drive the meaningful splits.

    Returns
    -------
    RawBinary
    """
    if csv_path is None:
        csv_path = _default_csv()
    rng = np.random.default_rng(seed)

    df = pd.read_csv(
        csv_path,
        usecols=["student_id", "question_id", "correct", "timestamp", "part"],
    )

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

    # Assign 1-based LOCAL item ids over the surviving items.
    present_items = np.array(sorted(df["question_id"].unique().tolist()))
    item2local = {q: i + 1 for i, q in enumerate(present_items)}
    n_items = len(present_items)

    df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

    records: List[Dict] = []
    summary: Dict[int, Dict] = {}
    for sid, grp in df.groupby("student_id", sort=False):
        q_ids = grp["question_id"].to_numpy()
        local = np.array([item2local[q] for q in q_ids], dtype=np.int64)
        corr = grp["correct"].to_numpy().astype(np.int64)
        parts = grp["part"].to_numpy().astype(np.int64)
        if local.shape[0] > max_seq_len:
            local = local[:max_seq_len]
            corr = corr[:max_seq_len]
            parts = parts[:max_seq_len]
        if local.shape[0] < min_seq_len:
            continue
        records.append({
            "student_id": int(sid),
            "questions": local,
            "responses": corr,
        })
        # Dominant part = the most-answered TOEIC part in this (truncated) history.
        vals, cnts = np.unique(parts, return_counts=True)
        dom_part = int(vals[int(np.argmax(cnts))])
        summary[int(sid)] = {
            "mean_correct": float(corr.mean()),
            "seq_len": int(local.shape[0]),
            "dom_part": dom_part,
        }

    return RawBinary(
        records=records,
        n_items=n_items,
        summary=summary,
        meta={
            "n_learners": len(records),
            "n_items": n_items,
            "min_item_responses": min_item_responses,
            "max_seq_len": max_seq_len,
            "min_seq_len": min_seq_len,
        },
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _records_for(raw: RawBinary, sids: set) -> List[Dict]:
    """Subset of raw.records whose student_id is in ``sids`` (copies)."""
    out = []
    for r in raw.records:
        if r["student_id"] in sids:
            out.append({
                "student_id": r["student_id"],
                "questions": r["questions"].copy(),
                "responses": r["responses"].copy(),
            })
    return out


def split_random_half(raw: RawBinary, seed: int = 0) -> GroupSplit:
    """RANDOM-HALF PLACEBO: split learners 50/50 at random.

    The NO-DIF baseline + reliability floor.  Item difficulties should replicate
    across the two halves; any |Δb| here is pure sampling + estimation noise.
    """
    rng = np.random.default_rng(2000 + seed)
    sids = np.array([r["student_id"] for r in raw.records])
    perm = rng.permutation(len(sids))
    half = len(sids) // 2
    set_a = set(sids[perm[:half]].tolist())
    set_b = set(sids[perm[half:2 * half]].tolist())
    return GroupSplit(
        name="placebo",
        label_a="rand-A", label_b="rand-B",
        records_a=_records_for(raw, set_a),
        records_b=_records_for(raw, set_b),
        note=f"random 50/50 of {len(sids)} learners (seed={seed})",
    )


def matched_random_split(
    raw: RawBinary, n_a: int, n_b: int, name: str, seed: int = 7,
) -> GroupSplit:
    """A size-matched RANDOM placebo: two random learner groups of EXACTLY
    ``n_a`` and ``n_b`` learners drawn from the full pool.

    The correct sampling-noise floor for a meaningful split whose groups have
    those sizes.  For axes only weakly correlated with item-level ability range
    (activity, dominant part) this matched placebo IS the right reference; for the
    ABILITY split it bounds the size/sampling component but does NOT reproduce the
    range-restriction noise intrinsic to ability grouping (read the ability row
    with that caveat).
    """
    rng = np.random.default_rng(7000 + seed)
    sids = np.array([r["student_id"] for r in raw.records])
    n_a = min(n_a, len(sids)); n_b = min(n_b, len(sids) - n_a)
    perm = rng.permutation(len(sids))
    set_a = set(sids[perm[:n_a]].tolist())
    set_b = set(sids[perm[n_a:n_a + n_b]].tolist())
    return GroupSplit(
        name=name,
        label_a="match-A", label_b="match-B",
        records_a=_records_for(raw, set_a),
        records_b=_records_for(raw, set_b),
        note=f"size-matched random placebo ({n_a} vs {n_b} learners, seed={seed})",
    )


def _tercile_split(
    raw: RawBinary, key: str, name: str, label_hi: str, label_lo: str,
) -> GroupSplit:
    """Split learners by a per-learner summary ``key`` into top vs bottom tercile
    (the middle tercile is DROPPED for contrast).  Group A = HIGH, B = LOW."""
    sids = np.array([r["student_id"] for r in raw.records])
    vals = np.array([raw.summary[int(s)][key] for s in sids], dtype=np.float64)
    lo_cut = np.quantile(vals, 1.0 / 3.0)
    hi_cut = np.quantile(vals, 2.0 / 3.0)
    set_hi = set(sids[vals >= hi_cut].tolist())
    set_lo = set(sids[vals <= lo_cut].tolist())
    return GroupSplit(
        name=name,
        label_a=label_hi, label_b=label_lo,
        records_a=_records_for(raw, set_hi),
        records_b=_records_for(raw, set_lo),
        note=(f"{key} terciles: HIGH>={hi_cut:.3f} ({len(set_hi)} learners) vs "
              f"LOW<={lo_cut:.3f} ({len(set_lo)} learners); middle dropped"),
    )


def split_ability(raw: RawBinary) -> GroupSplit:
    """ABILITY split: top vs bottom tercile by per-learner mean correctness."""
    return _tercile_split(raw, "mean_correct", "ability",
                          "high-ability", "low-ability")


def split_activity(raw: RawBinary) -> GroupSplit:
    """ACTIVITY split: top vs bottom tercile by history length (seq_len)."""
    return _tercile_split(raw, "seq_len", "activity",
                          "long-history", "short-history")


def split_part(raw: RawBinary, part_hi: int, part_lo: int) -> GroupSplit:
    """PART-engagement alternative: learners whose dominant part is ``part_hi``
    vs ``part_lo``.  An optional alternative to the activity split when the part
    structure is the more interesting axis."""
    sids = np.array([r["student_id"] for r in raw.records])
    set_hi = set(s for s in sids.tolist() if raw.summary[int(s)]["dom_part"] == part_hi)
    set_lo = set(s for s in sids.tolist() if raw.summary[int(s)]["dom_part"] == part_lo)
    return GroupSplit(
        name="part",
        label_a=f"part{part_hi}-dom", label_b=f"part{part_lo}-dom",
        records_a=_records_for(raw, set_hi),
        records_b=_records_for(raw, set_lo),
        note=(f"dominant-part {part_hi} ({len(set_hi)} learners) vs "
              f"{part_lo} ({len(set_lo)} learners)"),
    )


# ---------------------------------------------------------------------------
# Synthetic controls (validate the DIF metric end-to-end)
# ---------------------------------------------------------------------------

def make_synthetic_nodif(
    n_learners: int = 4000,
    n_items: int = 600,
    seq_len: int = 120,
    seed: int = 0,
) -> RawBinary:
    """POSITIVE control: one TRUE 2PL latent, the SAME item difficulty for every
    learner -> NO DIF by construction.  Splitting by ability or at random should
    recover item difficulties that replicate (high Spearman, low mean|Δb|) up to
    estimation noise.  This is the achievable CEILING / floor of the metric.

    Generative story: P(correct) = sigmoid(a*(theta - b)) with a single shared b
    per item.  Ability is drawn N(0,1); difficulty N(0,1).
    """
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_learners)
    b = rng.standard_normal(n_items)
    a = 1.5

    records: List[Dict] = []
    summary: Dict[int, Dict] = {}
    for li in range(n_learners):
        sid = li + 1
        q_local = rng.integers(1, n_items + 1, size=seq_len)
        z = theta[li] - b[q_local - 1]
        p = 1.0 / (1.0 + np.exp(-a * z))
        corr = (rng.random(seq_len) < p).astype(np.int64)
        records.append({"student_id": sid,
                        "questions": q_local.astype(np.int64),
                        "responses": corr})
        summary[sid] = {"mean_correct": float(corr.mean()),
                        "seq_len": int(seq_len), "dom_part": 1}
    return RawBinary(records=records, n_items=n_items, summary=summary,
                     meta={"synthetic": "nodif", "true_b": b, "true_theta": theta})


def make_synthetic_dif(
    n_learners: int = 4000,
    n_items: int = 600,
    seq_len: int = 120,
    dif_frac: float = 0.25,
    dif_shift: float = 1.5,
    seed: int = 0,
) -> Tuple[RawBinary, np.ndarray]:
    """NEGATIVE control: inject KNOWN DIF on a subset of items along the ABILITY
    axis.  For the DIF items, the difficulty differs between high- and low-ability
    learners by ``dif_shift`` (uniform DIF).  The ability split MUST detect this:
    the DIF items should show large |Δb| and be flagged, while the non-DIF items
    replicate.  Proves the metric can DETECT non-invariance (the FLOOR is not
    trivially high).

    The DIF is keyed to a per-learner GROUP assignment derived from the learner's
    own mean correctness (so the ability split recovers the same groups).  We
    bootstrap the group from a noisy ability proxy: learners with theta above the
    median are "high", the rest "low", and DIF items get b -> b +/- dif_shift/2
    depending on the learner group.

    Returns (raw, dif_item_mask (n_items,) bool) -- the ground-truth DIF items
    (0-based local index).
    """
    rng = np.random.default_rng(1000 + seed)
    theta = rng.standard_normal(n_learners)
    b = rng.standard_normal(n_items)
    a = 1.5
    n_dif = int(round(dif_frac * n_items))
    dif_items = rng.permutation(n_items)[:n_dif]
    dif_mask = np.zeros(n_items, dtype=bool)
    dif_mask[dif_items] = True

    theta_med = float(np.median(theta))
    records: List[Dict] = []
    summary: Dict[int, Dict] = {}
    for li in range(n_learners):
        sid = li + 1
        is_high = theta[li] >= theta_med
        # Per-learner item difficulty: DIF items shift by +/- dif_shift/2 by group.
        b_eff = b.copy()
        if is_high:
            b_eff[dif_mask] -= dif_shift / 2.0   # easier for high-ability
        else:
            b_eff[dif_mask] += dif_shift / 2.0   # harder for low-ability
        q_local = rng.integers(1, n_items + 1, size=seq_len)
        z = theta[li] - b_eff[q_local - 1]
        p = 1.0 / (1.0 + np.exp(-a * z))
        corr = (rng.random(seq_len) < p).astype(np.int64)
        records.append({"student_id": sid,
                        "questions": q_local.astype(np.int64),
                        "responses": corr})
        summary[sid] = {"mean_correct": float(corr.mean()),
                        "seq_len": int(seq_len), "dom_part": 1}
    raw = RawBinary(records=records, n_items=n_items, summary=summary,
                    meta={"synthetic": "dif", "true_b": b, "dif_mask": dif_mask,
                          "dif_shift": dif_shift})
    return raw, dif_mask
