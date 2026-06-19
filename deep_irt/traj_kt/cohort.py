"""cohort.py -- EdNet-KT1 dense-learner cohort builder for E2.

Reads only the top-N largest user CSVs (proxy for sequence length), joins
correctness and part/tags from EdNet-Contents, deduplicates bundle events,
drops invalid elapsed_time, and returns only learners with >= T_min clean
interactions.

K=4 ordinal label (per EdNetAdapter convention, ascending mastery):
    0  incorrect & slow   (elapsed_time > per-question train median)
    1  incorrect & fast
    2  correct   & slow
    3  correct   & fast
Cold items (unseen in train) fall back to the global train median.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELAPSED_MIN_MS = 1           # drop <= 0 ms
ELAPSED_MAX_MS = 600_000     # drop > 600 s
BUNDLE_GAP_MS  = 1_000       # rows sharing solving_id within this gap -> one event


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

class CohortData(NamedTuple):
    """Everything downstream steps need."""
    seqs_item:   list[np.ndarray]   # per-learner item index arrays (dtype int32)
    seqs_resp:   list[np.ndarray]   # per-learner K=4 ordinal response arrays (int8)
    seqs_corr:   list[np.ndarray]   # per-learner binary correct (0/1, int8)
    seqs_part:   list[np.ndarray]   # per-learner part 1-7 arrays (int8)
    seqs_uid:    list[str]          # user ids (original filename stems)
    item_vocab:  dict[str, int]     # question_id -> 0-based index
    n_items:     int
    stats:       dict               # cohort statistics for reporting


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _file_list_by_size(kt1_dir: Path, top_n: int) -> list[Path]:
    """Return top_n user CSVs ranked descending by file size (fast via scandir)."""
    pairs: list[tuple[int, Path]] = []
    with os.scandir(kt1_dir) as it:
        for entry in it:
            if entry.name.startswith("u") and entry.name.endswith(".csv"):
                pairs.append((entry.stat().st_size, Path(entry.path)))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in pairs[:top_n]]


def _load_questions(contents_dir: Path) -> pd.DataFrame:
    """Load questions.csv; return DataFrame with question_id as str index."""
    df = pd.read_csv(
        contents_dir / "questions.csv",
        usecols=["question_id", "correct_answer", "part"],
        dtype={"question_id": str, "correct_answer": str, "part": "Int8"},
    )
    df = df.set_index("question_id")
    return df


def _bundle_dedup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized bundle dedup: rows sharing solving_id within BUNDLE_GAP_MS.

    Groups are formed by a change in solving_id OR a gap >= BUNDLE_GAP_MS.
    Conservative correct: all sub-rows must be correct.
    elapsed_time: median of the group (as int).
    timestamp: first row of the group.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Build group IDs: a new group starts when solving_id changes OR gap is large
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    sid = df["solving_id"].fillna(-1).to_numpy(dtype=np.int64)

    gap = np.empty(len(df), dtype=bool)
    gap[0] = True
    gap[1:] = (sid[1:] != sid[:-1]) | ((ts[1:] - ts[:-1]) >= BUNDLE_GAP_MS)
    group_id = gap.cumsum() - 1
    df["_gid"] = group_id

    agg = df.groupby("_gid", sort=False).agg(
        timestamp    =("timestamp",    "first"),
        question_id  =("question_id",  "first"),
        user_answer  =("user_answer",  "first"),
        elapsed_time =("elapsed_time", "median"),
        _correct_raw =("_correct_raw", "all"),
        _uid         =("_uid",         "first"),
    ).reset_index(drop=True)
    agg["elapsed_time"] = agg["elapsed_time"].round().astype("Int64")
    return agg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_cohort(
    kt1_dir: str | Path,
    contents_dir: str | Path,
    top_n_files: int = 15_000,
    t_min: int = 200,
    verbose: bool = True,
) -> CohortData:
    """Build the dense-learner cohort from EdNet-KT1.

    Parameters
    ----------
    kt1_dir       : path to EdNet-KT1/KT1/
    contents_dir  : path to EdNet-Contents/contents/
    top_n_files   : how many largest user CSVs to read
    t_min         : minimum clean interactions to keep a learner
    verbose       : print progress
    """
    kt1_dir = Path(kt1_dir)
    contents_dir = Path(contents_dir)

    if verbose:
        print(f"[cohort] ranking files by size in {kt1_dir} ...")
    top_files = _file_list_by_size(kt1_dir, top_n_files)
    if verbose:
        print(f"[cohort] selected top {len(top_files)} files by size")

    q_df = _load_questions(contents_dir)

    # --- Read all user files in one vectorized pass -------------------------
    if verbose:
        print(f"[cohort] reading {len(top_files)} user files ...")

    frames: list[pd.DataFrame] = []
    n_rows_raw = 0
    n_rows_after_bundle = 0
    n_rows_bundle_collapsed = 0
    n_files_ok = 0

    for path in top_files:
        try:
            df = pd.read_csv(
                path,
                dtype={"solving_id": "Int64", "question_id": str,
                       "user_answer": str, "elapsed_time": "Int64",
                       "timestamp": "Int64"},
            )
        except Exception:
            continue
        if df.empty:
            continue

        uid = path.stem
        df["_uid"] = uid

        # Correctness: join from questions table (vectorized)
        ca = q_df["correct_answer"].reindex(df["question_id"].astype(str))
        df["_correct_raw"] = (ca.values == df["user_answer"].values)

        n_rows_raw += len(df)

        # Bundle dedup (vectorized)
        pre = len(df)
        df = _bundle_dedup_df(df)
        post = len(df)
        n_rows_bundle_collapsed += (pre - post)
        n_rows_after_bundle += post

        # Filter elapsed_time
        et = df["elapsed_time"].astype(float)
        mask = (et > ELAPSED_MIN_MS) & (et <= ELAPSED_MAX_MS) & et.notna()
        df = df[mask].copy()
        if df.empty:
            continue

        frames.append(df)
        n_files_ok += 1

    if not frames:
        raise RuntimeError("No valid user data found after reading files.")

    if verbose:
        print(f"[cohort] read {n_files_ok} files ok, "
              f"{n_rows_raw:,} raw rows, "
              f"{n_rows_bundle_collapsed:,} collapsed as bundles")

    big = pd.concat(frames, ignore_index=True)
    big["question_id"] = big["question_id"].astype(str)

    # --- 80/20 train split (by user, deterministic) -------------------------
    uids = big["_uid"].unique().tolist()
    n_train = int(len(uids) * 0.8)
    train_uid_set = set(uids[:n_train])

    train_mask = big["_uid"].isin(train_uid_set)
    train_df = big[train_mask & big["elapsed_time"].notna()].copy()
    train_df["_et_f"] = train_df["elapsed_time"].astype(float)

    global_train_median = float(train_df["_et_f"].median())
    if verbose:
        print(f"[cohort] global train median = {global_train_median:.0f} ms")

    # Per-question train median (vectorized)
    rt_per_q = train_df.groupby("question_id")["_et_f"].median()
    rt_per_q_map: dict[str, float] = rt_per_q.to_dict()
    if verbose:
        print(f"[cohort] per-question medians: {len(rt_per_q_map):,} questions")

    # --- K=4 ordinal label (fully vectorized) -------------------------------
    big["_et_f"] = big["elapsed_time"].astype(float)
    med_q = big["question_id"].map(rt_per_q_map).fillna(global_train_median)
    is_fast = (big["_et_f"] <= med_q).astype(int)
    is_corr = big["_correct_raw"].astype(int)
    # 0=inc-slow, 1=inc-fast, 2=corr-slow, 3=corr-fast
    big["_ordinal"] = (is_corr * 2 + is_fast).astype("int8")
    big["_correct"] = is_corr.astype("int8")

    # Part from questions table
    big["_part"] = q_df["part"].reindex(big["question_id"]).values
    big["_part"] = pd.array(big["_part"], dtype="Int8").fillna(0).astype("int8")

    # --- Build global item vocab --------------------------------------------
    all_qids = big["question_id"].unique().tolist()
    item_vocab: dict[str, int] = {q: i for i, q in enumerate(all_qids)}
    big["_item_idx"] = big["question_id"].map(item_vocab).astype("int32")

    # --- Filter learners by T_min -------------------------------------------
    seq_lengths = big.groupby("_uid").size()
    n_before = len(seq_lengths)
    keep_uids = set(seq_lengths[seq_lengths >= t_min].index.tolist())
    big = big[big["_uid"].isin(keep_uids)].copy()
    n_after = big["_uid"].nunique()

    if verbose:
        print(f"[cohort] learners: {n_before} before T>={t_min}, "
              f"{n_after} after")

    if n_after < 100:
        raise RuntimeError(
            f"Only {n_after} learners survived T>={t_min} filter; "
            "increase top_n_files."
        )

    # --- Build per-learner sequence arrays ----------------------------------
    big = big.sort_values(["_uid", "timestamp"])
    seqs_item: list[np.ndarray] = []
    seqs_resp: list[np.ndarray] = []
    seqs_corr: list[np.ndarray] = []
    seqs_part: list[np.ndarray] = []
    seqs_uid:  list[str] = []
    lengths = []

    for uid, grp in big.groupby("_uid", sort=False):
        seqs_item.append(grp["_item_idx"].to_numpy(dtype=np.int32))
        seqs_resp.append(grp["_ordinal"].to_numpy(dtype=np.int8))
        seqs_corr.append(grp["_correct"].to_numpy(dtype=np.int8))
        seqs_part.append(grp["_part"].to_numpy(dtype=np.int8))
        seqs_uid.append(str(uid))
        lengths.append(len(grp))

    lengths_arr = np.array(lengths)
    bundle_frac = n_rows_bundle_collapsed / max(n_rows_raw, 1)

    stats = {
        "n_learners_before_filter": n_before,
        "n_learners_after_filter":  n_after,
        "median_seq_len":  float(np.median(lengths_arr)),
        "p90_seq_len":     float(np.percentile(lengths_arr, 90)),
        "max_seq_len":     int(lengths_arr.max()),
        "item_vocab_size": len(item_vocab),
        "bundle_collapse_frac": float(bundle_frac),
        "global_train_median_ms": float(global_train_median),
    }

    if verbose:
        print(f"\n[cohort] === COHORT STATS ===")
        print(f"  Learners before T>={t_min} filter : {n_before}")
        print(f"  Learners after  T>={t_min} filter : {n_after}")
        print(f"  Median seq len   : {stats['median_seq_len']:.0f}")
        print(f"  90th pct seq len : {stats['p90_seq_len']:.0f}")
        print(f"  Max seq len      : {stats['max_seq_len']}")
        print(f"  Item vocab size  : {stats['item_vocab_size']}")
        print(f"  Bundle collapse  : {bundle_frac*100:.2f}% of raw rows")
        print()

    return CohortData(
        seqs_item=seqs_item,
        seqs_resp=seqs_resp,
        seqs_corr=seqs_corr,
        seqs_part=seqs_part,
        seqs_uid=seqs_uid,
        item_vocab=item_vocab,
        n_items=len(item_vocab),
        stats=stats,
    )
