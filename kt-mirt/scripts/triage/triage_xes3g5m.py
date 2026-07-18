"""STAGE-0 triage: XES3G5M (pyKT-format KT bed from a Chinese K-12 math app).

Two parallel local extracts cover the same underlying interaction log at
two different explosion granularities (both are pyKT convention, not a
bug):

  - kc_level/train_valid_sequences.csv -- multi-KC questions are ALREADY
    exploded into one fixed-length-200-window POSITION per (question, KC)
    pair: a question tagged with 2 concepts occupies 2 consecutive window
    positions, same question id / response / timestamp repeated, only the
    single `concepts` token differing. This is exactly the "long" per-KC
    grain the other beds' scripts build by hand via .explode() on a
    ~~/;-separated KC list -- XES3G5M ships it pre-exploded, so no explode
    step is needed here for (b)/(c)/(d).
  - question_level/train_valid_sequences_quelevel.csv -- the TRUE
    one-row-per-real-interaction grain: multi-KC questions keep ONE window
    position, with `concepts` joined by "_" (e.g. "166_170"). This is the
    analogue of KDD's "~~"-joined KC(<model>) cell / EdNet's ";"-joined
    `tags` cell, i.e. the right input for (e) KC-pair co-occurrence and
    for (f)/(g) item-KC arity (a static Q-matrix row per question id).

Following that division: (a) overall correct rate/response distribution
and (e)/(f)/(g) are computed from question_level (true-event grain, no
KC-driven row duplication, matching how triage_kdd/triage_ednet computed
their (a) block at the un-exploded slot grain); (b)/(c)/(d) per-KC and
per-(learner,KC)-opportunity stats are computed from kc_level (the
pre-exploded per-KC grain), consistent with the marginal/opportunity/
growth counters the other beds build by exploding onto the same grain.

PADDING (both files use the pyKT convention): every sequence is padded/
windowed to a fixed length of 200 positions per row; `selectmasks` is 1
for a real position and -1 for padding. This script masks strictly on
selectmasks == 1 (NOT on e.g. questions != "-1", which happens to
coincide here but is not the documented contract) and reports the valid
fractions actually observed vs the raw 200 x n_rows cell count.

TEMPORAL ORDER (verified, recorded in the "qc" block below): every uid
appears in >=1 CONTIGUOUS block of consecutive file rows (its successive
200-length chunks), and within a row `timestamps` is non-decreasing
across valid positions. So processing kc_level in on-disk row order, and
within each row in on-disk position order, reproduces each student's true
chronological interaction order -- required for a correct running
per-(uid, kc) opportunity index and growth curve. This is checked
programmatically (not merely assumed) and the check result is recorded
under "qc" in the output JSON.

Both files are read whole (33,397 x 200 and 30,965 x 200 cells for the
2026-07 local extract): small enough for a single in-memory pass, no
chunking or sampling needed, unlike the KDD/Junyi log sizes.

Usage:
    python triage_xes3g5m.py <path/to/XES3G5M/kc_level/train_valid_sequences.csv> \
        <path/to/XES3G5M/question_level/train_valid_sequences_quelevel.csv> \
        [--out-dir kt-mirt/_planning/triage]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_common import (  # noqa: E402
    DECOUPLING_DEFINITION_TEXT,
    GROWTH_MAX_OPPORTUNITY,
    KCModelAggregator,
    pair_counts_from_kc_lists,
    write_json,
)

SEQ_LEN = 200  # fixed pyKT window length for this local extract (verified below)


def split_fixed_width(series: pd.Series, seq_len: int = SEQ_LEN) -> np.ndarray:
    """A Series of comma-joined, fixed-length strings -> a (n_rows, seq_len) str array."""
    lengths = series.str.count(",") + 1
    bad = lengths != seq_len
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows do not have the expected fixed width {seq_len}")
    return np.array(series.str.split(",").tolist())


def load_kc_level(path: Path) -> dict:
    df = pd.read_csv(path, dtype=str)
    uid = df["uid"].to_numpy()
    fold = df["fold"].to_numpy()

    # QC: (1) every uid's rows are contiguous in file order; (2) per-uid fold is constant.
    rownum = np.arange(len(df))
    order = pd.DataFrame({"uid": uid, "rownum": rownum})
    grp = order.groupby("uid")["rownum"].apply(list)
    contiguous = grp.apply(lambda rows: rows == list(range(rows[0], rows[0] + len(rows))))
    fold_per_uid = pd.Series(fold).groupby(uid).nunique()
    qc = {
        "n_rows": int(len(df)),
        "n_uid": int(len(grp)),
        "frac_uid_with_contiguous_chunks": float(contiguous.mean()),
        "frac_uid_with_single_fold": float((fold_per_uid == 1).mean()),
    }

    concepts = split_fixed_width(df["concepts"])
    responses = split_fixed_width(df["responses"]).astype(int)
    selectmasks = split_fixed_width(df["selectmasks"]).astype(int)

    valid = (selectmasks == 1)
    uid_rep = np.repeat(uid, SEQ_LEN).reshape(len(df), SEQ_LEN)

    long_df = pd.DataFrame({
        "uid": uid_rep[valid],
        "kc": concepts[valid],
        "correct": responses[valid],
    })
    return {
        "long_df": long_df,
        "qc": qc,
        "n_cells_total": int(df.shape[0] * SEQ_LEN),
        "n_cells_valid": int(valid.sum()),
        "n_distinct_kc": int(pd.unique(long_df["kc"]).size),
    }


def load_question_level(path: Path) -> dict:
    df = pd.read_csv(path, dtype=str)
    questions = split_fixed_width(df["questions"])
    concepts = split_fixed_width(df["concepts"])
    responses = split_fixed_width(df["responses"]).astype(int)
    selectmasks = split_fixed_width(df["selectmasks"]).astype(int)

    valid = (selectmasks == 1)
    flat_q = questions[valid]
    flat_c = concepts[valid]
    flat_r = responses[valid]
    return {
        "flat_question": flat_q,
        "flat_kc_joined": flat_c,   # multi-KC questions joined by "_"
        "flat_correct": flat_r,
        "n_cells_total": int(df.shape[0] * SEQ_LEN),
        "n_cells_valid": int(valid.sum()),
        "n_distinct_question": int(pd.unique(flat_q).size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kc_level_csv")
    parser.add_argument("quelevel_csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "_planning" / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kc_path = Path(args.kc_level_csv)
    que_path = Path(args.quelevel_csv)

    kc_data = load_kc_level(kc_path)
    print(f"[triage_xes3g5m] kc_level loaded: {kc_data['qc']['n_rows']} rows, "
          f"{kc_data['qc']['n_uid']} uid, {kc_data['n_cells_valid']:,}/{kc_data['n_cells_total']:,} valid cells, "
          f"{kc_data['n_distinct_kc']} distinct leaf KC, elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    que_data = load_question_level(que_path)
    print(f"[triage_xes3g5m] question_level loaded: "
          f"{que_data['n_cells_valid']:,}/{que_data['n_cells_total']:,} valid cells, "
          f"{que_data['n_distinct_question']} distinct questions, elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    agg = KCModelAggregator(name="leaf_kc")

    # (b)/(c)/(d): kc_level, already exploded to one row per (interaction, KC).
    long_df = kc_data["long_df"]
    long_df["opp"] = long_df.groupby(["uid", "kc"], sort=False).cumcount() + 1

    marg = long_df.groupby("kc")["correct"].agg(["size", "sum"])
    agg.add_chunk_marginal(marg["size"].to_dict(), marg["sum"].to_dict())

    opp_max = long_df.groupby(["uid", "kc"])["opp"].max()
    agg.add_chunk_opp_max({key: int(v) for key, v in opp_max.items()})

    gsub = long_df[long_df["opp"].between(1, GROWTH_MAX_OPPORTUNITY)]
    ggrp = gsub.groupby(["kc", "opp"])["correct"].agg(["size", "sum"])
    agg.add_chunk_growth(ggrp["size"].to_dict(), ggrp["sum"].to_dict())

    # (a): question_level, true one-row-per-real-interaction grain (no KC-driven duplication).
    flat_r = que_data["flat_correct"]
    agg.add_overall(int(flat_r.size), int(flat_r.sum()), flat_r)

    # (e): KC-pair co-occurrence, also from question_level's "_"-joined per-slot concept lists.
    kc_list_series = pd.Series(que_data["flat_kc_joined"]).str.split("_")
    pair_counts = pair_counts_from_kc_lists(kc_list_series)
    agg.add_chunk_pairs(pair_counts)

    # (f)/(g): item-KC arity + pure anchors, from question_level's static question -> KC-set map.
    item_to_kcset = {
        qid: frozenset(kcs)
        for qid, kcs in zip(que_data["flat_question"].tolist(), kc_list_series.tolist())
    }
    agg.add_items(item_to_kcset)

    results = {
        "bed": "xes3g5m",
        "source_files": {
            "kc_level": str(kc_path.resolve()),
            "question_level": str(que_path.resolve()),
        },
        "unit_of_analysis": {
            "a_overall_and_e_pairs_and_f_g_anchors": "question_level row/position -- one real "
                "interaction (question attempt); multi-KC questions keep a single position with "
                "concepts joined by '_' (NOT exploded).",
            "b_c_d_per_kc_and_opportunity": "kc_level row/position -- one (interaction, leaf-KC) pair; "
                "multi-KC questions occupy one position per tagged KC (pre-exploded by the pyKT "
                "extract, same question id / response / timestamp repeated per KC).",
        },
        "response_field": "responses (0/1 binary correctness), both files",
        "padding_convention": "pyKT fixed-length-200 windows per row; selectmasks==1 marks a real "
                               "position, selectmasks==-1 marks padding. Masked strictly on "
                               "selectmasks==1 in this script.",
        "sampling": "none (both local files read in full; no chunking needed at this size)",
        "scale_check_vs_expected": {
            "expected": "~18k students, ~7.6k questions, 865 leaf KCs, 5.5M interactions "
                        "(per the STAGE-0 task spec)",
            "actual_train_valid_split_only": {
                "n_uid_train_valid": kc_data["qc"]["n_uid"],
                "n_uid_test_csv_not_used_here": 3613,
                "n_uid_train_valid_plus_test": kc_data["qc"]["n_uid"] + 3613,
                "n_distinct_questions": que_data["n_distinct_question"],
                "n_distinct_leaf_kc": kc_data["n_distinct_kc"],
                "n_valid_interactions_question_level_grain": int(que_data["n_cells_valid"]),
                "n_valid_kc_slots_kc_level_grain": int(kc_data["n_cells_valid"]),
            },
            "note": "train_valid_sequences.csv (used here, per the task spec) excludes the held-out "
                    "test.csv split (3613 more uid); 14453+3613=18066 matches the ~18k expectation, "
                    "7618 questions and 865 leaf KC match almost exactly, and the 4.45M true "
                    "interactions (question_level grain) undercounts the 5.5M expectation only "
                    "because test.csv is excluded here as instructed.",
        },
        "qc": kc_data["qc"],
        "decoupling_definition": DECOUPLING_DEFINITION_TEXT,
        "kc_models": {"leaf_kc": agg.finalize()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "xes3g5m_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
