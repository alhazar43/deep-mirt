"""STAGE-0 triage: Junyi Academy 2020 (Log_Problem.csv, Info_Content.csv).

KC LEVEL CHOICE. Info_Content ships a strict 4-level topic hierarchy per
exercise (level1_id .. level4_id, each exercise belongs to exactly one
node at each level -- level4->level3 is a many-to-one tree, verified
programmatically). In this local extract: level1_id has 1 distinct value
(all 1330 exercises are "math"), level2_id has 10 (median 108 exercises
each -- a subject-area split, too coarse), level3_id has 42 (median 23
exercises each -- still fairly coarse), level4_id has 171 (median 8
exercises each, min 1, max 18). level4_id is the finest level offered and
lands at a plausible single-skill grain (a handful of exercises per node,
comparable in spirit to KDD's SubSkills granularity), so it is used as
the KC layer here. This release ships NO separate prerequisite/skill
field (verified in an earlier pass over this dataset) -- level4_id is a
topic-tree leaf being reused as the KC, not a purpose-built skill tag, and
that distinction is preserved in the output JSON.

NOTE ON NAMING COLLISION: Log_Problem.csv has its OWN column literally
named `level` (0-based, Junyi's own adaptive difficulty/mastery-ladder
stage reached on that exercise attempt). This is UNRELATED to
Info_Content's level1_id..level4_id topic hierarchy and is not used as a
KC candidate here or anywhere in this script -- named explicitly to avoid
the two "level"s being conflated.

ITEM GRAIN. Info_Content is keyed by ucid (exercise); Log_Problem's finer
`upid` (a specific problem instance within an exercise) has no separate
KC crosswalk of its own and a given upid is not always nested in a single
ucid in this extract (checked: not 1:1), so the Q-matrix item unit for
(f)/(g) is the EXERCISE (ucid), the same unit Info_Content tags. One
practice slot for all other stats is one row of Log_Problem (one problem
attempt).

STRUCTURAL CONSEQUENCE FOR (e). Because the level1..4 hierarchy is a
strict tree, every exercise (and therefore every practice slot) carries
EXACTLY ONE KC. Two distinct KCs can therefore never co-occur on the same
slot by construction, so the co-occurrence-based decoupling metric (as
defined identically to the other beds in triage_common.py) is
structurally vacuous here: n_pairs_with_cross_occurrence is 0, not a
small or unlucky number. This is recorded explicitly rather than left to
look like a coincidence or a bug.

SCALE / RUNTIME. Full file: 16,217,311 rows, 72,758 distinct uuid (users),
1326 of 1330 Info_Content exercises actually attempted. A full streamed
pass (parse timestamp_TW as a plain ISO-8601-prefixed STRING -- it sorts
correctly lexicographically without an expensive to_datetime parse --
sort by (uuid, timestamp_TW), groupby-cumcount for opportunity) completes
in well under a minute end-to-end on this machine, so no user sampling
was needed; the --n-users/--seed sampling fallback described in the task
spec is implemented but unused by default (n_users=0 disables it).

Usage:
    python triage_junyi2020.py <path/to/junyi2020/Log_Problem.csv> \
        <path/to/junyi2020/Info_Content.csv> \
        [--out-dir kt-mirt/_planning/triage] [--n-users 0] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_common import (  # noqa: E402
    DECOUPLING_DEFINITION_TEXT,
    GROWTH_MAX_OPPORTUNITY,
    KCModelAggregator,
    write_json,
)

KC_LEVEL_COLUMN = "level4_id"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_problem_csv")
    parser.add_argument("info_content_csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-users", type=int, default=0, help="0 = full pass (default, fits budget easily)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "_planning" / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log_path = Path(args.log_problem_csv)
    info_path = Path(args.info_content_csv)

    info = pd.read_csv(info_path, usecols=["ucid", "level4_id", "level3_id", "level2_id", "level1_id"])
    ucid2kc = dict(zip(info["ucid"], info[KC_LEVEL_COLUMN]))
    kc_group_sizes = info.groupby(KC_LEVEL_COLUMN).size()
    hierarchy_shape = {
        lvl: int(info[lvl].nunique()) for lvl in ["level1_id", "level2_id", "level3_id", "level4_id"]
    }
    print(f"[triage_junyi2020] Info_Content: {len(info)} exercises, hierarchy shape {hierarchy_shape}, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    df = pd.read_csv(
        log_path,
        usecols=["timestamp_TW", "uuid", "ucid", "is_correct"],
        dtype={"uuid": "category", "ucid": "category", "is_correct": "bool"},
    )
    n_rows_total = len(df)
    n_uuid_total = int(df["uuid"].nunique())
    n_ucid_attempted = int(df["ucid"].nunique())
    print(f"[triage_junyi2020] Log_Problem loaded: {n_rows_total:,} rows, "
          f"{n_uuid_total} uuid, {n_ucid_attempted} ucid, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    sampling_note = "none (full file streamed)"
    if args.n_users and args.n_users > 0:
        all_users = df["uuid"].cat.categories.tolist()
        rng = random.Random(args.seed)
        sampled_users = set(rng.sample(all_users, min(args.n_users, len(all_users))))
        df = df[df["uuid"].isin(sampled_users)].copy()
        sampling_note = (f"{len(sampled_users)} of {len(all_users)} uuid, "
                          f"random.Random(seed={args.seed}).sample() over pandas category order")
        print(f"[triage_junyi2020] sampled to {len(df):,} rows ({sampling_note}), "
              f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    df["kc"] = df["ucid"].map(ucid2kc)
    n_unmapped = int(df["kc"].isna().sum())
    df = df.dropna(subset=["kc"])
    df["is_correct"] = df["is_correct"].astype("int8")

    # Chronological order per user: timestamp_TW is "YYYY-MM-DD HH:MM:SS UTC", which sorts
    # correctly as a plain string (ISO-8601 date prefix) -- no need for an expensive
    # to_datetime parse. Stable sort preserves original row order as the tie-break for
    # same-timestamp rows (this timestamp is coarsened, ties are common).
    df.sort_values(["uuid", "timestamp_TW"], kind="stable", inplace=True)
    print(f"[triage_junyi2020] sorted for chronological per-user order, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    df["opp"] = df.groupby(["uuid", "kc"], sort=False, observed=True).cumcount() + 1

    agg = KCModelAggregator(name=KC_LEVEL_COLUMN)

    # (a) overall: one row of Log_Problem IS one real interaction, no KC-driven duplication
    # (every exercise carries exactly one KC -- see module docstring).
    agg.add_overall(len(df), int(df["is_correct"].sum()), df["is_correct"].to_numpy())

    marg = df.groupby("kc", observed=True)["is_correct"].agg(["size", "sum"])
    agg.add_chunk_marginal(marg["size"].to_dict(), marg["sum"].to_dict())

    opp_max = df.groupby(["uuid", "kc"], observed=True)["opp"].max()
    agg.add_chunk_opp_max({key: int(v) for key, v in opp_max.items()})

    gsub = df[df["opp"].between(1, GROWTH_MAX_OPPORTUNITY)]
    ggrp = gsub.groupby(["kc", "opp"], observed=True)["is_correct"].agg(["size", "sum"])
    agg.add_chunk_growth(ggrp["size"].to_dict(), ggrp["sum"].to_dict())

    # (e) KC-pair co-occurrence: structurally empty here (see module docstring) -- every
    # slot carries exactly one KC under this strict-tree hierarchy, so no pair of distinct
    # KCs ever co-occurs on the same slot. Left as an empty pair_both Counter (never
    # populated) rather than forced through pair_counts_from_kc_lists on singleton lists.

    # (f)/(g) item-KC arity + pure anchors: static Q-matrix from the FULL Info_Content bank
    # (1330 exercises), not restricted to the log sample -- same convention as EdNet's
    # full-question-bank item_to_kcset. Arity is 1 everywhere by construction (single-parent
    # tree), so every exercise is trivially "pure"; frac_kc_with_ge_3_pure reduces to "does
    # this KC/level4 group have >=3 exercises", which is stated plainly in the report rather
    # than presented as evidence of a genuine multi-KC Q-matrix.
    item_to_kcset = {ucid: frozenset({kc}) for ucid, kc in ucid2kc.items()}
    agg.add_items(item_to_kcset)

    results = {
        "bed": "junyi2020",
        "source_files": {
            "log_problem": str(log_path.resolve()),
            "info_content": str(info_path.resolve()),
        },
        "kc_level_choice": {
            "chosen": KC_LEVEL_COLUMN,
            "hierarchy_shape_n_distinct": hierarchy_shape,
            "exercises_per_level4_kc_quartiles": {
                "min": int(kc_group_sizes.min()), "median": float(kc_group_sizes.median()),
                "max": int(kc_group_sizes.max()),
            },
            "justification": "level1_id is a single value (all math), level2_id (10 groups) and "
                              "level3_id (42 groups) are coarse subject/unit splits (median "
                              "108 / 23 exercises per group), level4_id (171 groups, median 8 "
                              "exercises, max 18) is the finest level offered and the closest "
                              "analogue to a single skill; used as the KC layer. This release "
                              "has NO separate prerequisite/skill field (verified earlier), so "
                              "level4_id is a repurposed topic-tree leaf, not a purpose-built "
                              "KC tag -- noted explicitly, not papered over.",
            "note_on_log_problem_level_column": "Log_Problem.csv's own `level` column is Junyi's "
                                                 "per-exercise adaptive difficulty/mastery-ladder "
                                                 "stage (0-based), UNRELATED to level1_id..level4_id "
                                                 "and not used as a KC candidate here.",
        },
        "item_grain": "ucid (exercise); Log_Problem's finer upid has no separate KC crosswalk and "
                      "is not 1:1 nested in ucid in this extract, so ucid is the Q-matrix item unit.",
        "unit_of_analysis": "one row of Log_Problem.csv (one problem attempt)",
        "response_field": "is_correct (bool)",
        "sampling": sampling_note,
        "n_rows_total": n_rows_total,
        "n_rows_used_after_sampling": int(len(df) + n_unmapped),
        "n_rows_kc_unmapped_dropped": n_unmapped,
        "n_uuid_total": n_uuid_total,
        "n_exercises_in_bank": int(len(info)),
        "n_exercises_attempted_in_log": n_ucid_attempted,
        "decoupling_definition": DECOUPLING_DEFINITION_TEXT,
        "e_decoupling_structural_note": "Structurally vacuous for this bed: every practice slot "
            "carries exactly one KC (strict level1..4 tree), so two distinct KCs never co-occur "
            "on the same slot. n_pairs_with_cross_occurrence below is 0 by construction, not an "
            "unlucky sample.",
        "kc_models": {KC_LEVEL_COLUMN: agg.finalize()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "junyi2020_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
