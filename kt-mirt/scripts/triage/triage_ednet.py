"""STAGE-0 triage: EdNet KT1 (Riiid), joined to the EdNet-Contents item bank.

EdNet KT1 ships one CSV per student (784,309 files total) rather than one
big log, so this script SAMPLES a fixed number of user files (default
4000) with a fixed seed rather than reading all of them -- see --n-users /
--seed in the output JSON for the exact sample drawn. Item-KC arity and
pure-anchor stats (f)/(g) are the exception: those are computed from the
FULL 13,169-question item bank (questions.csv), not the sample, since that
file is small and static (every question's tag set is available regardless
of whether any sampled user answered it).

Unit of analysis: one question attempt (one row of a user's KT1 CSV).
Response: the chosen option (`user_answer`) vs `correct_answer` from
questions.csv. KC = a tag in the `;`-separated `tags` column (multi-tag
questions are common). Opportunity index for a (user, tag) pair is our own
1-based running count of that user's attempts tagged with that tag, in
`solving_id` order (EdNet does not ship an explicit opportunity counter
the way KDD does).

Usage:
    python triage_ednet.py <path/to/EdNet-KT1/KT1> <path/to/questions.csv> \
        [--out-dir kt-mirt/_planning/triage] [--n-users 4000] [--seed 42]
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
    pair_counts_from_kc_lists,
    write_json,
)

DEFAULT_N_USERS = 4000
DEFAULT_SEED = 42


def load_question_bank(questions_csv: str):
    """questions.csv -> {question_id: (correct_answer, [tag,...])}, plus full-bank item->KC map."""
    q = pd.read_csv(questions_csv, dtype=str, usecols=["question_id", "correct_answer", "tags"])
    q["tags"] = q["tags"].fillna("")
    tags_list = q["tags"].apply(lambda s: [t for t in s.split(";") if t] if s else [])
    qbank = dict(zip(q["question_id"], zip(q["correct_answer"], tags_list)))
    item_to_kcset = {qid: frozenset(tags) for qid, (_, tags) in qbank.items() if tags}
    return qbank, item_to_kcset


def process_user_file(path: Path, user_id: str, qbank: dict, agg: KCModelAggregator, growth_max_o: int) -> dict:
    df = pd.read_csv(path, dtype=str, usecols=["solving_id", "question_id", "user_answer"])
    if df.empty:
        return {"n_rows": 0, "n_unmapped": 0}
    df["solving_id"] = pd.to_numeric(df["solving_id"], errors="coerce")
    df = df.sort_values("solving_id", kind="stable")

    ca = df["question_id"].map(lambda qid: qbank.get(qid, (None, None))[0])
    tags = df["question_id"].map(lambda qid: qbank.get(qid, (None, None))[1])
    unmapped = ca.isna() | tags.isna()
    n_unmapped = int(unmapped.sum())
    df = df.loc[~unmapped].copy()
    if df.empty:
        return {"n_rows": 0, "n_unmapped": n_unmapped}
    df["correct"] = (df["user_answer"].values == ca[~unmapped].values).astype(int)
    df["tags_list"] = tags[~unmapped].values

    # (a) overall + response-category (chosen-option) distribution: one row per slot.
    agg.add_overall(len(df), int(df["correct"].sum()), df["user_answer"].to_numpy())

    # (e) KC-pair co-occurrence: pairs of tags on the SAME question attempt.
    pair_counts = pair_counts_from_kc_lists(df["tags_list"])
    agg.add_chunk_pairs(pair_counts)

    # explode to (slot, tag) grain for marginal / opportunity / growth stats.
    long = df[["correct", "tags_list"]].explode("tags_list").rename(columns={"tags_list": "kc"})
    long = long.dropna(subset=["kc"])
    if long.empty:
        return {"n_rows": len(df), "n_unmapped": n_unmapped}
    long["opp"] = long.groupby("kc").cumcount() + 1

    marg = long.groupby("kc")["correct"].agg(["size", "sum"])
    agg.add_chunk_marginal(marg["size"].to_dict(), marg["sum"].to_dict())

    opp_max = long.groupby("kc")["opp"].max()
    agg.add_chunk_opp_max({(user_id, kc): int(v) for kc, v in opp_max.items()})

    gsub = long[long["opp"].between(1, growth_max_o)]
    ggrp = gsub.groupby(["kc", "opp"])["correct"].agg(["size", "sum"])
    agg.add_chunk_growth(ggrp["size"].to_dict(), ggrp["sum"].to_dict())

    return {"n_rows": len(df), "n_unmapped": n_unmapped}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kt1_dir")
    parser.add_argument("questions_csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-users", type=int, default=DEFAULT_N_USERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "_planning" / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    qbank, item_to_kcset = load_question_bank(args.questions_csv)
    print(f"[triage_ednet] question bank: {len(qbank)} questions, "
          f"{len(item_to_kcset)} with >=1 tag, elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    kt1_dir = Path(args.kt1_dir)
    all_files = sorted(p.name for p in kt1_dir.iterdir() if p.suffix == ".csv")
    n_total_files = len(all_files)
    rng = random.Random(args.seed)
    sampled = rng.sample(all_files, min(args.n_users, n_total_files))
    print(f"[triage_ednet] sampled {len(sampled)} of {n_total_files} user files "
          f"(seed={args.seed}), elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    agg = KCModelAggregator(name="tags")
    agg.add_items(item_to_kcset)  # exact, full item bank -- not sample-dependent.

    n_users_processed = 0
    n_users_empty = 0
    n_rows_total = 0
    n_unmapped_total = 0
    for i, fname in enumerate(sampled):
        user_id = fname[:-4]  # strip ".csv"
        try:
            r = process_user_file(kt1_dir / fname, user_id, qbank, agg, GROWTH_MAX_OPPORTUNITY)
        except pd.errors.EmptyDataError:
            r = {"n_rows": 0, "n_unmapped": 0}
        if r["n_rows"] == 0:
            n_users_empty += 1
        else:
            n_users_processed += 1
        n_rows_total += r["n_rows"]
        n_unmapped_total += r["n_unmapped"]
        if (i + 1) % args.progress_every == 0:
            print(f"[triage_ednet] {i + 1}/{len(sampled)} user files, "
                  f"rows_so_far={n_rows_total:,}, elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    results = {
        "bed": "ednet_kt1",
        "source_dir": str(kt1_dir.resolve()),
        "questions_csv": str(Path(args.questions_csv).resolve()),
        "unit_of_analysis": "one question attempt (one row of a user's KT1 CSV)",
        "response_field": "user_answer vs questions.csv.correct_answer (4-option multiple choice)",
        "sampling": f"{len(sampled)} of {n_total_files} per-user CSV files, "
                    f"random.Random(seed={args.seed}).sample() over the sorted filename list",
        "n_users_sampled": len(sampled),
        "n_users_with_data": n_users_processed,
        "n_users_empty_or_unreadable": n_users_empty,
        "n_rows_total": n_rows_total,
        "n_rows_unmapped_question_id_dropped": n_unmapped_total,
        "decoupling_definition": DECOUPLING_DEFINITION_TEXT,
        "kc_models": {"tags": agg.finalize()},
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "ednet_kt1_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
