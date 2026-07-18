"""STAGE-0 triage: KDD Cup 2010 Algebra 2008-2009 (challenge set).

Streams the tab-separated Cognitive Tutor transaction log in chunks (the
raw file is ~3 GB / ~8.9M rows) and computes the (a)-(g) triage statistics
for each requested KC model (SubSkills, KTracedSkills, Rules -- DataShop
ships multiple competing KC taggings for the same step log).

Unit of analysis: one "step" (one Cognitive-Tutor problem-solving
transaction), identified by (Problem Hierarchy, Problem Name, Step Name).
This is the natural item/anchor unit for (f)/(g) -- NOT one row, since the
same step is attempted by many different students, one row each. Response
field: "Correct First Attempt" (1/0). KC(<model>) cells and the matching
Opportunity(<model>) cells are "~~"-separated for multi-KC steps; the two
lists are assumed positionally aligned (rows where they mismatch in length
are dropped from that model's stats and counted in n_kc_opp_mismatch).

Usage:
    python triage_kdd.py <path/to/algebra_2008_2009_train.txt> \
        [--out-dir kt-mirt/_planning/triage] [--chunksize 500000] \
        [--nrows N] [--kc-models SubSkills,KTracedSkills,Rules]
"""
from __future__ import annotations

import argparse
import csv
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

ALL_KC_MODELS = ["SubSkills", "KTracedSkills", "Rules"]
BASE_COLS = ["Anon Student Id", "Problem Hierarchy", "Problem Name", "Step Name", "Correct First Attempt"]


def usecols_for(model_names):
    cols = list(BASE_COLS)
    for m in model_names:
        cols.append(f"KC({m})")
        cols.append(f"Opportunity({m})")
    return cols


def process_chunk(chunk: pd.DataFrame, aggregators: dict, model_names, growth_max_o: int) -> dict:
    item_id = (
        chunk["Problem Hierarchy"].fillna("") + "||"
        + chunk["Problem Name"].fillna("") + "||"
        + chunk["Step Name"].fillna("")
    )
    cfa = pd.to_numeric(chunk["Correct First Attempt"], errors="coerce")
    valid_cfa = cfa.notna() & cfa.isin([0, 1])
    n_invalid_cfa = int((~valid_cfa).sum())

    per_model = {}
    for model in model_names:
        kc_col = f"KC({model})"
        opp_col = f"Opportunity({model})"
        kc_raw = chunk[kc_col]
        has_kc = kc_raw.notna() & (kc_raw != "")
        mask = (valid_cfa & has_kc).to_numpy()
        n = int(mask.sum())
        if n == 0:
            per_model[model] = {"n_slots": 0, "n_kc_opp_mismatch": 0}
            continue

        students = chunk["Anon Student Id"].to_numpy()[mask]
        items = item_id.to_numpy()[mask]
        correct = cfa.to_numpy()[mask].astype(int)
        kc_lists = kc_raw[mask].str.split("~~")
        opp_lists = chunk[opp_col][mask].str.split("~~")

        len_kc = kc_lists.str.len().to_numpy()
        len_opp = opp_lists.str.len().to_numpy()
        match = len_kc == len_opp
        n_mismatch = int((~match).sum())

        long_df = pd.DataFrame({
            "student": students[match],
            "item": items[match],
            "correct": correct[match],
            "kc_list": kc_lists.to_numpy()[match],
            "opp_list": opp_lists.to_numpy()[match],
        })
        n_kept = len(long_df)
        if n_kept == 0:
            per_model[model] = {"n_slots": 0, "n_kc_opp_mismatch": n_mismatch}
            continue

        pair_counts_chunk = pair_counts_from_kc_lists(long_df["kc_list"])

        items_df = long_df[["item", "kc_list"]].drop_duplicates(subset="item")
        item_map = {row.item: frozenset(row.kc_list) for row in items_df.itertuples(index=False)}

        exploded = long_df.explode(["kc_list", "opp_list"]).rename(columns={"kc_list": "kc", "opp_list": "opp"})
        exploded["opp"] = pd.to_numeric(exploded["opp"], errors="coerce")
        exploded = exploded.dropna(subset=["kc", "opp"])
        exploded["opp"] = exploded["opp"].astype(int)

        marg = exploded.groupby("kc")["correct"].agg(["size", "sum"])
        opp_grp = exploded.groupby(["student", "kc"])["opp"].max()
        gmask = exploded["opp"].between(1, growth_max_o)
        gsub = exploded[gmask]
        ggrp = gsub.groupby(["kc", "opp"])["correct"].agg(["size", "sum"])

        agg = aggregators[model]
        agg.add_chunk_marginal(marg["size"].to_dict(), marg["sum"].to_dict())
        agg.add_chunk_opp_max({k: int(v) for k, v in opp_grp.to_dict().items()})
        agg.add_chunk_growth(ggrp["size"].to_dict(), ggrp["sum"].to_dict())
        agg.add_chunk_pairs(pair_counts_chunk)
        agg.add_items(item_map)
        agg.add_overall(n_kept, int(long_df["correct"].sum()), long_df["correct"].to_numpy())

        per_model[model] = {"n_slots": n_kept, "n_kc_opp_mismatch": n_mismatch}

    return {"n_invalid_cfa": n_invalid_cfa, "per_model": per_model}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_path")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--nrows", type=int, default=None, help="debug: limit total rows read")
    parser.add_argument("--kc-models", default=",".join(ALL_KC_MODELS))
    args = parser.parse_args()

    model_names = [m.strip() for m in args.kc_models.split(",") if m.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "_planning" / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregators = {m: KCModelAggregator(name=m) for m in model_names}

    t0 = time.time()
    reader = pd.read_csv(
        args.data_path, sep="\t", usecols=usecols_for(model_names), dtype=str,
        chunksize=args.chunksize, nrows=args.nrows, engine="c",
        quoting=csv.QUOTE_NONE, na_filter=True,
    )

    n_rows_seen = 0
    n_invalid_cfa_total = 0
    mismatch_totals = {m: 0 for m in model_names}
    for i, chunk in enumerate(reader):
        n_rows_seen += len(chunk)
        chunk_result = process_chunk(chunk, aggregators, model_names, GROWTH_MAX_OPPORTUNITY)
        n_invalid_cfa_total += chunk_result["n_invalid_cfa"]
        for m in model_names:
            mismatch_totals[m] += chunk_result["per_model"].get(m, {}).get("n_kc_opp_mismatch", 0)
        elapsed = time.time() - t0
        print(f"[triage_kdd] chunk {i + 1}: rows_seen={n_rows_seen:,} elapsed={elapsed:.1f}s", file=sys.stderr)

    results = {
        "bed": "kdd_algebra_2008_2009",
        "source_file": str(Path(args.data_path).resolve()),
        "unit_of_analysis": "step (one Cognitive Tutor problem-solving transaction; identified by "
                             "Problem Hierarchy||Problem Name||Step Name). NOT one 'question' -- see "
                             "module docstring.",
        "response_field": "Correct First Attempt (1=correct on first attempt, 0=incorrect/hint-first)",
        "sampling": "none (full file streamed)" if args.nrows is None else f"DEBUG RUN: first {args.nrows} rows only",
        "n_rows_read": n_rows_seen,
        "n_rows_invalid_cfa_dropped": n_invalid_cfa_total,
        "n_kc_opp_length_mismatch_dropped_by_model": mismatch_totals,
        "decoupling_definition": DECOUPLING_DEFINITION_TEXT,
        "kc_models": {m: aggregators[m].finalize() for m in model_names},
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "kdd_algebra_2008_2009_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
