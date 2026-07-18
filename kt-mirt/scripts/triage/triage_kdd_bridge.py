"""STAGE-0 triage: KDD Cup 2010 Bridge to Algebra 2008-2009 (challenge set).

Same DataShop transaction-log format as the Algebra 2008-2009 set that
`triage_kdd.py` already covers (see that module's docstring for the unit
of analysis, KC(<model>)/Opportunity(<model>) cell format, and the
step-grain Q-matrix convention). This is a thin bed-specific wrapper, not
a fork: it imports `process_chunk`/`usecols_for` from `triage_kdd` so the
parsing and aggregation logic is defined exactly once and shared across
both KDD beds, and only supplies this bed's file path, KC-model list, and
output filename.

KC-MODEL CHOICE. Bridge to Algebra 2008-2009's raw header carries only two
KC(<model>) columns -- KC(SubSkills) and KC(KTracedSkills) -- there is no
KC(Rules) column at all in this file (checked directly against the header
row), unlike the Algebra 2008-2009 set which ships all three. This matches
the task framing (KTracedSkills primary, SubSkills secondary) exactly, so
both available models are run and nothing is silently dropped to hit that
framing.

KEY QUESTION THIS BED ANSWERS. The original triage's avenue map favored
KDD Algebra 2006-2007 (G1) partly on KTracedSkills' near-1-to-1 KC arity
and high top-30 decoupling (see the Algebra 2008-2009 section of
triage_report.md for the 08-09 numbers this triage was itself checked
against). Bridge to Algebra 2008-2009 is a same-vintage, larger (~20M row)
companion set from the same KDD Cup; this run checks whether its
KTracedSkills tagging keeps that same profile or diverges.

Usage:
    python triage_kdd_bridge.py <path/to/bridge_to_algebra_2008_2009_train.txt> \
        [--out-dir kt-mirt/_planning/triage] [--chunksize 500000] \
        [--nrows N] [--kc-models KTracedSkills,SubSkills]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_common import DECOUPLING_DEFINITION_TEXT, GROWTH_MAX_OPPORTUNITY, KCModelAggregator, write_json  # noqa: E402
from triage_kdd import process_chunk, usecols_for  # noqa: E402

import pandas as pd  # noqa: E402

DEFAULT_KC_MODELS = ["KTracedSkills", "SubSkills"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_path")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--nrows", type=int, default=None, help="debug: limit total rows read")
    parser.add_argument("--kc-models", default=",".join(DEFAULT_KC_MODELS))
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
        print(f"[triage_kdd_bridge] chunk {i + 1}: rows_seen={n_rows_seen:,} elapsed={elapsed:.1f}s", file=sys.stderr)

    results = {
        "bed": "kdd_bridge_to_algebra_2008_2009",
        "source_file": str(Path(args.data_path).resolve()),
        "unit_of_analysis": "step (one Cognitive Tutor problem-solving transaction; identified by "
                             "Problem Hierarchy||Problem Name||Step Name). NOT one 'question' -- see "
                             "triage_kdd.py module docstring (shared parsing logic).",
        "response_field": "Correct First Attempt (1=correct on first attempt, 0=incorrect/hint-first)",
        "sampling": "none (full file streamed)" if args.nrows is None else f"DEBUG RUN: first {args.nrows} rows only",
        "kc_model_note": "Only KC(SubSkills) and KC(KTracedSkills) exist in this file's header -- unlike "
                          "Algebra 2008-2009, there is no KC(Rules) column at all for Bridge to Algebra "
                          "2008-2009, so nothing was dropped to match the task's KTracedSkills-primary/"
                          "SubSkills-secondary framing; both models present are run.",
        "n_rows_read": n_rows_seen,
        "n_rows_invalid_cfa_dropped": n_invalid_cfa_total,
        "n_kc_opp_length_mismatch_dropped_by_model": mismatch_totals,
        "decoupling_definition": DECOUPLING_DEFINITION_TEXT,
        "kc_models": {m: aggregators[m].finalize() for m in model_names},
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "kdd_bridge_2008_2009_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
