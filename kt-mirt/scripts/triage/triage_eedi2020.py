"""STAGE-0 triage: Eedi NeurIPS 2020 (task 1/2 -- "Diagnostic Questions").

KC LEVEL CHOICE. subject_metadata.csv is a strict tree (SubjectId, Name,
ParentId, Level 0-3). question_metadata_task_1_2.csv tags each question
with a SubjectId LIST that is verified (programmatically, see below) to
always be a single root-to-leaf ANCESTOR CHAIN, not an independent
multi-skill tag set: every non-root subject in a question's list has its
parent also in that same list (checked over all 27,613 questions, holds
100%). The KC layer used here is the LEAF: for each question, the
subject id(s) at the MAXIMUM Level actually reached by its own tag list
(not a fixed global depth, since branches differ in depth). 25,219/27,613
questions (91.3%) have exactly one such leaf; 2,394 (8.7%) have >=2
tied-depth leaves (a question spanning two equally-specific sub-topics),
which is what gives this bed a genuine (if modest) multi-KC Q-matrix,
arity mean 1.10 / max 6, unlike Junyi's strictly single-parent tree.

ITEM/RESPONSE MODEL. Unit of analysis is one row of train_task_1_2.csv
(one question attempt): QuestionId, UserId, AnswerId, IsCorrect (0/1),
CorrectAnswer (1-4), AnswerValue (1-4, the option the student picked).
DateAnswered (needed for opportunity ordering) is not in train_task_1_2
itself and is joined in from answer_metadata_task_1_2.csv via AnswerId
(100% join coverage, verified). DateAnswered is a fixed-format
"YYYY-MM-DD HH:MM:SS.fff" string that sorts correctly lexicographically,
so (as in triage_junyi2020.py) no expensive to_datetime parse is done --
rows are sorted by (UserId, DateAnswered) as plain strings, stable tie-
break on original file order.

DATA-QUALITY NOTE: 48 of 27,613 questions (0.17%) have an inconsistent
CorrectAnswer value across rows (a known minor wrinkle of this export).
IsCorrect is used as-shipped (never recomputed from CorrectAnswer==
AnswerValue) everywhere in this script, so this does not affect (a)-(g).
It could in principle put a tiny amount of noise into the wrong-option
concentration block below (h), which is noted there.

BED-SPECIFIC ADDITION (h): beyond the shared (a)-(g) battery, this bed is
flagged in the task spec as the NRM/option-tracing candidate (NRM =
nominal response model, an IRT model of which specific wrong option a
respondent picks, not just right/wrong), so this script also reports (h)
the marginal answer-OPTION distribution and (h) per-question wrong-option
concentration: among a question's wrong (IsCorrect==0) answers, what
fraction pile onto that question's single most-common wrong option. High
concentration is the signal a misconception-cluster analysis (a same
distractor shared by many students -> a shared, nameable misconception)
would need; a flat 1/3-ish split across the 3 distractors would mean no
such signal is present raw, before any modeling.

Usage:
    python triage_eedi2020.py <path/to/eedi/data/train_data/train_task_1_2.csv> \
        <path/to/eedi/data/metadata/answer_metadata_task_1_2.csv> \
        <path/to/eedi/data/metadata/question_metadata_task_1_2.csv> \
        <path/to/eedi/data/metadata/subject_metadata.csv> \
        [--out-dir kt-mirt/_planning/triage]
"""
from __future__ import annotations

import argparse
import ast
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_common import (  # noqa: E402
    DECOUPLING_DEFINITION_TEXT,
    GROWTH_MAX_OPPORTUNITY,
    KCModelAggregator,
    category_distribution,
    fraction_at_least,
    pair_counts_from_kc_lists,
    quartiles,
    write_json,
)

MIN_WRONG_N_FOR_CONCENTRATION = 10
CONCENTRATION_MAJORITY_THRESHOLD = 0.5


def leaf_subject_set(subject_ids: list, level_map: dict) -> frozenset:
    levels = [level_map[s] for s in subject_ids]
    max_level = max(levels)
    return frozenset(str(s) for s, lvl in zip(subject_ids, levels) if lvl == max_level)


def check_chain_property(qm: pd.DataFrame, parent_map: dict) -> float:
    """Fraction of questions whose SubjectId list is a valid root-to-leaf ancestor chain."""
    def is_chain(lst):
        lst_set = set(lst)
        for s in lst:
            p = parent_map.get(s)
            if p is not None and not (isinstance(p, float) and np.isnan(p)) and int(p) not in lst_set:
                return False
        return True
    return float(qm["SubjectId"].apply(is_chain).mean())


def wrong_option_concentration(df_wrong: pd.DataFrame) -> dict:
    """Per-question: among wrong answers, the modal-distractor share. See module docstring (h)."""
    counts = df_wrong.groupby(["QuestionId", "AnswerValue"]).size().rename("n").reset_index()
    per_q_total = counts.groupby("QuestionId")["n"].sum()
    per_q_max = counts.groupby("QuestionId")["n"].max()
    modal_share = (per_q_max / per_q_total).rename("modal_wrong_share")
    gated = modal_share[per_q_total >= MIN_WRONG_N_FOR_CONCENTRATION]
    gated_totals = per_q_total[per_q_total >= MIN_WRONG_N_FOR_CONCENTRATION]
    pooled_weighted_mean = (
        float((gated * gated_totals).sum() / gated_totals.sum()) if len(gated) else None
    )
    return {
        "definition": "for each question, among its IsCorrect==0 rows, the fraction of those "
                      "wrong answers landing on the single most-common wrong AnswerValue "
                      "('modal_wrong_share'); 1/3 ~ flat/no-concentration, 1.0 ~ every wrong "
                      "answer picks the same one distractor.",
        "min_wrong_n_gate": MIN_WRONG_N_FOR_CONCENTRATION,
        "n_questions_total_with_ge1_wrong": int(len(per_q_total)),
        "n_questions_passing_gate": int(len(gated)),
        "modal_wrong_share_quartiles_unweighted": quartiles(gated),
        "modal_wrong_share_pooled_wrong_count_weighted_mean": pooled_weighted_mean,
        "frac_questions_majority_concentrated_ge_0.5": fraction_at_least(gated, CONCENTRATION_MAJORITY_THRESHOLD),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_csv")
    parser.add_argument("answer_metadata_csv")
    parser.add_argument("question_metadata_csv")
    parser.add_argument("subject_metadata_csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "_planning" / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sm = pd.read_csv(args.subject_metadata_csv)
    level_map = dict(zip(sm["SubjectId"], sm["Level"]))
    parent_map = dict(zip(sm["SubjectId"], sm["ParentId"]))

    qm = pd.read_csv(args.question_metadata_csv)
    qm["SubjectId"] = qm["SubjectId"].apply(ast.literal_eval)
    chain_frac = check_chain_property(qm, parent_map)
    qm["leaf"] = qm["SubjectId"].apply(lambda lst: leaf_subject_set(lst, level_map))
    arity_of_leaf = qm["leaf"].apply(len)
    q2leaf = dict(zip(qm["QuestionId"], qm["leaf"]))
    print(f"[triage_eedi2020] question_metadata: {len(qm)} questions, chain-property holds for "
          f"{chain_frac:.4f} of them, leaf arity mean={arity_of_leaf.mean():.3f} max={arity_of_leaf.max()}, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    df = pd.read_csv(args.train_csv)
    n_rows_total = len(df)
    am = pd.read_csv(args.answer_metadata_csv, usecols=["AnswerId", "DateAnswered"])
    df = df.merge(am, on="AnswerId", how="left")
    n_missing_date = int(df["DateAnswered"].isna().sum())
    print(f"[triage_eedi2020] train_task_1_2: {n_rows_total:,} rows, {df['UserId'].nunique()} users, "
          f"{df['QuestionId'].nunique()} questions, {n_missing_date} missing DateAnswered after join, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    df["leaf"] = df["QuestionId"].map(q2leaf)
    n_unmapped = int(df["leaf"].isna().sum())
    df = df.dropna(subset=["leaf", "DateAnswered"])

    # Chronological order per user: DateAnswered is "YYYY-MM-DD HH:MM:SS.fff", sorts correctly
    # as a plain string (as in triage_junyi2020.py -- no to_datetime parse needed).
    df.sort_values(["UserId", "DateAnswered"], kind="stable", inplace=True)
    print(f"[triage_eedi2020] sorted for chronological per-user order, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    agg = KCModelAggregator(name="subject_leaf")

    # (a) overall + response distribution: one row of train_task_1_2 IS one real interaction
    # (no KC-driven duplication -- the multi-leaf explode happens only below, for b/c/d/e).
    agg.add_overall(len(df), int(df["IsCorrect"].sum()), df["IsCorrect"].to_numpy())

    # (e) KC-pair co-occurrence, from the PRE-explode per-slot leaf sets (same "list per slot"
    # input pattern as KDD's ~~-split / EdNet's ;-split / xes3g5m's _-split).
    kc_list_series = df["leaf"].apply(lambda s: sorted(s))
    pair_counts = pair_counts_from_kc_lists(kc_list_series)
    agg.add_chunk_pairs(pair_counts)

    # (b)/(c)/(d): explode to (interaction, KC) grain, preserving the pre-sorted chronological
    # per-user row order (explode never reorders rows, only duplicates in place).
    long_df = df[["UserId", "leaf", "IsCorrect"]].explode("leaf").rename(columns={"leaf": "kc"})
    long_df["opp"] = long_df.groupby(["UserId", "kc"], sort=False)["IsCorrect"].cumcount() + 1

    marg = long_df.groupby("kc")["IsCorrect"].agg(["size", "sum"])
    agg.add_chunk_marginal(marg["size"].to_dict(), marg["sum"].to_dict())

    opp_max = long_df.groupby(["UserId", "kc"])["opp"].max()
    agg.add_chunk_opp_max({key: int(v) for key, v in opp_max.items()})

    gsub = long_df[long_df["opp"].between(1, GROWTH_MAX_OPPORTUNITY)]
    ggrp = gsub.groupby(["kc", "opp"])["IsCorrect"].agg(["size", "sum"])
    agg.add_chunk_growth(ggrp["size"].to_dict(), ggrp["sum"].to_dict())

    # (f)/(g): static Q-matrix from the FULL question_metadata bank (27,613 questions), same
    # full-bank convention as EdNet/xes3g5m/junyi2020.
    item_to_kcset = {int(qid): kcset for qid, kcset in q2leaf.items()}
    agg.add_items(item_to_kcset)

    # (h) bed-specific: answer-option distribution + per-question wrong-option concentration.
    answer_value_dist = category_distribution(Counter(df["AnswerValue"].tolist()))
    correct_answer_dist = category_distribution(Counter(df["CorrectAnswer"].tolist()))
    n_inconsistent_correct_answer = int((df.groupby("QuestionId")["CorrectAnswer"].nunique() != 1).sum())
    conc = wrong_option_concentration(df[df["IsCorrect"] == 0][["QuestionId", "AnswerValue"]])

    results = {
        "bed": "eedi2020_task1_2",
        "source_files": {
            "train": str(Path(args.train_csv).resolve()),
            "answer_metadata": str(Path(args.answer_metadata_csv).resolve()),
            "question_metadata": str(Path(args.question_metadata_csv).resolve()),
            "subject_metadata": str(Path(args.subject_metadata_csv).resolve()),
        },
        "kc_level_choice": {
            "chosen": "leaf of the per-question SubjectId ancestor chain (deepest level reached "
                      "by that question's own tags, not a fixed global depth)",
            "justification": "subject_metadata.csv is a strict tree (Level 0-3); every question's "
                              "SubjectId list is verified to be a single root-to-leaf ancestor "
                              "chain (chain-property holds for "
                              f"{chain_frac:.4f} of {len(qm)} questions), so 'the leaf' is the "
                              "natural KC-equivalent rather than an arbitrary hierarchy cut. "
                              "91.3% of questions have exactly one leaf at their own max depth; "
                              "8.7% have >=2 tied-depth leaves, which is what gives this bed a "
                              "genuine (if modest) multi-KC Q-matrix (arity mean "
                              f"{float(arity_of_leaf.mean()):.3f}, max {int(arity_of_leaf.max())}) "
                              "unlike Junyi's strictly single-parent tree.",
            "n_distinct_leaf_kc_in_bank": int(len(set().union(*q2leaf.values()))) if q2leaf else 0,
        },
        "unit_of_analysis": "one row of train_task_1_2.csv (one question attempt)",
        "response_field": "IsCorrect (0/1, as shipped -- never recomputed from CorrectAnswer==AnswerValue)",
        "opportunity_ordering": "DateAnswered joined in from answer_metadata_task_1_2.csv via AnswerId "
                                "(100% join coverage); sorted as a plain string (ISO-ish fixed format, "
                                "sorts correctly), stable tie-break on original file row order.",
        "sampling": "none (full file streamed/loaded; 15.87M-row scale ran end-to-end well inside budget)",
        "n_rows_total": n_rows_total,
        "n_rows_missing_date_dropped": n_missing_date,
        "n_rows_kc_unmapped_dropped": n_unmapped,
        "n_users": int(df["UserId"].nunique()),
        "n_questions_in_bank": int(len(qm)),
        "n_questions_attempted_in_log": int(df["QuestionId"].nunique()),
        "data_quality_note": {
            "n_questions_with_inconsistent_correct_answer": n_inconsistent_correct_answer,
            "frac_of_bank": n_inconsistent_correct_answer / len(qm),
            "handling": "IsCorrect used as-shipped throughout (a)-(g); only (h)'s wrong-option "
                        "tally could carry a small amount of noise from this, and does not "
                        "require a single canonical CorrectAnswer to compute.",
        },
        "decoupling_definition": DECOUPLING_DEFINITION_TEXT,
        "kc_models": {"subject_leaf": agg.finalize()},
        "h_answer_option_distribution": {
            "chosen_answer_value_distribution": answer_value_dist,
            "correct_answer_position_distribution": correct_answer_dist,
            "note": "AnswerValue/CorrectAnswer in {1,2,3,4}; both reported to check whether option "
                    "position is roughly balanced (no baked-in 'always pick option 3' shortcut).",
        },
        "h_wrong_option_concentration": conc,
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "eedi2020_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
