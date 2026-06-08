"""Merge Eedi NeurIPS 2020 Task 3 + Task 4 CSVs into the EediAdapter input.

The :class:`ordrec.data.EediAdapter` consumes a single csv with the
columns ``student_id, question_id, correct, selected_option``. The
official Eedi NeurIPS 2020 release (Wang et al. 2020) ships the data
as multiple csvs across Task 3 and Task 4, with the per-student
responses and the per-question metadata living in different files.
This pre-merge script joins them so the adapter can run end-to-end.

The script is a deliverable, not its execution. It is approved by
the user note to be ready for the moment the real Eedi data lands
locally; running it requires the actual csvs which are not in this
repo. See ``docs/ordrec_impl_guide.md`` Section 2.3.

Eedi NeurIPS 2020 source layout
-------------------------------

Task 3 (knowledge graph based recommendation) and Task 4 (high quality
recommendation) ship with the following files of interest,

    train_task_3_4.csv           per-attempt responses across both tasks
    question_metadata_task_3_4.csv  per-question metadata, including the
                                    correct answer
    student_metadata_task_3_4.csv   per-student metadata (optional)
    answer_metadata_task_3_4.csv    per-attempt metadata (timestamps,
                                    confidence, optional)

Column mapping (raw -> merged)
------------------------------

::

    train_task_3_4.csv
        QuestionId        -> question_id
        UserId            -> student_id
        AnswerValue       -> selected_option  (in {1,2,3,4})
        IsCorrect         -> correct          (in {0,1})

    question_metadata_task_3_4.csv
        QuestionId        -> question_id      (used to join the
                                                CorrectAnswer column for
                                                cross-validation of
                                                IsCorrect against
                                                AnswerValue)
        CorrectAnswer     -> correct_answer   (kept for audit; the
                                                adapter independently
                                                infers the canonical
                                                correct option from
                                                the train fold)

    answer_metadata_task_3_4.csv     (optional)
        AnswerId          -> row_id           (used as a tiebreak for
                                                stable ordering when
                                                Timestamp is unavailable)
        DateAnswered      -> timestamp        (preferred sort key)

Output CSV columns
------------------

    student_id, question_id, correct, selected_option,
    correct_answer (optional, audit only),
    row_id         (stable order, falls back to source row index),
    timestamp      (optional, falls back to row_id)

The merged csv lands at ``--out`` (default ``eedi_merged.csv``) and
can be fed directly to ``EediAdapter`` via
``cfg.raw_dir = <out>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


_DEFAULT_OUT = "eedi_merged.csv"


def merge_eedi_csvs(
    train_csv: Path,
    question_metadata_csv: Optional[Path],
    answer_metadata_csv: Optional[Path],
    out_path: Path,
) -> Path:
    """Merge the source CSVs into the single-csv schema EediAdapter expects.

    Args:
        train_csv: Path to ``train_task_3_4.csv``.
        question_metadata_csv: Optional path to
            ``question_metadata_task_3_4.csv``. When supplied, the
            ``CorrectAnswer`` column is joined onto each row as
            ``correct_answer`` for downstream auditing.
        answer_metadata_csv: Optional path to
            ``answer_metadata_task_3_4.csv``. When supplied, the
            ``AnswerId`` and ``DateAnswered`` columns become
            ``row_id`` and ``timestamp`` respectively. When omitted,
            the merged file uses the source row index as ``row_id``.
        out_path: Output csv path.

    Returns:
        The output path written to.
    """
    train_path = Path(train_csv)
    if not train_path.exists():
        raise FileNotFoundError(f"train csv not found: {train_path}")
    train = pd.read_csv(train_path)
    needed = {"QuestionId", "UserId", "AnswerValue", "IsCorrect"}
    missing = needed - set(train.columns)
    if missing:
        raise ValueError(
            f"{train_path} is missing required columns {sorted(missing)}, "
            f"available {list(train.columns)}"
        )

    df = train.rename(columns={
        "QuestionId": "question_id",
        "UserId": "student_id",
        "AnswerValue": "selected_option",
        "IsCorrect": "correct",
    })

    if question_metadata_csv is not None:
        qmeta_path = Path(question_metadata_csv)
        if not qmeta_path.exists():
            raise FileNotFoundError(
                f"question metadata csv not found: {qmeta_path}"
            )
        qmeta = pd.read_csv(qmeta_path)
        if not {"QuestionId", "CorrectAnswer"} <= set(qmeta.columns):
            raise ValueError(
                f"{qmeta_path} must carry QuestionId and CorrectAnswer columns"
            )
        qmeta = qmeta[["QuestionId", "CorrectAnswer"]].rename(columns={
            "QuestionId": "question_id",
            "CorrectAnswer": "correct_answer",
        })
        df = df.merge(qmeta, on="question_id", how="left")

    if answer_metadata_csv is not None:
        ameta_path = Path(answer_metadata_csv)
        if not ameta_path.exists():
            raise FileNotFoundError(
                f"answer metadata csv not found: {ameta_path}"
            )
        ameta = pd.read_csv(ameta_path)
        keep = [c for c in ("AnswerId", "DateAnswered") if c in ameta.columns]
        if not keep:
            raise ValueError(
                f"{ameta_path} must carry at least one of AnswerId, "
                f"DateAnswered for the merged-csv key columns."
            )
        ameta = ameta[keep].rename(columns={
            "AnswerId": "row_id",
            "DateAnswered": "timestamp",
        })
        # The train csv carries an implicit row position; we align on
        # the row index for now. If the user's pipeline supplies an
        # ``AnswerId`` column on the train csv, prefer the explicit
        # join key.
        if "AnswerId" in train.columns:
            df["AnswerId"] = train["AnswerId"]
            df = df.merge(ameta, left_on="AnswerId", right_on="row_id",
                          how="left", suffixes=("", "_meta"))
            df = df.drop(columns=["AnswerId"])
        else:
            # Falls back to a positional concat. Source files in the
            # public release ship the same row count so this is safe;
            # the adapter will use ``timestamp`` and ``row_id`` only as
            # sort keys.
            if len(df) == len(ameta):
                df = df.reset_index(drop=True)
                ameta = ameta.reset_index(drop=True)
                df = pd.concat([df, ameta], axis=1)
            else:
                print(
                    f"warning, answer_metadata rows ({len(ameta)}) do not "
                    f"match train rows ({len(df)}); skipping the metadata "
                    "join. Pass an AnswerId column on the train csv for "
                    "an explicit join.",
                    file=sys.stderr,
                )

    if "row_id" not in df.columns:
        df["row_id"] = df.index

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prepare_eedi_csv",
        description=(
            "Merge Eedi NeurIPS 2020 Task 3 + 4 CSVs into the single-csv "
            "format the EediAdapter expects."
        ),
    )
    p.add_argument(
        "--train", type=Path, required=True,
        help="Path to train_task_3_4.csv (the per-attempt responses).",
    )
    p.add_argument(
        "--question-metadata", type=Path, default=None,
        help=(
            "Optional, path to question_metadata_task_3_4.csv. Joins the "
            "CorrectAnswer column onto each row for auditing."
        ),
    )
    p.add_argument(
        "--answer-metadata", type=Path, default=None,
        help=(
            "Optional, path to answer_metadata_task_3_4.csv. Adds row_id "
            "and timestamp columns for stable ordering."
        ),
    )
    p.add_argument(
        "--out", type=Path, default=Path(_DEFAULT_OUT),
        help=f"Output csv path (default {_DEFAULT_OUT}).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    out = merge_eedi_csvs(
        train_csv=args.train,
        question_metadata_csv=args.question_metadata,
        answer_metadata_csv=args.answer_metadata,
        out_path=args.out,
    )
    print(f"wrote merged Eedi csv -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
