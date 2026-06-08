"""``EdNetAdapter``, K=4 ordinal from ``(correctness, response_time)``.

EdNet (Choi et al. 2020 AIED) does not ship distractor information.
We fall back to the polytomous-from-binary literature pattern
(Pelanek 2017, Khajah et al. 2016), folding response time into the
ordinal label. See ``docs/ordrec_impl_guide.md`` Section 2.4.

Coercion table.

::

                       fast (rt <= median_q)        slow (rt > median_q)
    correctness = 0    1 (incorrect, fast)          0 (incorrect, slow)
    correctness = 1    3 (correct,   fast)          2 (correct,   slow)

The ordering ``incorrect slow < incorrect fast < correct slow < correct
fast`` follows the rationale that genuine struggle without success is
the lowest mastery signal, fast incorrect is a guess-and-move-on,
slow correct is mastery present but not automatic, and fast correct is
the strongest mastery signal.

Per-question median, not global median. A global median conflates
item difficulty with student speed. The median is computed on train
only and persisted as ``coercion_artefacts.json["rt_median_per_q"]``.
Cold items on test fall back to the global median computed on the
train fold, with the count logged.

KT3 vs KT4. EdNet KT3 lacks a hint-usage field. The adapter detects
the presence of a ``hint_used`` (or ``used_hint``) column at load
time and records the version under
``coercion_artefacts.json["ednet_level"]``. K=5 with a hint axis is
left as a follow-up KT4 sub-experiment, the current adapter coerces
KT4 down to K=4 by ignoring the hint column. See the
``Notes`` section for the KT3/KT4 column expectations.

Notes
-----
Expected raw input is a single csv (or a directory containing one)
with at least these columns,

    student_id       int or str, mapped to a contiguous 0-based index
    question_id      int or str, mapped to a contiguous 1-based id
    correct          {0, 1}
    response_time    numeric, milliseconds or seconds (units are
                     inferred per-question by taking the median, so
                     the absolute scale is irrelevant)

Optional column ``hint_used`` (KT4) is acknowledged in
``coercion_artefacts.json`` but does not change the recode.
Aliases accepted, ``is_correct`` for ``correct``,
``user_id`` for ``student_id``, ``elapsed_time`` for
``response_time``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .base import AdapterConfig, OrdinalDatasetBase, _SPLIT_CODES
from .schema import (
    COERCION_FILENAME,
    METADATA_FILENAME,
    SEQUENCES_FILENAME,
    dump_coercion_artefacts,
    dump_metadata,
    dump_sequences,
    load_coercion_artefacts,
    load_metadata,
    load_sequences,
    validate_metadata,
    validate_sequences,
)
from .split import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID, make_split


N_CATEGORIES_EDNET = 4

# Ordinal codes (ascending mastery signal).
ORD_INCORRECT_SLOW = 0
ORD_INCORRECT_FAST = 1
ORD_CORRECT_SLOW = 2
ORD_CORRECT_FAST = 3

_CODE_TO_SPLIT = {SPLIT_TRAIN: "train", SPLIT_VALID: "valid", SPLIT_TEST: "test"}


class EdNetAdapter(OrdinalDatasetBase):
    """K=4 EdNet adapter using per-question response-time quadrants."""

    # --------------------------- Raw input ---------------------------------

    def _locate_csv(self) -> Path:
        raw = Path(self.cfg.raw_dir)
        if raw.is_file() and raw.suffix.lower() == ".csv":
            return raw
        if not raw.exists():
            raise FileNotFoundError(f"EdNet raw_dir does not exist: {raw}")
        csvs = sorted(raw.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"EdNet adapter expected a *.csv under {raw}, found none."
            )
        if len(csvs) > 1:
            preferred = raw / "ednet.csv"
            if preferred.exists():
                return preferred
            raise ValueError(
                f"Multiple csvs under {raw}, set cfg.raw_dir to a specific file."
            )
        return csvs[0]

    def _read_csv(self) -> pd.DataFrame:
        path = self._locate_csv()
        df = pd.read_csv(path)
        renames: Dict[str, str] = {}
        if "is_correct" in df.columns and "correct" not in df.columns:
            renames["is_correct"] = "correct"
        if "user_id" in df.columns and "student_id" not in df.columns:
            renames["user_id"] = "student_id"
        if "elapsed_time" in df.columns and "response_time" not in df.columns:
            renames["elapsed_time"] = "response_time"
        if "used_hint" in df.columns and "hint_used" not in df.columns:
            renames["used_hint"] = "hint_used"
        if renames:
            df = df.rename(columns=renames)

        required = ("student_id", "question_id", "correct", "response_time")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"EdNet csv {path} missing required columns {missing}, "
                f"available {list(df.columns)}"
            )

        sort_cols: List[str] = []
        for col in ("timestamp", "row_id", "order"):
            if col in df.columns:
                sort_cols.append(col)
        if not sort_cols:
            sort_cols = ["student_id", "question_id"]
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        return df

    # --------------------------- Materialise -------------------------------

    def materialise(self) -> None:
        df = self._read_csv()
        has_hint = "hint_used" in df.columns
        ednet_level = "kt4" if has_hint else "kt3"

        student_ids = pd.unique(df["student_id"])
        student_id_map: Dict[Any, int] = {
            sid: i for i, sid in enumerate(student_ids.tolist())
        }
        n_students = len(student_id_map)

        question_ids_unique = pd.unique(df["question_id"])
        question_id_map: Dict[Any, int] = {
            qid: i + 1 for i, qid in enumerate(question_ids_unique.tolist())
        }
        n_questions = len(question_id_map)

        df["_student"] = df["student_id"].map(student_id_map).astype(np.int64)
        df["_item"] = df["question_id"].map(question_id_map).astype(np.int64)
        df["_correct"] = df["correct"].astype(int).clip(0, 1)
        # Missing response time is filled later via ``label_as_slow``;
        # to detect missingness we coerce non-finite to NaN here.
        rt_numeric = pd.to_numeric(df["response_time"], errors="coerce")
        df["_rt"] = rt_numeric.astype(float)

        # Per-user split BEFORE chunking.
        user_codes = make_split(
            n_students=n_students,
            test_frac=self.cfg.test_frac,
            valid_frac=self.cfg.valid_frac,
            split_seed=self.cfg.split_seed,
        )
        train_students = set(np.where(user_codes == SPLIT_TRAIN)[0].tolist())

        # Per-question median computed on the train fold only. Missing
        # rt rows are excluded from the median computation; they fall
        # to the "slow" bucket downstream.
        train_mask = df["_student"].isin(train_students)
        train_df = df[train_mask & df["_rt"].notna()]
        if len(train_df) == 0:
            raise ValueError(
                "EdNet adapter found zero train-fold rows with valid response_time."
            )
        global_train_median = float(train_df["_rt"].median())

        rt_median_per_q: Dict[int, float] = {}
        for item, sub in train_df.groupby("_item"):
            rt_median_per_q[int(item)] = float(sub["_rt"].median())

        # Build per-student sequences (preserve dataframe row order
        # within a student).
        questions_per_stu: List[List[int]] = [[] for _ in range(n_students)]
        responses_per_stu: List[List[int]] = [[] for _ in range(n_students)]
        cold_item_fallbacks = 0
        missing_rt_count = 0

        df_arr = df[["_student", "_item", "_correct", "_rt"]].to_numpy()
        for stu_f, item_f, correct_f, rt_f in df_arr:
            stu = int(stu_f)
            item = int(item_f)
            correct = int(correct_f)
            rt_val = float(rt_f) if not np.isnan(rt_f) else float("nan")

            if np.isnan(rt_val):
                # Missing rt: label as slow.
                is_fast = False
                missing_rt_count += 1
            else:
                median_q = rt_median_per_q.get(item)
                if median_q is None:
                    median_q = global_train_median
                    cold_item_fallbacks += 1
                is_fast = rt_val <= median_q

            if correct == 1 and is_fast:
                code = ORD_CORRECT_FAST
            elif correct == 1 and not is_fast:
                code = ORD_CORRECT_SLOW
            elif correct == 0 and is_fast:
                code = ORD_INCORRECT_FAST
            else:
                code = ORD_INCORRECT_SLOW

            questions_per_stu[stu].append(item)
            responses_per_stu[stu].append(code)

        out_records: List[Dict[str, Any]] = []
        for stu_i in range(n_students):
            q_seq = questions_per_stu[stu_i]
            if len(q_seq) < self.cfg.min_seq_len:
                continue
            out_records.append({
                "questions": q_seq,
                "responses": responses_per_stu[stu_i],
                "split": _CODE_TO_SPLIT[int(user_codes[stu_i])],
            })

        if not out_records:
            raise ValueError(
                "EdNetAdapter produced zero sequences after min_seq_len filter."
            )

        codes_surviving = np.array(
            [_SPLIT_CODES[r["split"]] for r in out_records], dtype=np.int8
        )
        n_train_kept = int((codes_surviving == SPLIT_TRAIN).sum())
        n_valid_kept = int((codes_surviving == SPLIT_VALID).sum())
        n_test_kept = int((codes_surviving == SPLIT_TEST).sum())
        total = max(1, n_train_kept + n_valid_kept + n_test_kept)
        seq_lens = [len(r["questions"]) for r in out_records]
        seq_len_range = [int(min(seq_lens)), int(max(seq_lens))]

        meta: Dict[str, Any] = {
            "dataset_name": self.cfg.name,
            "adapter_class": type(self).__name__,
            "n_students": n_students,
            "n_questions": n_questions,
            "n_categories": N_CATEGORIES_EDNET,
            "n_kcs": 0,
            "seq_len_range": seq_len_range,
            "ordinal_coercion_method": "correctness_x_time_quadrant",
            "splits": {
                "split_seed": int(self.cfg.split_seed),
                "train_frac": n_train_kept / total,
                "valid_frac": n_valid_kept / total,
                "test_frac": n_test_kept / total,
                "n_train": n_train_kept,
                "n_valid": n_valid_kept,
                "n_test": n_test_kept,
            },
            "question_id_map": {str(k): int(v) for k, v in question_id_map.items()},
            "coercion_artefacts_path": COERCION_FILENAME,
        }

        coercion_payload: Dict[str, Any] = {
            "method": "correctness_x_time_quadrant",
            "ednet_level": ednet_level,
            "has_hint_field": bool(has_hint),
            "rt_median_per_q": {
                str(int(k)): float(v) for k, v in sorted(rt_median_per_q.items())
            },
            "global_train_median_rt": float(global_train_median),
            "n_cold_item_fallbacks": int(cold_item_fallbacks),
            "n_missing_rt_labeled_as_slow": int(missing_rt_count),
            "ordinal_map": {
                "incorrect_slow": ORD_INCORRECT_SLOW,
                "incorrect_fast": ORD_INCORRECT_FAST,
                "correct_slow": ORD_CORRECT_SLOW,
                "correct_fast": ORD_CORRECT_FAST,
            },
            "hint_handling": (
                "kt3, hint field absent" if not has_hint
                else "kt4, hint_used column observed but ignored for K=4 coercion"
            ),
        }

        validate_metadata(meta)
        validate_sequences(
            out_records, n_questions=n_questions, n_categories=N_CATEGORIES_EDNET
        )

        out_dir = self.artefact_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_sequences(out_records, out_dir / SEQUENCES_FILENAME)
        dump_metadata(meta, out_dir / METADATA_FILENAME)
        dump_coercion_artefacts(coercion_payload, out_dir / COERCION_FILENAME)

    # --------------------------- Load -------------------------------------

    def load(self) -> None:
        artefact = self.artefact_dir
        records = load_sequences(artefact / SEQUENCES_FILENAME)
        meta = load_metadata(artefact / METADATA_FILENAME)
        validate_metadata(meta)
        validate_sequences(
            records,
            n_questions=int(meta["n_questions"]),
            n_categories=int(meta["n_categories"]),
        )

        self._questions = [list(map(int, rec["questions"])) for rec in records]
        self._responses = [list(map(int, rec["responses"])) for rec in records]
        self._student_split = np.asarray(
            [_SPLIT_CODES[rec["split"]] for rec in records], dtype=np.int8,
        )
        self._metadata = meta
        self._q_matrix = None

        coercion_path = artefact / COERCION_FILENAME
        if coercion_path.exists():
            self._coercion = load_coercion_artefacts(coercion_path)
        else:
            self._coercion = {}


__all__ = ["EdNetAdapter", "N_CATEGORIES_EDNET"]
