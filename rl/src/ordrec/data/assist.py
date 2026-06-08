"""``AssistAdapter``, K=2 identity passthrough for ASSISTments 2009.

The point of this adapter is not to push K=2 prediction accuracy. It
is to provide the binary ablation control against which the K=4
ordinal datasets must outperform on the same backbone, encoder,
reward and eval pipeline. See ``docs/ordrec_impl_guide.md`` Section
2.5.

Coercion is identity, ``y_ord = correct`` clipped to ``{0, 1}``,
``metadata["n_categories"] = 2`` and
``ordinal_coercion_method = "binary"``. User-level deterministic split
via :func:`ordrec.data.split.make_split`.

Expected raw input is a single csv (or a directory containing one)
with these columns,

    student_id       int or str, mapped to a contiguous 0-based index
    question_id      int or str, mapped to a contiguous 1-based id
    correct          {0, 1}

Aliases accepted, ``is_correct`` for ``correct``, ``user_id`` for
``student_id``, ``skill_id`` is acknowledged but unused (the
adapter does not currently emit a Q-matrix; that follows in a later
milestone when the skill mapping is needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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


N_CATEGORIES_ASSIST = 2

_CODE_TO_SPLIT = {SPLIT_TRAIN: "train", SPLIT_VALID: "valid", SPLIT_TEST: "test"}


class AssistAdapter(OrdinalDatasetBase):
    """K=2 ASSISTments 2009 adapter, identity passthrough on correctness."""

    # --------------------------- Raw input ---------------------------------

    def _locate_csv(self) -> Path:
        raw = Path(self.cfg.raw_dir)
        if raw.is_file() and raw.suffix.lower() == ".csv":
            return raw
        if not raw.exists():
            raise FileNotFoundError(f"ASSIST raw_dir does not exist: {raw}")
        csvs = sorted(raw.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"ASSIST adapter expected a *.csv under {raw}, found none."
            )
        if len(csvs) > 1:
            preferred = raw / "assist.csv"
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
        if "problem_id" in df.columns and "question_id" not in df.columns:
            renames["problem_id"] = "question_id"
        if renames:
            df = df.rename(columns=renames)

        required = ("student_id", "question_id", "correct")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"ASSIST csv {path} missing required columns {missing}, "
                f"available {list(df.columns)}"
            )

        sort_cols: List[str] = []
        for col in ("order_id", "timestamp", "row_id"):
            if col in df.columns:
                sort_cols.append(col)
        if not sort_cols:
            sort_cols = ["student_id", "question_id"]
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        return df

    # --------------------------- Materialise -------------------------------

    def materialise(self) -> None:
        df = self._read_csv()

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

        user_codes = make_split(
            n_students=n_students,
            test_frac=self.cfg.test_frac,
            valid_frac=self.cfg.valid_frac,
            split_seed=self.cfg.split_seed,
        )

        questions_per_stu: List[List[int]] = [[] for _ in range(n_students)]
        responses_per_stu: List[List[int]] = [[] for _ in range(n_students)]
        for stu_f, item_f, correct_f in df[
            ["_student", "_item", "_correct"]
        ].to_numpy():
            stu = int(stu_f)
            questions_per_stu[stu].append(int(item_f))
            responses_per_stu[stu].append(int(correct_f))

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
                "AssistAdapter produced zero sequences after min_seq_len filter."
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
            "n_categories": N_CATEGORIES_ASSIST,
            "n_kcs": 0,
            "seq_len_range": seq_len_range,
            "ordinal_coercion_method": "binary",
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
            "method": "binary",
            "note": (
                "identity passthrough, y_ord = correct in {0, 1}. "
                "n_categories = 2."
            ),
        }

        validate_metadata(meta)
        validate_sequences(
            out_records, n_questions=n_questions, n_categories=N_CATEGORIES_ASSIST
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


__all__ = ["AssistAdapter", "N_CATEGORIES_ASSIST"]
