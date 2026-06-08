"""Synthetic adapter, wraps an ma-irt static-GPCM artefact.

The synthetic generator at ``ma-irt/utils/datagen/static.py`` already
produces a ``sequences.json`` plus ``metadata.json`` pair in our raw
schema. The job of this adapter is to add the OrdRec contract on top,
specifically a per-user deterministic split and the canonical metadata
block (``adapter_class``, ``ordinal_coercion_method``, ``splits`` and
``question_id_map``). It exists so that downstream code can treat the
ma-irt synthetic dataset the same way it treats Eedi or EdNet, without
branching on dataset identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base import AdapterConfig, OrdinalDatasetBase, _SPLIT_CODES
from .schema import (
    COERCION_METHODS,
    METADATA_FILENAME,
    SEQUENCES_FILENAME,
    dump_metadata,
    dump_sequences,
    load_metadata,
    load_sequences,
    validate_metadata,
    validate_sequences,
)
from .split import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    _chunk_sequences,
    make_split,
)


_CODE_TO_SPLIT = {
    SPLIT_TRAIN: "train",
    SPLIT_VALID: "valid",
    SPLIT_TEST: "test",
}


class SyntheticAdapter(OrdinalDatasetBase):
    """Wraps an ma-irt synthetic-generator directory.

    Expected raw layout (``cfg.raw_dir``):

        sequences.json          list of {"questions": [...], "responses": [...]}
        metadata.json           ma-irt format with at least n_questions,
                                n_categories, n_students
        true_irt_parameters.json (optional, ignored by the adapter)

    Materialised layout (``cfg.out_dir / cfg.name``):

        sequences.json          our schema, with a "split" field per record
        metadata.json           COMMON_RECORD_SCHEMA fields
        coercion_artefacts.json (empty, kept for schema parity)
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def materialise(self) -> None:
        raw_dir = Path(self.cfg.raw_dir)
        raw_seq = raw_dir / SEQUENCES_FILENAME
        raw_meta = raw_dir / METADATA_FILENAME
        if not raw_seq.exists() or not raw_meta.exists():
            raise FileNotFoundError(
                f"Synthetic adapter requires {SEQUENCES_FILENAME} and "
                f"{METADATA_FILENAME} in {raw_dir}. Run the ma-irt static "
                "generator first."
            )

        with raw_seq.open("r", encoding="utf-8") as fh:
            raw_records: List[Dict[str, Any]] = json.load(fh)
        with raw_meta.open("r", encoding="utf-8") as fh:
            raw_meta_dict: Dict[str, Any] = json.load(fh)

        n_questions = int(raw_meta_dict["n_questions"])
        n_categories = int(raw_meta_dict["n_categories"])

        # Per-student filter then chunk. Chunking is applied at the
        # original-student granularity so all chunks of one student share
        # the same split, which preserves user-level separation.
        questions = [list(map(int, rec["questions"])) for rec in raw_records]
        responses = [list(map(int, rec["responses"])) for rec in raw_records]

        # Drop short sequences before splitting so the fractions reflect
        # the usable corpus, not the raw one.
        keep = [
            i for i, q in enumerate(questions) if len(q) >= self.cfg.min_seq_len
        ]
        questions = [questions[i] for i in keep]
        responses = [responses[i] for i in keep]
        n_students = len(questions)
        if n_students == 0:
            raise ValueError(
                "Synthetic adapter found zero sequences passing min_seq_len."
            )

        # Per-user split BEFORE chunking, so all chunks of a student
        # inherit the same code.
        user_codes = make_split(
            n_students=n_students,
            test_frac=self.cfg.test_frac,
            valid_frac=self.cfg.valid_frac,
            split_seed=self.cfg.split_seed,
        )

        if self.cfg.chunk_long_sequences and self.cfg.max_seq_len > 0:
            chunked_q, chunked_r, parent = _chunk_sequences(
                questions, responses, self.cfg.max_seq_len
            )
            chunked_codes = np.asarray(
                [int(user_codes[p]) for p in parent], dtype=np.int8
            )
        else:
            chunked_q = questions
            chunked_r = responses
            chunked_codes = user_codes

        seq_lens = [len(q) for q in chunked_q]
        seq_len_range = [int(min(seq_lens)), int(max(seq_lens))]

        # Build the on-disk records in input order; the split goes into
        # each record so external readers do not need ``metadata.json``
        # to know the fold.
        out_records: List[Dict[str, Any]] = []
        for q, r, code in zip(chunked_q, chunked_r, chunked_codes):
            out_records.append({
                "questions": list(map(int, q)),
                "responses": list(map(int, r)),
                "split": _CODE_TO_SPLIT[int(code)],
            })

        n_train = int((chunked_codes == SPLIT_TRAIN).sum())
        n_valid = int((chunked_codes == SPLIT_VALID).sum())
        n_test = int((chunked_codes == SPLIT_TEST).sum())
        total = max(1, n_train + n_valid + n_test)

        meta: Dict[str, Any] = {
            "dataset_name": self.cfg.name,
            "adapter_class": type(self).__name__,
            "n_students": n_students,
            "n_questions": n_questions,
            "n_categories": n_categories,
            "n_kcs": 0,
            "seq_len_range": seq_len_range,
            "ordinal_coercion_method": "synthetic_gpcm",
            "splits": {
                "split_seed": int(self.cfg.split_seed),
                "train_frac": n_train / total,
                "valid_frac": n_valid / total,
                "test_frac": n_test / total,
                "n_train": n_train,
                "n_valid": n_valid,
                "n_test": n_test,
            },
            "question_id_map": {str(q): q for q in range(1, n_questions + 1)},
        }
        validate_metadata(meta)
        validate_sequences(out_records, n_questions=n_questions, n_categories=n_categories)

        out_dir = self.artefact_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_sequences(out_records, out_dir / SEQUENCES_FILENAME)
        dump_metadata(meta, out_dir / METADATA_FILENAME)
        # Empty coercion file kept for schema parity with the K=4 adapters.
        # Persisted via dump_metadata for stable key ordering.
        dump_metadata({"method": "synthetic_gpcm"},
                      out_dir / "coercion_artefacts.json")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        artefact = self.artefact_dir
        records = load_sequences(artefact / SEQUENCES_FILENAME)
        meta = load_metadata(artefact / METADATA_FILENAME)
        validate_metadata(meta)
        validate_sequences(records, n_questions=int(meta["n_questions"]),
                           n_categories=int(meta["n_categories"]))

        self._questions = [list(map(int, rec["questions"])) for rec in records]
        self._responses = [list(map(int, rec["responses"])) for rec in records]
        self._student_split = np.asarray(
            [_SPLIT_CODES[rec["split"]] for rec in records], dtype=np.int8,
        )
        self._metadata = meta
        self._q_matrix = None
