"""On-disk schema definitions and validators for materialised adapters.

Every adapter writes the same four-file layout under
``out_dir / name``,

    sequences.json          list[ {"questions": [int...], "responses": [int...],
                                   "split": "train"|"valid"|"test"} ]
    metadata.json           the keys listed in ``COMMON_RECORD_SCHEMA``
    coercion_artefacts.json adapter-specific train-only stats
    q_matrix.npz            optional, present iff metadata["n_kcs"] > 0

This module owns the schema constants and the JSON validators. Loading
and writing happens in the concrete adapters so they retain control
over their on-disk byte layout (the determinism contract is satisfied
by stable key order in ``json.dump(..., sort_keys=True)`` calls there).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SEQUENCES_FILENAME = "sequences.json"
METADATA_FILENAME = "metadata.json"
COERCION_FILENAME = "coercion_artefacts.json"
Q_MATRIX_FILENAME = "q_matrix.npz"

# Required keys in each record of ``sequences.json``.
SEQUENCE_RECORD_KEYS = ("questions", "responses", "split")

VALID_SPLITS = ("train", "valid", "test")

# Required keys in ``metadata.json``. Optional keys are flagged in
# ``OPTIONAL_METADATA_KEYS`` and not required by the validator.
COMMON_RECORD_SCHEMA: Dict[str, type] = {
    "dataset_name": str,
    "adapter_class": str,
    "n_students": int,
    "n_items": int,
    "n_categories": int,
    "n_kcs": int,
    "seq_len_range": list,
    "ordinal_coercion_method": str,
    "splits": dict,
    "question_id_map": dict,
}

OPTIONAL_METADATA_KEYS = (
    "coercion_artefacts_path",
    "fallback_questions",
    "build_timestamp",
)

# Allowed values of ``metadata["ordinal_coercion_method"]``.
COERCION_METHODS = (
    "binary",
    "distractor_difficulty_2pl",
    "correctness_x_time_quadrant",
    "synthetic_gpcm",
)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a materialised artefact violates the schema."""


def validate_metadata(meta: Mapping[str, Any]) -> None:
    """Validate a ``metadata.json`` mapping.

    Raises:
        SchemaError: When a required key is missing or has the wrong type.
    """
    for key, expected_type in COMMON_RECORD_SCHEMA.items():
        if key not in meta:
            raise SchemaError(f"metadata is missing required key '{key}'")
        if not isinstance(meta[key], expected_type):
            raise SchemaError(
                f"metadata['{key}'] must be of type {expected_type.__name__}, "
                f"got {type(meta[key]).__name__}"
            )
    if meta["n_categories"] < 2:
        raise SchemaError(
            f"metadata['n_categories'] must be >= 2, got {meta['n_categories']}"
        )
    if meta["n_items"] < 1:
        raise SchemaError(
            f"metadata['n_items'] must be >= 1, got {meta['n_items']}"
        )
    if meta["ordinal_coercion_method"] not in COERCION_METHODS:
        raise SchemaError(
            "metadata['ordinal_coercion_method'] must be one of "
            f"{COERCION_METHODS}, got {meta['ordinal_coercion_method']!r}"
        )
    splits = meta["splits"]
    for k in ("split_seed", "train_frac", "valid_frac", "test_frac",
              "n_train", "n_valid", "n_test"):
        if k not in splits:
            raise SchemaError(f"metadata['splits'] is missing '{k}'")
    # The three fractions must sum to roughly 1 (allow numerical slop).
    total = float(splits["train_frac"] + splits["valid_frac"] + splits["test_frac"])
    if not 0.99 <= total <= 1.01:
        raise SchemaError(
            f"metadata['splits'] train/valid/test fractions must sum to 1, "
            f"got {total:.4f}"
        )


def validate_sequences(records: Iterable[Mapping[str, Any]],
                       n_items: int, n_categories: int) -> None:
    """Validate the list of sequence records.

    Args:
        records: Iterable of dicts matching ``SEQUENCE_RECORD_KEYS``.
        n_items: Item bank size; question ids must lie in ``[1, n_items]``.
        n_categories: K; responses must lie in ``[0, K - 1]``.

    Raises:
        SchemaError: On any malformed record.
    """
    for i, rec in enumerate(records):
        for k in SEQUENCE_RECORD_KEYS:
            if k not in rec:
                raise SchemaError(
                    f"sequence record {i} is missing required key '{k}'"
                )
        if rec["split"] not in VALID_SPLITS:
            raise SchemaError(
                f"sequence record {i} has split={rec['split']!r}, expected "
                f"one of {VALID_SPLITS}"
            )
        q = rec["questions"]
        r = rec["responses"]
        if not isinstance(q, list) or not isinstance(r, list):
            raise SchemaError(
                f"sequence record {i}: 'questions' and 'responses' must be lists"
            )
        if len(q) != len(r):
            raise SchemaError(
                f"sequence record {i}: questions and responses length mismatch "
                f"({len(q)} vs {len(r)})"
            )
        if len(q) == 0:
            raise SchemaError(f"sequence record {i} is empty")
        # Range checks. We avoid numpy here because records may be plain
        # Python lists at this stage and validation must work pre-load.
        for qi in q:
            if not (1 <= int(qi) <= n_items):
                raise SchemaError(
                    f"sequence record {i}: question id {qi} out of range "
                    f"[1, {n_items}]"
                )
        for ri in r:
            if not (0 <= int(ri) <= n_categories - 1):
                raise SchemaError(
                    f"sequence record {i}: response {ri} out of range "
                    f"[0, {n_categories - 1}]"
                )


def validate_q_matrix(q_matrix: np.ndarray, n_items: int, n_kcs: int) -> None:
    """Validate a Q-matrix ``np.ndarray``."""
    if q_matrix.dtype != np.uint8:
        raise SchemaError(
            f"q_matrix dtype must be uint8, got {q_matrix.dtype}"
        )
    if q_matrix.shape != (n_items, n_kcs):
        raise SchemaError(
            f"q_matrix shape must be ({n_items}, {n_kcs}), got {q_matrix.shape}"
        )
    if not np.all((q_matrix == 0) | (q_matrix == 1)):
        raise SchemaError("q_matrix must be binary (0/1)")


# ---------------------------------------------------------------------------
# Round-trip helpers (kept simple so byte-identical rebuild is easy)
# ---------------------------------------------------------------------------


def dump_sequences(records: List[Mapping[str, Any]], path: Path) -> None:
    """Write ``records`` to ``path`` with stable key ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``sort_keys=True`` plus deterministic record order in callers gives
    # byte-identical output across rebuilds; the trailing newline mirrors
    # what most editors enforce.
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(list(records), fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")


def load_sequences(path: Path) -> List[Dict[str, Any]]:
    """Read records back as plain Python objects."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_metadata(meta: Mapping[str, Any], path: Path) -> None:
    """Write a metadata mapping with stable key ordering, two-space indent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(meta), fh, sort_keys=True, indent=2)
        fh.write("\n")


def load_metadata(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_coercion_artefacts(payload: Mapping[str, Any], path: Path) -> None:
    """Write the adapter-specific coercion artefacts file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(payload), fh, sort_keys=True, indent=2)
        fh.write("\n")


def load_coercion_artefacts(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
