"""``SlamAdapter`` -- SLAM 2018 en_es track, K=3 ordinal per exercise.

SLAM 2018 (Settles et al., BEA at NAACL 2018) is a second-language
acquisition dataset released by Duolingo under CC0.  The en_es
(English speakers learning Spanish) track contains per-token binary
mistake labels grouped into exercise attempts.  We aggregate token
labels to a per-exercise ordinal category on the K=3 scale::

    0  all tokens wrong   (mistake fraction = 1.0)
    1  partial           (0.0 < mistake fraction < 1.0)
    2  all tokens correct (mistake fraction = 0.0)

This is the natural GPCM ordering from worst to best performance.

Item identity: SLAM has no canonical cross-user exercise ID.  We derive
a stable signature by hashing (format, lowercased word sequence).  The
hash is a 16-hex MD5 digest of ``format + "|" + space-joined lowercased
words``.

Long-tail handling: Exercises seen by fewer than ``min_count`` distinct
users are collapsed into per-format catch-all buckets (one bucket per
format, three in total for SLAM).  ``min_count=10`` leaves 5 072 named
items from 14 458 unique signatures in the en_es train fold, covering
~74% of train exercises.  The remaining ~26% map to catch-all buckets.

Split choice: We respect the official SLAM temporal split so that
results are comparable with SLAM-era per-token AUC baselines.
``AdapterConfig.split_seed``, ``test_frac``, and ``valid_frac`` are
ignored for splitting; AdapterConfig is still required because it carries
``out_dir``, ``name``, ``min_seq_len``, ``max_seq_len``, and
``chunk_long_sequences``.

Category-balance: ~62% all-correct (cat 2), ~36% partial (cat 1), ~2%
all-wrong (cat 0) in the en_es train fold.  Category 0 is rare because
most exercises have at least one correctly recalled token.  K=3
thresholds are strict (0.0 and 1.0 mistake fractions).  Whether to shift
cat-0 to >0.5 mistake fraction is a D2 open question.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

N_CATEGORIES_SLAM = 3

# Ordinal codes (ascending mastery signal).
ORD_ALL_WRONG = 0     # mistake fraction == 1.0
ORD_PARTIAL = 1       # 0.0 < mistake fraction < 1.0
ORD_ALL_CORRECT = 2   # mistake fraction == 0.0

# Per-format catch-all bucket names for the long tail.
_FORMATS = ("reverse_translate", "reverse_tap", "listen")

# Default minimum distinct-user count for a named item ID.
_MIN_COUNT_DEFAULT = 10

_HEADER_RE = re.compile(r"^# user:(?P<user>\S+)\s+.*?format:(?P<format>\w+)")


def _exercise_hash(format_: str, words: List[str]) -> str:
    """Return a 16-hex-char MD5 hash for (format, lowercased word sequence)."""
    canonical = format_ + "|" + " ".join(w.lower() for w in words)
    return hashlib.md5(canonical.encode()).hexdigest()[:16]


def _parse_slam_file(
    path: Path,
    has_labels: bool,
    label_lookup: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Parse a SLAM-format file into a list of exercise dicts.

    Args:
        path: Path to a SLAM train/dev/test file.
        has_labels: If ``True``, the last column of every token row is
            the label (0 or 1).  If ``False``, ``label_lookup`` is
            used to fill in labels from a separate ``.key`` file.
        label_lookup: ``{token_id: label}`` mapping used when
            ``has_labels=False``.

    Returns:
        List of dicts with keys: ``user`` (str), ``format`` (str),
        ``days`` (float), ``words`` (list[str], lowercased),
        ``labels`` (list[int], 0=correct 1=mistake).
    """
    if label_lookup is None:
        label_lookup = {}

    exercises: List[Dict[str, Any]] = []
    current_user: Optional[str] = None
    current_format: Optional[str] = None
    current_days: float = 0.0
    current_words: List[str] = []
    current_labels: List[int] = []

    def flush() -> None:
        if current_words and current_user is not None:
            exercises.append({
                "user": current_user,
                "format": current_format,
                "days": current_days,
                "words": list(current_words),
                "labels": list(current_labels),
            })
        current_words.clear()
        current_labels.clear()

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("# user:"):
                flush()
                m = _HEADER_RE.match(line)
                if m:
                    current_user = m.group("user")
                    current_format = m.group("format")
                    dm = re.search(r"days:([0-9.]+)", line)
                    current_days = float(dm.group(1)) if dm else 0.0
            elif line.startswith("#"):
                flush()
            elif line.strip() == "":
                flush()
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                token_id = parts[0]
                word = parts[1]
                if has_labels:
                    try:
                        label = int(parts[-1])
                    except (ValueError, IndexError):
                        label = 0
                else:
                    label = label_lookup.get(token_id, 0)
                current_words.append(word.lower())
                current_labels.append(label)

    flush()
    return exercises


def _load_key_file(path: Path) -> Dict[str, int]:
    """Load a ``.key`` file into a ``{token_id: label}`` dict."""
    lookup: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                lookup[parts[0]] = int(parts[1])
    return lookup


def _mistake_fraction_to_ordinal(labels: List[int]) -> int:
    """Convert a token-level mistake list to a K=3 ordinal code."""
    if not labels:
        return ORD_ALL_CORRECT
    frac = sum(labels) / len(labels)
    if frac == 0.0:
        return ORD_ALL_CORRECT
    elif frac == 1.0:
        return ORD_ALL_WRONG
    else:
        return ORD_PARTIAL


def _locate_slam_files(
    raw_dir: Path,
) -> Tuple[Path, Path, Path, Path, Path]:
    """Locate the five required SLAM files under ``raw_dir``.

    Returns:
        Tuple of (train, dev, dev_key, test, test_key) Path objects.

    Raises:
        FileNotFoundError: If the directory or any required file is missing.
        ValueError: If multiple train files are present.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"SlamAdapter raw_dir does not exist: {raw_dir}"
        )
    train_files = [p for p in raw_dir.glob("*.slam.*.train") if not p.name.startswith(".")]
    if not train_files:
        raise FileNotFoundError(
            "SlamAdapter: no '*.slam.*.train' file found under "
            f"{raw_dir}"
        )
    if len(train_files) > 1:
        raise ValueError(
            f"SlamAdapter: multiple train files under {raw_dir}. "
            "Set cfg.raw_dir to the directory containing one track."
        )
    train = train_files[0]
    base = train.name.replace(".train", "")
    dev = raw_dir / f"{base}.dev"
    dev_key = raw_dir / f"{base}.dev.key"
    test = raw_dir / f"{base}.test"
    test_key = raw_dir / f"{base}.test.key"
    for p, name in [
        (dev, "dev"), (dev_key, "dev.key"),
        (test, "test"), (test_key, "test.key"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"SlamAdapter: expected {name} file at {p}"
            )
    return train, dev, dev_key, test, test_key


class SlamAdapter(OrdinalDatasetBase):
    """K=3 SLAM 2018 adapter using mistake-fraction per exercise.

    Respects the official SLAM train/dev/test temporal split.
    Coercion artefacts are computed on the train fold only and
    persisted to ``coercion_artefacts.json``.

    Args:
        cfg: AdapterConfig.  ``out_dir``, ``name``, ``min_seq_len``,
            ``max_seq_len``, and ``chunk_long_sequences`` are used.
            ``split_seed``, ``test_frac``, and ``valid_frac`` are
            ignored because SLAM ships its own split.
        min_count: Minimum number of distinct users who must have seen
            an exercise for it to receive a named item ID. Defaults
            to 10.
    """

    def __init__(
        self,
        cfg: AdapterConfig,
        min_count: int = _MIN_COUNT_DEFAULT,
    ) -> None:
        super().__init__(cfg)
        self.min_count = min_count

    def materialise(self) -> None:  # noqa: C901
        """Read SLAM raw files from ``cfg.raw_dir`` and write the artefact."""
        train_path, dev_path, dev_key_path, test_path, test_key_path = (
            _locate_slam_files(Path(self.cfg.raw_dir))
        )

        train_exs = _parse_slam_file(train_path, has_labels=True)
        dev_key_data = _load_key_file(dev_key_path)
        dev_exs = _parse_slam_file(
            dev_path, has_labels=False, label_lookup=dev_key_data
        )
        test_key_data = _load_key_file(test_key_path)
        test_exs = _parse_slam_file(
            test_path, has_labels=False, label_lookup=test_key_data
        )

        # Build item vocabulary from the train fold only.
        sig_user_sets: Dict[str, set] = defaultdict(set)
        for ex in train_exs:
            sig_user_sets[
                _exercise_hash(ex["format"], ex["words"])
            ].add(ex["user"])

        sig_user_counts: Counter = Counter(
            {h: len(u) for h, u in sig_user_sets.items()}
        )
        named_sigs = sorted(
            h for h, cnt in sig_user_counts.items() if cnt >= self.min_count
        )
        n_named = len(named_sigs)
        sig_to_id: Dict[str, int] = {
            sig: i + 1 for i, sig in enumerate(named_sigs)
        }
        catchall_ids: Dict[str, int] = {
            fmt: n_named + i + 1 for i, fmt in enumerate(_FORMATS)
        }
        n_questions = n_named + len(_FORMATS)
        _default_fmt = _FORMATS[0]

        def get_item_id(sig: str, fmt: str) -> int:
            if sig in sig_to_id:
                return sig_to_id[sig]
            actual = fmt if fmt in catchall_ids else _default_fmt
            return catchall_ids[actual]

        total_train_ex = len(train_exs)
        named_count_train = sum(
            1 for ex in train_exs
            if _exercise_hash(ex["format"], ex["words"]) in sig_to_id
        )

        out_records: List[Dict[str, Any]] = []

        def emit_split(exs: List[Dict[str, Any]], split_tag: str) -> None:
            student_data: Dict[str, Dict[str, Any]] = {}
            for ex in exs:
                uid = ex["user"]
                item_id = get_item_id(
                    _exercise_hash(ex["format"], ex["words"]), ex["format"]
                )
                ordinal = _mistake_fraction_to_ordinal(ex["labels"])
                if uid not in student_data:
                    student_data[uid] = {
                        "questions": [],
                        "responses": [],
                        "split": split_tag,
                    }
                student_data[uid]["questions"].append(item_id)
                student_data[uid]["responses"].append(ordinal)
            for uid, data in student_data.items():
                qs, rs = data["questions"], data["responses"]
                seq_len = len(qs)
                if seq_len < self.cfg.min_seq_len:
                    continue
                if (
                    self.cfg.max_seq_len > 0
                    and seq_len > self.cfg.max_seq_len
                ):
                    if self.cfg.chunk_long_sequences:
                        for start in range(0, seq_len, self.cfg.max_seq_len):
                            cq = qs[start: start + self.cfg.max_seq_len]
                            cr = rs[start: start + self.cfg.max_seq_len]
                            if len(cq) >= self.cfg.min_seq_len:
                                out_records.append({
                                    "questions": cq,
                                    "responses": cr,
                                    "split": split_tag,
                                })
                    else:
                        out_records.append({
                            "questions": qs[: self.cfg.max_seq_len],
                            "responses": rs[: self.cfg.max_seq_len],
                            "split": split_tag,
                        })
                else:
                    out_records.append({
                        "questions": qs,
                        "responses": rs,
                        "split": split_tag,
                    })

        emit_split(train_exs, "train")
        emit_split(dev_exs, "valid")
        emit_split(test_exs, "test")

        if not out_records:
            raise ValueError(
                "SlamAdapter produced zero sequences after min_seq_len filter."
            )

        # K=3 category distribution (train fold only).
        cat_counts = [0, 0, 0]
        for ex in train_exs:
            cat_counts[_mistake_fraction_to_ordinal(ex["labels"])] += 1
        total_train_ex_count = sum(cat_counts)

        split_counts: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
        for rec in out_records:
            split_counts[rec["split"]] += 1
        total_kept = sum(split_counts.values())
        seq_lens = [len(r["questions"]) for r in out_records]
        all_student_ids: set = {
            ex["user"] for ex in train_exs + dev_exs + test_exs
        }

        meta: Dict[str, Any] = {
            "dataset_name": self.cfg.name,
            "adapter_class": type(self).__name__,
            "n_students": len(all_student_ids),
            "n_questions": n_questions,
            "n_categories": N_CATEGORIES_SLAM,
            "n_kcs": 0,
            "seq_len_range": [int(min(seq_lens)), int(max(seq_lens))],
            "ordinal_coercion_method": "binary",
            "splits": {
                "split_seed": int(self.cfg.split_seed),
                "train_frac": split_counts["train"] / max(1, total_kept),
                "valid_frac": split_counts["valid"] / max(1, total_kept),
                "test_frac": split_counts["test"] / max(1, total_kept),
                "n_train": split_counts["train"],
                "n_valid": split_counts["valid"],
                "n_test": split_counts["test"],
            },
            "question_id_map": {
                sig: int(id_) for sig, id_ in sig_to_id.items()
            },
            "coercion_artefacts_path": COERCION_FILENAME,
        }

        coercion_payload: Dict[str, Any] = {
            "method": "mistake_fraction_k3",
            "ordinal_map": {
                "all_correct": ORD_ALL_CORRECT,
                "partial": ORD_PARTIAL,
                "all_wrong": ORD_ALL_WRONG,
            },
            "item_selection": {
                "min_count": int(self.min_count),
                "n_unique_signatures_train": len(sig_user_counts),
                "n_named_items": int(n_named),
                "n_catchall_buckets": int(len(_FORMATS)),
                "n_questions_total": int(n_questions),
                "catchall_item_ids": {
                    fmt: int(id_) for fmt, id_ in catchall_ids.items()
                },
                "coverage_train_frac": float(
                    named_count_train / max(1, total_train_ex)
                ),
                "note": (
                    f"Exercises seen by >= {self.min_count} distinct users "
                    "receive a named item ID. The remaining "
                    f"{len(sig_user_counts) - n_named} signatures are "
                    "remapped to per-format catch-all buckets "
                    f"(IDs {n_named + 1} to {n_questions})."
                ),
            },
            "split_policy": {
                "policy": "official_temporal",
                "note": (
                    "The official SLAM train/dev/test split is respected "
                    "verbatim. AdapterConfig split_seed, test_frac, and "
                    "valid_frac are not used for splitting."
                ),
            },
            "category_distribution_train": {
                "all_wrong_cat0": int(cat_counts[0]),
                "partial_cat1": int(cat_counts[1]),
                "all_correct_cat2": int(cat_counts[2]),
                "total_exercises": int(total_train_ex_count),
                "frac_all_wrong": float(
                    cat_counts[0] / max(1, total_train_ex_count)
                ),
                "frac_partial": float(
                    cat_counts[1] / max(1, total_train_ex_count)
                ),
                "frac_all_correct": float(
                    cat_counts[2] / max(1, total_train_ex_count)
                ),
                "note": (
                    "Category 0 (all-wrong) is rare (~2%) because most "
                    "exercises have at least one correctly recalled token. "
                    "K=3 thresholds are strict (0.0 and 1.0 mistake "
                    "fractions). Whether to shift cat-0 to >0.5 is a D2 "
                    "open question."
                ),
            },
        }

        validate_metadata(meta)
        validate_sequences(
            out_records,
            n_questions=n_questions,
            n_categories=N_CATEGORIES_SLAM,
        )

        out_dir = self.artefact_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_sequences(out_records, out_dir / SEQUENCES_FILENAME)
        dump_metadata(meta, out_dir / METADATA_FILENAME)
        dump_coercion_artefacts(coercion_payload, out_dir / COERCION_FILENAME)

    def load(self) -> None:
        """Load a previously materialised artefact into memory."""
        artefact = self.artefact_dir
        records = load_sequences(artefact / SEQUENCES_FILENAME)
        meta = load_metadata(artefact / METADATA_FILENAME)
        validate_metadata(meta)
        validate_sequences(
            records,
            n_questions=int(meta["n_questions"]),
            n_categories=int(meta["n_categories"]),
        )
        self._questions = [
            list(map(int, rec["questions"])) for rec in records
        ]
        self._responses = [
            list(map(int, rec["responses"])) for rec in records
        ]
        self._student_split = np.asarray(
            [_SPLIT_CODES[rec["split"]] for rec in records], dtype=np.int8
        )
        self._metadata = meta
        self._q_matrix = None
        coercion_path = artefact / COERCION_FILENAME
        self._coercion = (
            load_coercion_artefacts(coercion_path)
            if coercion_path.exists()
            else {}
        )


__all__ = [
    "SlamAdapter",
    "N_CATEGORIES_SLAM",
    "ORD_ALL_WRONG",
    "ORD_PARTIAL",
    "ORD_ALL_CORRECT",
]
