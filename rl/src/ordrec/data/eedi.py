"""``EediAdapter``, the empirical-difficulty K=4 distractor ordering.

Eedi (Wang et al. 2020 NeurIPS) ships four-option MC data with the
chosen distractor recorded. Treating the three distractors as
psychometrically distinct gives a K=4 ordinal label whose levels are
ordered by the mean ability of students who selected each distractor.
See ``docs/ordrec_impl_guide.md`` Section 2.3 for the spec.

The adapter is built to run on either the small fixture
(``tests/fixtures/eedi_mini.csv``, 50 rows) or the full Eedi release.
The two cases differ only in the size of the dataframe; the algorithm
is identical.

Expected raw columns
--------------------
The adapter accepts a single csv at ``cfg.raw_dir / "eedi.csv"``,
or a directory containing exactly one ``*.csv`` it can autodetect.
Required columns,

    student_id        int or str, mapped to a contiguous 0-based index
    question_id       int or str, mapped to a contiguous 1-based id
    correct           {0, 1}, ground-truth correctness flag
    selected_option   int in {1, 2, 3, 4}, raw chosen option id

An ``answer_value`` column is allowed as an alias for ``selected_option``
to mirror the official Eedi schema, and an ``is_correct`` column aliases
``correct``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .base import AdapterConfig, OrdinalDatasetBase, _SPLIT_CODES
from .placeholder_2pl import fit_placeholder_2pl
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


N_CATEGORIES_EEDI = 4  # K = 4: three ordered distractors + correct
N_OPTIONS_EEDI = 4     # four-way MC

# Ordinal codes
ORDINAL_LOW = 0
ORDINAL_MID = 1
ORDINAL_HIGH = 2
ORDINAL_CORRECT = 3

_CODE_TO_SPLIT = {SPLIT_TRAIN: "train", SPLIT_VALID: "valid", SPLIT_TEST: "test"}


class EediAdapter(OrdinalDatasetBase):
    """K=4 ordinal Eedi adapter with empirical-difficulty distractor ordering."""

    # --------------------------- Raw input ---------------------------------

    def _locate_csv(self) -> Path:
        raw = Path(self.cfg.raw_dir)
        if raw.is_file() and raw.suffix.lower() == ".csv":
            return raw
        if not raw.exists():
            raise FileNotFoundError(f"Eedi raw_dir does not exist: {raw}")
        csvs = sorted(raw.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"Eedi adapter expected a *.csv under {raw}, found none."
            )
        if len(csvs) > 1:
            preferred = raw / "eedi.csv"
            if preferred.exists():
                return preferred
            raise ValueError(
                f"Multiple csvs under {raw}, set cfg.raw_dir to a specific file."
            )
        return csvs[0]

    def _read_csv(self) -> pd.DataFrame:
        path = self._locate_csv()
        df = pd.read_csv(path)
        # Normalise column aliases.
        renames: Dict[str, str] = {}
        if "is_correct" in df.columns and "correct" not in df.columns:
            renames["is_correct"] = "correct"
        if "answer_value" in df.columns and "selected_option" not in df.columns:
            renames["answer_value"] = "selected_option"
        if renames:
            df = df.rename(columns=renames)

        required = ("student_id", "question_id", "correct", "selected_option")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Eedi csv {path} missing required columns {missing}, "
                f"available {list(df.columns)}"
            )

        # Stable order so re-materialisation produces byte-identical
        # sequences.json. Sort by (student_id, question_id) but use the
        # natural per-csv ordering when a "row_id" or timestamp column is
        # present.
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

        # Map student_id -> compact 0-based index, in first-appearance order.
        student_ids = pd.unique(df["student_id"])
        student_id_map: Dict[Any, int] = {
            sid: i for i, sid in enumerate(student_ids.tolist())
        }
        n_students = len(student_id_map)

        # Map question_id -> compact 1-based id (0 reserved for padding).
        question_ids_unique = pd.unique(df["question_id"])
        question_id_map: Dict[Any, int] = {
            qid: i + 1 for i, qid in enumerate(question_ids_unique.tolist())
        }
        n_items = len(question_id_map)

        df["_student"] = df["student_id"].map(student_id_map).astype(np.int64)
        df["_item"] = df["question_id"].map(question_id_map).astype(np.int64)
        df["_correct"] = df["correct"].astype(int).clip(0, 1)
        df["_option"] = df["selected_option"].astype(int)

        # Per-student question and correctness lists. Order within a
        # student follows the dataframe's stable order.
        questions: List[List[int]] = [[] for _ in range(n_students)]
        options: List[List[int]] = [[] for _ in range(n_students)]
        correctness: List[List[int]] = [[] for _ in range(n_students)]
        for stu, item, opt, cor in zip(
            df["_student"].to_numpy(),
            df["_item"].to_numpy(),
            df["_option"].to_numpy(),
            df["_correct"].to_numpy(),
        ):
            questions[int(stu)].append(int(item))
            options[int(stu)].append(int(opt))
            correctness[int(stu)].append(int(cor))

        # Per-user split BEFORE any chunking.
        user_codes = make_split(
            n_students=n_students,
            test_frac=self.cfg.test_frac,
            valid_frac=self.cfg.valid_frac,
            split_seed=self.cfg.split_seed,
        )
        train_indices = np.where(user_codes == SPLIT_TRAIN)[0]

        # Step 1: fit placeholder 2PL on train fold using binary correctness.
        # 20 epochs gives stable theta_hat ordering on the small fixture
        # while staying well under the ~30s budget for full Eedi.
        fit = fit_placeholder_2pl(
            questions=questions,
            binary_responses=correctness,
            train_indices=train_indices,
            n_items=n_items,
            max_iters=20,
            lr=5e-2,
            batch_size=128,
            device="cpu",
            seed=int(self.cfg.split_seed) ^ 0xE3D1,
        )

        # Step 2 + 3: for every question, compute mean theta_hat per
        # distractor on the train fold and rank ascending. Correct answer
        # per question is the option chosen when ``correct == 1``; we
        # take the most common such value as the canonical "correct"
        # option for robustness against a few mislabelled rows.
        correct_option_per_q = _infer_correct_option_per_question(
            df, n_items=n_items, train_students=set(train_indices.tolist())
        )

        sigma_per_q, distractor_means, fallback_qs = _rank_distractors_by_difficulty(
            df=df,
            theta_hat=fit.theta_hat,
            n_items=n_items,
            train_students=set(train_indices.tolist()),
            correct_option_per_q=correct_option_per_q,
        )

        # Step 4: recode every response (train and test) using sigma_q.
        recoded_responses: List[List[int]] = [[] for _ in range(n_students)]
        unseen_distractor_count = 0
        for stu_i in range(n_students):
            for item_i, opt_i in zip(questions[stu_i], options[stu_i]):
                correct_opt = correct_option_per_q[item_i]
                if opt_i == correct_opt:
                    recoded_responses[stu_i].append(ORDINAL_CORRECT)
                    continue
                sigma = sigma_per_q[item_i]
                if opt_i in sigma:
                    recoded_responses[stu_i].append(sigma.index(opt_i))
                else:
                    # Unseen distractor: log and fall back to the
                    # middle wrong category as specified.
                    recoded_responses[stu_i].append(ORDINAL_MID)
                    unseen_distractor_count += 1

        # Build the on-disk records.
        out_records: List[Dict[str, Any]] = []
        for stu_i in range(n_students):
            q_seq = questions[stu_i]
            if len(q_seq) < self.cfg.min_seq_len:
                continue
            out_records.append({
                "questions": q_seq,
                "responses": recoded_responses[stu_i],
                "split": _CODE_TO_SPLIT[int(user_codes[stu_i])],
            })

        if not out_records:
            raise ValueError(
                "EediAdapter produced zero sequences after min_seq_len filter."
            )

        # Split-fraction stats are recomputed against the surviving
        # records so they match what users actually consume.
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
            "n_items": n_items,
            "n_categories": N_CATEGORIES_EEDI,
            "n_kcs": 0,
            "seq_len_range": seq_len_range,
            "ordinal_coercion_method": "distractor_difficulty_2pl",
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
            "method": "distractor_difficulty_2pl",
            "distractor_order_per_q": {
                str(q): [int(o) for o in sigma_per_q[q]] for q in range(1, n_items + 1)
            },
            "correct_option_per_q": {
                str(q): int(correct_option_per_q[q]) for q in range(1, n_items + 1)
            },
            "distractor_mean_theta_per_q": {
                str(q): {str(k): float(v) for k, v in distractor_means[q].items()}
                for q in range(1, n_items + 1)
            },
            "fallback_questions": fallback_qs,
            "unseen_distractor_count_total": int(unseen_distractor_count),
            "placeholder_2pl": {
                "max_iters": int(fit.n_iters),
                "lr": 1e-2,
                "n_train_students": int(len(train_indices)),
            },
        }

        validate_metadata(meta)
        validate_sequences(out_records, n_items=n_items, n_categories=N_CATEGORIES_EEDI)

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
        validate_sequences(records, n_items=int(meta["n_items"]),
                           n_categories=int(meta["n_categories"]))

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_correct_option_per_question(
    df: pd.DataFrame, n_items: int, train_students: set
) -> Dict[int, int]:
    """Return ``{item_id_1based: correct_option_int}`` from the train fold.

    The "correct" option is taken to be the most common ``selected_option``
    among rows with ``correct == 1`` on the train fold. Falls back to
    the global mode when the train fold has no positive observations
    for an item.
    """
    out: Dict[int, int] = {}
    train_mask = df["_student"].isin(train_students)
    for item, sub in df.groupby("_item"):
        sub_train = sub[train_mask.loc[sub.index]]
        positives = sub_train[sub_train["_correct"] == 1]
        if not positives.empty:
            out[int(item)] = int(positives["_option"].mode().iat[0])
            continue
        # Fall back to the full df.
        positives_full = sub[sub["_correct"] == 1]
        if not positives_full.empty:
            out[int(item)] = int(positives_full["_option"].mode().iat[0])
            continue
        # Last resort: pick option 4 (matches Eedi convention "D == correct"
        # in the public release where labelled rows are absent).
        out[int(item)] = N_OPTIONS_EEDI
    # Fill any item missing from groupby (no rows at all should not happen).
    for q in range(1, n_items + 1):
        out.setdefault(q, N_OPTIONS_EEDI)
    return out


def _rank_distractors_by_difficulty(
    df: pd.DataFrame,
    theta_hat: np.ndarray,
    n_items: int,
    train_students: set,
    correct_option_per_q: Dict[int, int],
) -> tuple[Dict[int, List[int]], Dict[int, Dict[int, float]], List[Dict[str, Any]]]:
    """Compute sigma_q ordering for every question.

    Returns:
        ``(sigma_per_q, distractor_means, fallback_questions)``.
        ``sigma_per_q[q]`` is a length-3 list of option ids ranked
        ascending by mean theta_hat on the train fold.
        ``distractor_means[q][option]`` is the mean theta_hat used.
        ``fallback_questions`` lists the items that needed the
        lexicographic fallback when an option was unseen on train.
    """
    sigma_per_q: Dict[int, List[int]] = {}
    distractor_means: Dict[int, Dict[int, float]] = {}
    fallback: List[Dict[str, Any]] = []

    train_mask = df["_student"].isin(train_students)
    df_train = df[train_mask]

    for q in range(1, n_items + 1):
        correct_opt = correct_option_per_q[q]
        all_options = list(range(1, N_OPTIONS_EEDI + 1))
        candidate_distractors = [o for o in all_options if o != correct_opt]

        sub = df_train[df_train["_item"] == q]
        means_q: Dict[int, float] = {}
        for opt in candidate_distractors:
            mask = (sub["_option"] == opt)
            if not mask.any():
                continue
            stus = sub.loc[mask, "_student"].to_numpy()
            thetas = theta_hat[stus]
            thetas = thetas[~np.isnan(thetas)]
            if thetas.size == 0:
                continue
            means_q[opt] = float(np.mean(thetas))

        distractor_means[q] = means_q

        if len(means_q) == len(candidate_distractors):
            # All three distractors observed on train; rank ascending.
            ranked = sorted(candidate_distractors, key=lambda o: means_q[o])
            sigma_per_q[q] = ranked
        else:
            # Lexicographic fallback for the unobserved distractors so
            # sigma_q is always length 3. Observed ones still rank by
            # their mean theta; unobserved tail keeps ascending option id.
            observed = sorted(means_q.keys(), key=lambda o: means_q[o])
            unobserved = sorted(o for o in candidate_distractors if o not in means_q)
            sigma_per_q[q] = observed + unobserved
            fallback.append({
                "question_id_1based": int(q),
                "n_observed_distractors_train": int(len(observed)),
                "unobserved_options": [int(o) for o in unobserved],
            })

    return sigma_per_q, distractor_means, fallback
