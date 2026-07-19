"""A4 real-bed bridge: KDD Cup 2010 Algebra 2008-2009 chunked loader
(`_planning/design/a4_design.md` v1.1, section 6's KDD row; vendor
attachment point #2, section 2.5). Produces the per-learner event streams
`kt_mirt.growth.bank`/`kt_mirt.growth.slices` consume: one
`kt_mirt.growth.synth.LearnerLog` per student (ordered by true
chronological arrival, learner id = the 0-based position in the returned
list, matching `synth.py`'s own convention so `bank.split_learners`'s
index arrays, and every existing consumer that tests ``log.learner in
some_index_set``, work unmodified on real-bed data), plus the static item
bank (`kt_mirt.growth.bank.ItemHierarchy` via `build_kdd_hierarchy`, and a
`kt_mirt.growth.qmatrix.QMatrix`) the bank calibrator and per-KC readouts
need.

Parsing reuses `scripts/triage/triage_kdd.py`'s proven logic exactly
(same file, same columns, same Correct-First-Attempt validity rule, same
``~~``-split multi-KC convention, same length-mismatch handling): this
loader's real-data agreement table against
`_planning/triage/kdd_algebra_2008_2009_stats.json` lives in
`_planning/design/realbed_bridge_notes.md`.

Interpretation notes (recorded per the harness instructions):

1. **Row inclusion is EVERY valid-CFA interaction, not only
   KTracedSkills-tagged ones.** The design's own consumer contract
   (`bank.LearnerRecord`) has no "this row carries no KC" special case --
   an untagged row is simply one whose ``tag_mask`` is all-``False``,
   which every downstream consumer already treats as a no-op for slice
   membership and opportunity counting (`slices.build_slices`'s own
   ``valid = rows.kc_mask.reshape(-1)`` filter; `bank._BankModel.
   growth_term`'s ``vals * kc_mask`` zeroing). Item difficulty (`b_j`),
   however, is a property of the STEP, not of the KC tagging, so a step's
   response is informative for bank calibration whether or not that
   particular row carries a KTracedSkills tag. Excluding untagged rows
   entirely (as `triage_kdd.py` does for its own per-KC-only statistics)
   would silently starve the bank calibrator of real response data on
   every step DataShop's KTracedSkills tagging happens to skip (measured
   at ~50% of rows here, see the notes file) -- a real fidelity cost with
   no compensating benefit, since those rows already self-exclude from
   every per-KC readout via an all-``False`` mask. This loader therefore
   builds one interaction event per valid-CFA row regardless of KC-tag
   presence, and reports the KTracedSkills-only subset counts (matching
   `triage_kdd.py`'s own basis) separately in `LoadStats` for the
   agreement table.
2. **A length-mismatched KC/Opportunity pair is treated as untagged**,
   not dropped from the interaction stream (same rationale as note 1):
   the row is still a real practice event on a real step; only its
   KTracedSkills tag is unusable. `triage_kdd.py` drops such rows from
   its per-KC stats entirely, which this loader's `LoadStats.
   n_kc_opp_mismatch` counter still reports (0 on the real file, per the
   triage JSON) for exact comparability, but the row itself is retained
   as an untagged interaction here rather than removed outright.
3. **Item identity is the step key** (``Problem Hierarchy || Problem Name
   || Step Name``, identical to `triage_kdd.py`'s composite), never a KC
   name -- the ASSISTments circularity the design and `qmatrix.py` guard
   against structurally cannot arise here since the two vocabularies are
   built from disjoint raw columns, and `qmatrix.build_qmatrix` still
   runs the runtime guard as a second, independent check.
4. **A step's KC tag set is taken at its FIRST tagged occurrence in file
   order.** DataShop's per-step KC tagging is a static content property
   (the same step should carry the same tag set on every occurrence);
   taking the first tagged sighting (rather than `triage_kdd.py`'s
   per-chunk-then-dict-``update`` "last chunk wins, first-within-chunk"
   rule) is a simpler, equally-valid tie-break for a property that should
   not vary, and the acceptance check's pure-anchor/arity numbers
   (recomputed independently here) serve as the empirical check that this
   tie-break choice does not matter on the real file.
5. **Chronological order** uses ``Step Start Time`` (a fixed-format,
   lexicographically-sortable ISO-like string -- the same no-datetime-
   parse trick `triage_report.md`'s Junyi/Eedi sections use), tie-broken
   by the file's own ``Row`` counter. Sorting is done ONCE, globally,
   after the whole file is streamed (a stable three-pass argsort chain:
   student, then timestamp, then row -- see `_chronological_order`),
   rather than assumed from raw file layout, since DataShop exports are
   not guaranteed to group one student's rows contiguously.
6. **Vocabulary interning is two-tier** (`_intern_new_only`): a vectorized
   `pandas.Series.map` against the running vocabulary handles every
   ALREADY-SEEN key in one C-level pass; only genuinely new keys (bounded
   by the vocabulary's own final size -- 291,084 items / 515 KCs on the
   real file, not by the 8.9M row count) fall through to a small Python
   loop that assigns the next id and records any side data (an item's
   hierarchy components). This keeps the full-file pass inside the "a few
   minutes" CPU budget without a second dependency beyond `pandas`
   (already a `triage_kdd.py` dependency; NOT currently declared in
   `kt-mirt/pyproject.toml`'s runtime deps, since that file is outside
   this stage's permitted edit set -- flagged in the notes file as a
   follow-up, not silently worked around).
7. **`LearnerLog` gains no timestamp field.** The design lists
   "timestamps" among what this bridge must carry, but `synth.LearnerLog`
   is a frozen, already-tested 4-field contract shared by every existing
   consumer; widening it is out of proportion to this stage's "minimal
   additive wiring" mandate. Timestamps are instead carried on
   `KddLoadResult.learner_timestamps` (one array per learner, aligned
   with that learner's ``item_ids``), available for provenance or future
   forgetting-interval work without touching the shared dataclass.
8. **Every raw-column parse failure is counted, not silently absorbed.**
   ``Correct First Attempt`` already had a validity counter
   (``LoadStats.n_rows_invalid_cfa``); ``Row``, ``Anon Student Id``, and
   ``Step Start Time`` now get the same treatment. A row with no
   resolvable ``Anon Student Id`` cannot be attributed to any
   `LearnerLog` and is excluded from the interaction stream entirely
   (counted in ``n_rows_missing_student_id``) rather than folded into a
   spurious "NaN" learner (every missing student id is a distinct
   unhashable-by-equality ``NaN`` float, so the prior unguarded code path
   would have silently minted a new bogus learner per occurrence). An
   unparseable ``Row`` value does not drop the row (it is still a real
   response on a real step; ``Row`` is only ever a same-timestamp/
   -student tie-break, note 5) but is counted
   (``n_rows_bad_row_num``) and given a sentinel value that sorts
   deterministically LAST within its tie group, rather than an
   implementation-defined ``NaN``-to-``int64`` cast. A missing
   ``Step Start Time`` already fell back to ``""`` (sorts first); it now
   also gets a counter (``n_rows_missing_timestamp``). All three counters
   are 0 on the real file (per the acceptance table in
   `realbed_bridge_notes.md`); they exist so a dirtier bed or a corrupted
   file fails loudly instead of silently.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kt_mirt.growth import qmatrix as qmatrix_mod
from kt_mirt.growth.bank import ItemHierarchy, build_kdd_hierarchy
from kt_mirt.growth.synth import LearnerLog

KC_MODEL_DEFAULT = "KTracedSkills"
BASE_COLS = ("Row", "Anon Student Id", "Problem Hierarchy", "Problem Name", "Step Name", "Step Start Time",
             "Correct First Attempt")


def _usecols(kc_model: str) -> list:
    return list(BASE_COLS) + [f"KC({kc_model})", f"Opportunity({kc_model})"]


# ---------------------------------------------------------------------------
# Vocabulary interning (module docstring note 6)
# ---------------------------------------------------------------------------


def _intern_new_only(values: np.ndarray, vocab_to_id: dict, vocab_names: list) -> np.ndarray:
    """Vectorized map against the running vocabulary; only genuinely new
    values fall through to a Python loop (bounded by vocabulary size, not
    row count). Returns int64 ids, same shape as ``values``."""
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    s = pd.Series(values)
    mapped = s.map(vocab_to_id)
    is_new = mapped.isna().to_numpy()
    out = np.empty(len(values), dtype=np.int64)
    if (~is_new).any():
        out[~is_new] = mapped[~is_new].to_numpy(dtype=np.int64)
    if is_new.any():
        new_pos = np.where(is_new)[0]
        seen_this_batch: dict = {}
        for pos in new_pos:
            k = values[pos]
            idx = seen_this_batch.get(k)
            if idx is None:
                idx = len(vocab_names)
                vocab_to_id[k] = idx
                vocab_names.append(k)
                seen_this_batch[k] = idx
            out[pos] = idx
    return out


def _intern_items(
    item_keys: np.ndarray, hier: np.ndarray, prob: np.ndarray, step: np.ndarray,
    vocab_to_id: dict, vocab_names: list,
    hier_by_item: list, prob_by_item: list, step_by_item: list,
) -> np.ndarray:
    """Like `_intern_new_only`, but also records each NEW item's raw
    hierarchy components (needed by `build_kdd_hierarchy`) the first time
    it is seen (module docstring note 4)."""
    if item_keys.size == 0:
        return np.zeros(0, dtype=np.int64)
    s = pd.Series(item_keys)
    mapped = s.map(vocab_to_id)
    is_new = mapped.isna().to_numpy()
    out = np.empty(len(item_keys), dtype=np.int64)
    if (~is_new).any():
        out[~is_new] = mapped[~is_new].to_numpy(dtype=np.int64)
    if is_new.any():
        new_pos = np.where(is_new)[0]
        seen_this_batch: dict = {}
        for pos in new_pos:
            k = item_keys[pos]
            idx = seen_this_batch.get(k)
            if idx is None:
                idx = len(vocab_names)
                vocab_to_id[k] = idx
                vocab_names.append(k)
                hier_by_item.append(hier[pos])
                prob_by_item.append(prob[pos])
                step_by_item.append(step[pos])
                seen_this_batch[k] = idx
            out[pos] = idx
    return out


def _intern_kc_matrix(name_matrix: np.ndarray, vocab_to_id: dict, vocab_names: list) -> np.ndarray:
    """Interns a ragged, ``NaN``/``None``-padded object matrix of KC-name
    strings into an ``int64`` matrix, ``-1`` for missing/pad slots."""
    shape = name_matrix.shape
    flat = pd.Series(name_matrix.reshape(-1))
    non_null = flat.notna().to_numpy()
    out = np.full(flat.shape[0], -1, dtype=np.int64)
    if non_null.any():
        ids = _intern_new_only(flat[non_null].to_numpy(), vocab_to_id, vocab_names)
        out[non_null] = ids
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------


@dataclass
class _ChunkArrays:
    student_key: np.ndarray  # (n,) object str
    item_key: np.ndarray  # (n,) object str
    hier: np.ndarray
    prob: np.ndarray
    step: np.ndarray
    response: np.ndarray  # (n,) int8
    row_num: np.ndarray  # (n,) int64
    timestamp: np.ndarray  # (n,) object str
    kc_name_matrix: np.ndarray  # (n, arity_chunk) object, NaN-padded


def _process_chunk(chunk: pd.DataFrame, kc_model: str) -> tuple[Optional[_ChunkArrays], dict]:
    item_key = (
        chunk["Problem Hierarchy"].fillna("") + "||"
        + chunk["Problem Name"].fillna("") + "||"
        + chunk["Step Name"].fillna("")
    )
    cfa = pd.to_numeric(chunk["Correct First Attempt"], errors="coerce")
    valid_cfa = cfa.notna() & cfa.isin([0, 1])
    valid_cfa_np = valid_cfa.to_numpy()
    n_invalid_cfa = int((~valid_cfa).sum())

    student_raw = chunk["Anon Student Id"]
    has_student = (student_raw.notna() & (student_raw != "")).to_numpy()
    # Scoped like triage_kdd's own "would otherwise be usable" universe
    # (valid_cfa), not the raw row count -- see module docstring note 8.
    n_missing_student = int((valid_cfa_np & ~has_student).sum())

    kc_col, opp_col = f"KC({kc_model})", f"Opportunity({kc_model})"
    kc_raw = chunk[kc_col]
    has_kc = (kc_raw.notna() & (kc_raw != "")).to_numpy()

    kc_lists_all = kc_raw.str.split("~~")
    opp_lists_all = chunk[opp_col].str.split("~~")
    len_kc = kc_lists_all.str.len().to_numpy()
    len_opp = opp_lists_all.str.len().to_numpy()
    length_ok = np.where(has_kc, len_kc == len_opp, False)
    # triage_kdd.py's `mask` is `valid_cfa & has_kc`; its `n_kc_opp_mismatch`
    # and `a_overall.n` (the "usable tag" count) are BOTH scoped to that same
    # mask (computed before restricting to `match`/`length_ok`), never to the
    # raw row count or to `has_kc` alone. Reproduce that scope exactly so a
    # row that is both invalid-CFA and length-mismatched -- or both
    # length-mismatched and otherwise-valid -- is attributed identically to
    # both scripts (this was previously a definitional mismatch: see
    # `_planning/design/realbed_bridge_notes.md` section 3).
    scoped_has_kc = valid_cfa_np & has_kc
    n_mismatch = int((scoped_has_kc & ~length_ok).sum())
    n_has_kc = int((scoped_has_kc & length_ok).sum())

    tagged_mask = has_kc & length_ok  # rows whose KC tags are usable
    # Every valid-CFA row with a resolvable student id becomes an
    # interaction (note 1); a missing student id makes the row
    # unattributable to any `LearnerLog` and is excluded, not silently
    # folded into a bogus "nan" learner (note 8).
    keep_mask = valid_cfa_np & has_student
    n_used = int(keep_mask.sum())
    stats = {
        "n_rows": len(chunk), "n_invalid_cfa": n_invalid_cfa,
        "n_missing_student": n_missing_student,
        "n_has_kc": n_has_kc, "n_mismatch": n_mismatch, "n_used": n_used,
        "n_bad_row_num": 0, "n_missing_timestamp": 0,
    }
    if n_used == 0:
        return None, stats

    idx = np.where(keep_mask)[0]
    kc_lists_np = kc_lists_all.to_numpy()
    tagged_used = tagged_mask[idx]
    tag_lists_used = [
        kc_lists_np[pos] if tagged_used[i] else []
        for i, pos in enumerate(idx)
    ]
    kc_df = pd.DataFrame(tag_lists_used)
    kc_name_matrix = kc_df.to_numpy(dtype=object) if kc_df.shape[1] else np.zeros((n_used, 0), dtype=object)

    # `Row` is only ever used as a same-timestamp/-student tie-break (note
    # 5); an unparseable value must not silently become an
    # implementation-defined int64 (a raw NaN->int64 cast) that could
    # corrupt that tie-break with zero diagnostic trace (note 8). Count it
    # and fall back to a sentinel that sorts deterministically LAST within
    # its tie group -- conservative (never claims false precedence) and
    # visible (the counter fires whenever this branch is taken).
    row_num_f = pd.to_numeric(chunk["Row"], errors="coerce").to_numpy()[idx]
    bad_row = np.isnan(row_num_f)
    n_bad_row = int(bad_row.sum())
    if n_bad_row:
        # `int64`'s max value is not exactly representable as a `float64`
        # (53-bit mantissa vs. 63-bit magnitude), so casting a float sentinel
        # straight to int64 overflows into undefined behavior. Zero out the
        # bad slots before the cast, then overwrite them with the sentinel
        # AFTER casting to int64, entirely in integer arithmetic.
        row_num_f = np.where(bad_row, 0.0, row_num_f)
        row_num_arr = row_num_f.astype(np.int64)
        row_num_arr[bad_row] = np.iinfo(np.int64).max
    else:
        row_num_arr = row_num_f.astype(np.int64)

    # `Step Start Time` already falls back to "" (sorts first) when
    # missing; only the diagnostic counter was absent (note 8).
    ts_raw = chunk["Step Start Time"]
    missing_ts = (ts_raw.isna() | (ts_raw == "")).to_numpy()[idx]
    n_missing_ts = int(missing_ts.sum())

    stats["n_bad_row_num"] = n_bad_row
    stats["n_missing_timestamp"] = n_missing_ts

    arrays = _ChunkArrays(
        student_key=chunk["Anon Student Id"].to_numpy()[idx],
        item_key=item_key.to_numpy()[idx],
        hier=chunk["Problem Hierarchy"].fillna("").to_numpy()[idx],
        prob=chunk["Problem Name"].fillna("").to_numpy()[idx],
        step=chunk["Step Name"].fillna("").to_numpy()[idx],
        response=cfa.to_numpy()[idx].astype(np.int8),
        row_num=row_num_arr,
        timestamp=chunk["Step Start Time"].fillna("").to_numpy()[idx],
        kc_name_matrix=kc_name_matrix,
    )
    return arrays, stats


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LoadStats:
    n_rows_read: int = 0
    n_rows_invalid_cfa: int = 0
    n_rows_missing_student_id: int = 0  # valid-CFA rows dropped: no resolvable Anon Student Id
    # Both of the next two are scoped to triage_kdd's own `mask` (valid_cfa &
    # has_kc), matching its `a_overall.n` / `n_kc_opp_mismatch` exactly --
    # NOT to the raw row count or to `has_kc` alone (see `_process_chunk`).
    n_rows_has_kc: int = 0  # matches triage_kdd's per-model "a_overall.n" basis
    n_kc_opp_mismatch: int = 0
    n_rows_used: int = 0  # valid-CFA rows with a resolvable student id (note 1); >= n_rows_has_kc
    n_rows_bad_row_num: int = 0  # used rows whose "Row" value was unparseable (sentinel tie-break, note 8)
    n_rows_missing_timestamp: int = 0  # used rows whose "Step Start Time" was missing/blank (note 8)


@dataclass
class KddLoadResult:
    learners: list  # list[LearnerLog], learner id = position (synth.py convention)
    n_learners: int
    n_kcs: int
    kc_names: list  # index = kc id
    item_hierarchy: ItemHierarchy
    qmatrix: qmatrix_mod.QMatrix
    learner_timestamps: list  # list[np.ndarray], aligned with `learners`
    stats: LoadStats


# ---------------------------------------------------------------------------
# Chronological ordering
# ---------------------------------------------------------------------------


def _chronological_order(student_id: np.ndarray, timestamp: np.ndarray, row_num: np.ndarray) -> np.ndarray:
    """Stable three-key sort (module docstring note 5): primarily by
    student, ties broken by timestamp, ties broken by file row order.
    Composed as three chained stable sorts (each stage's own tie-break is
    exactly the previous stage's already-correct order) rather than
    `np.lexsort`, since `lexsort` does not support object/string dtype
    keys uniformly across numpy versions."""
    order = np.argsort(row_num, kind="stable")
    order = order[np.argsort(timestamp[order], kind="stable")]
    order = order[np.argsort(student_id[order], kind="stable")]
    return order


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_kdd_kc_traced(
    path,
    kc_model: str = KC_MODEL_DEFAULT,
    chunksize: int = 500_000,
    nrows: Optional[int] = None,
    min_pure_anchors: int = 3,
) -> KddLoadResult:
    """Streams the KDD Algebra 2008-2009 transaction log
    (`_planning/design/a4_design.md` v1.1 section 6) in chunks and builds
    the real-bed bridge: per-learner `LearnerLog`s (ordered
    chronologically, one interaction per valid-CFA row, KTracedSkills
    tags where usable), the 3-level item hierarchy
    (`bank.build_kdd_hierarchy`), and the static `qmatrix.QMatrix`.

    ``nrows`` limits the total rows read (debug/test only, matching
    `triage_kdd.py`'s own ``--nrows`` convention); ``None`` streams the
    full file.
    """
    item_vocab_to_id: dict = {}
    item_vocab_names: list = []
    hier_by_item: list = []
    prob_by_item: list = []
    step_by_item: list = []
    kc_vocab_to_id: dict = {}
    kc_vocab_names: list = []
    student_vocab_to_id: dict = {}
    student_vocab_names: list = []

    chunk_student_ids: list = []
    chunk_item_ids: list = []
    chunk_responses: list = []
    chunk_row_nums: list = []
    chunk_timestamps: list = []
    chunk_kc_ids: list = []  # each (n_i, arity_i) int64, arity_i may differ per chunk

    stats = LoadStats()
    item_kc_recorded: dict = {}  # item id -> list[int] (first tagged occurrence)

    reader = pd.read_csv(
        path, sep="\t", usecols=_usecols(kc_model), dtype=str,
        chunksize=chunksize, nrows=nrows, engine="c",
        quoting=csv.QUOTE_NONE, na_filter=True,
    )
    for chunk in reader:
        arrays, chunk_stats = _process_chunk(chunk, kc_model)
        stats.n_rows_read += chunk_stats["n_rows"]
        stats.n_rows_invalid_cfa += chunk_stats["n_invalid_cfa"]
        stats.n_rows_missing_student_id += chunk_stats["n_missing_student"]
        stats.n_rows_has_kc += chunk_stats["n_has_kc"]
        stats.n_kc_opp_mismatch += chunk_stats["n_mismatch"]
        stats.n_rows_used += chunk_stats["n_used"]
        stats.n_rows_bad_row_num += chunk_stats["n_bad_row_num"]
        stats.n_rows_missing_timestamp += chunk_stats["n_missing_timestamp"]
        if arrays is None:
            continue

        student_ids = _intern_new_only(arrays.student_key, student_vocab_to_id, student_vocab_names)
        item_ids = _intern_items(
            arrays.item_key, arrays.hier, arrays.prob, arrays.step,
            item_vocab_to_id, item_vocab_names, hier_by_item, prob_by_item, step_by_item,
        )
        kc_ids = _intern_kc_matrix(arrays.kc_name_matrix, kc_vocab_to_id, kc_vocab_names)

        # Record each item's first tagged occurrence's KC set (note 4).
        row_has_tag = (kc_ids >= 0).any(axis=1) if kc_ids.size else np.zeros(len(item_ids), dtype=bool)
        if row_has_tag.any():
            tagged_positions = np.where(row_has_tag)[0]
            seen_this_chunk: set = set()
            for pos in tagged_positions:
                item = int(item_ids[pos])
                if item in item_kc_recorded or item in seen_this_chunk:
                    continue
                row = kc_ids[pos]
                item_kc_recorded[item] = row[row >= 0].tolist()
                seen_this_chunk.add(item)

        chunk_student_ids.append(student_ids)
        chunk_item_ids.append(item_ids)
        chunk_responses.append(arrays.response)
        chunk_row_nums.append(arrays.row_num)
        chunk_timestamps.append(arrays.timestamp)
        chunk_kc_ids.append(kc_ids)

    n_items = len(item_vocab_names)
    n_kcs = len(kc_vocab_names)
    n_learners = len(student_vocab_names)

    if not chunk_student_ids:
        empty_hierarchy = build_kdd_hierarchy(np.zeros(0, dtype=object), np.zeros(0, dtype=object), np.zeros(0, dtype=object))
        empty_qmatrix = qmatrix_mod.build_qmatrix([], [], [], arity_max=1)
        return KddLoadResult(
            learners=[], n_learners=0, n_kcs=0, kc_names=[],
            item_hierarchy=empty_hierarchy, qmatrix=empty_qmatrix,
            learner_timestamps=[], stats=stats,
        )

    global_arity = max((arr.shape[1] for arr in chunk_kc_ids), default=1)
    global_arity = max(global_arity, 1)
    padded_kc_ids = []
    for arr in chunk_kc_ids:
        if arr.shape[1] < global_arity:
            pad = np.full((arr.shape[0], global_arity - arr.shape[1]), -1, dtype=np.int64)
            arr = np.concatenate([arr, pad], axis=1)
        padded_kc_ids.append(arr)

    student_all = np.concatenate(chunk_student_ids)
    item_all = np.concatenate(chunk_item_ids)
    response_all = np.concatenate(chunk_responses)
    row_all = np.concatenate(chunk_row_nums)
    ts_all = np.concatenate(chunk_timestamps)
    kc_ids_all = np.concatenate(padded_kc_ids, axis=0) if padded_kc_ids else np.zeros((0, global_arity), dtype=np.int64)

    order = _chronological_order(student_all, ts_all, row_all)
    student_o = student_all[order]
    item_o = item_all[order]
    response_o = response_all[order]
    ts_o = ts_all[order]
    kc_ids_o = kc_ids_all[order]

    change = np.empty(len(student_o), dtype=bool)
    change[0] = True
    change[1:] = student_o[1:] != student_o[:-1]
    starts = np.where(change)[0]
    ends = np.append(starts[1:], len(student_o))

    learners: list = []
    learner_timestamps: list = []
    for local_idx, (s, e) in enumerate(zip(starts, ends)):
        learners.append(
            LearnerLog(
                learner=local_idx,
                item_ids=item_o[s:e].copy(),
                responses=response_o[s:e].astype(np.int8).copy(),
                tag_ids=kc_ids_o[s:e].copy(),
                tag_mask=(kc_ids_o[s:e] >= 0).copy(),
            )
        )
        learner_timestamps.append(ts_o[s:e].copy())

    hierarchy = build_kdd_hierarchy(
        np.asarray(hier_by_item, dtype=object),
        np.asarray(prob_by_item, dtype=object),
        np.asarray(step_by_item, dtype=object),
    )

    item_kc_id_lists = [item_kc_recorded.get(i, []) for i in range(n_items)]
    qmatrix = qmatrix_mod.build_qmatrix(
        item_keys=item_vocab_names,
        item_kc_id_lists=item_kc_id_lists,
        kc_names=kc_vocab_names,
        arity_max=max(global_arity, max((len(t) for t in item_kc_id_lists), default=1)),
    )

    return KddLoadResult(
        learners=learners,
        n_learners=n_learners,
        n_kcs=n_kcs,
        kc_names=kc_vocab_names,
        item_hierarchy=hierarchy,
        qmatrix=qmatrix,
        learner_timestamps=learner_timestamps,
        stats=stats,
    )


__all__ = [
    "KC_MODEL_DEFAULT",
    "LoadStats",
    "KddLoadResult",
    "load_kdd_kc_traced",
]
