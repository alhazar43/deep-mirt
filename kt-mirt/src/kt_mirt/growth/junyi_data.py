"""A4 real-bed bridge: Junyi Academy 2015 (EDM 2015 exercise-relationship
release) chunked loader (`_planning/design/a4_design.md` v1.1; mirrors
`kt_mirt.growth.kc_data`'s hostile-reviewed KDD bridge structure, adapted
to Junyi15's two-table schema). Produces the per-learner event streams
`kt_mirt.growth.bank`/`kt_mirt.growth.slices` consume: one
`kt_mirt.growth.synth.LearnerLog` per student (ordered by true
chronological arrival via ``time_done``, learner id = the 0-based
position in the returned list, matching `kc_data.py`'s and `synth.py`'s
own convention), plus the static item bank (`kt_mirt.growth.qmatrix.
QMatrix` and a flat `kt_mirt.growth.bank.ItemHierarchy`) the bank
calibrator and per-KC readouts need.

Parsing reuses `scripts/triage/triage_junyi15.py`'s proven logic exactly
(same two files, same columns, same exercise-table dedup rule, same
"correct" convention): this loader's real-data agreement table against
`_planning/triage/junyi15_stats.json` was produced by an ad-hoc,
scratchpad-only check (large full-file runs stay out of the committed
test suite by design); a skip-if-absent integration test in
`tests/test_growth_junyi_data.py` exercises the real file lightly
instead.

Two source files:
  - ``junyi_Exercise_table.csv``: one row per exercise (835 distinct after
    dropping 2 exact-duplicate rows), carrying the ``topic`` tag (the KC
    layer chosen here, per the task spec -- ``area``, 8 values, is a
    coarser split not used) and a ``prerequisites`` cell (exercise-to-
    exercise edges, exposed by `load_prerequisite_graph` for the later
    cross-KC-influence goal, independent of the KC-traced loader below).
  - ``junyi_ProblemLog_original.csv``: one row per problem attempt
    (``user_id``, ``exercise``, ``correct``, ``time_done``).
    ``problem_number`` is not read here: `bank.build_calibration_rows`
    derives its own per-tag opportunity index from interaction ORDER, not
    from a raw column (`slices.py` module docstring), so no per-exercise
    running count needs to survive into `LearnerLog`.

Interpretation notes (recorded per the harness instructions):

1. **Item identity is the exercise name** (the ``exercise``/``name``
   column), never the topic -- disjoint vocabularies by construction
   (mirrors `kc_data.py` note 3's ASSISTments-circularity avoidance).
   `qmatrix.build_qmatrix`'s circularity guard still runs as an
   independent runtime check.
2. **The item and KC vocabularies are built from the FULL exercise
   catalog, not from the log.** Unlike KDD (one transaction file IS the
   item catalog), Junyi15 ships a separate, static exercise table; the
   triage's own methodology note for this bed states the static Q-matrix
   is built "from the FULL exercise bank ... not restricted to the log
   sample -- same convention as EdNet/Junyi2020" (`triage_junyi15.py`
   lines 374-376). This loader follows that convention: `n_items` = 835
   (the full deduped catalog) and `n_kcs` = 40 (every distinct ``topic``
   value in that catalog), fixed BEFORE the log is streamed. A row whose
   exercise carries no topic (19 of 835 catalog exercises) is retained as
   an untagged interaction (all-``False`` tag mask) -- the same rationale
   as `kc_data.py` note 1: the response is still real practice-event
   evidence for item-difficulty calibration even without a usable KC tag.
   Because the KC vocabulary can therefore contain a topic with ZERO
   log-tagged interactions (if every exercise under it happens to be
   unattempted or itself untagged), `JunyiLoadResult.n_kcs_attempted`
   additionally reports the log-observed count (39, matching
   `triage_junyi15.py`'s `kc_models.topic.n_distinct_kc`) alongside the
   full-catalog `n_kcs` (40) used to size every downstream per-KC array,
   so the two are never conflated.
3. **Two exact-duplicate exercise-table rows are dropped**
   (``matrix_mul_two``, ``matrix_app_fruit_oil``, each duplicated
   verbatim), via ``drop_duplicates(subset="name", keep="first")``,
   identical to `triage_junyi15.py`'s own rule.
4. **Every exercise carries at most ONE topic** (this release's KC tag is
   single-valued, unlike KDD's ``~~``-multi-tag cells), so ``tag_ids``/
   ``tag_mask`` here are always shape ``(T, 1)`` and `qmatrix.QMatrix.
   arity_max` is always 1.
5. **Chronological order** is ``time_done`` (a plain Unix-microsecond
   integer column, parsed numerically -- no lexicographic-string trick
   needed, unlike KDD's formatted date string), tie-broken by the file's
   own row order. This release ships no analogue of KDD's ``Row`` column;
   a stable three-key sort chain's LAST tie-break IS the physical file
   order, exactly as `triage_junyi15.py`'s own
   ``sort_values(["user_id", "time_done"], kind="stable")`` relies on.
6. **No item hierarchy beyond flat.** Unlike KDD's 3-level Problem
   Hierarchy > Problem > Step structure, an exercise here has no natural
   multi-level ancestry (the exercise table's ``h_position``/
   ``v_position`` are knowledge-map plotting coordinates, not a content
   hierarchy). `JunyiLoadResult.item_hierarchy` is built via
   `bank.flat_hierarchy` -- the same single-level code path `run.py`'s
   synthetic cells and (per `bank.py` note 5) EdNet already use -- not a
   bespoke Junyi hierarchy builder.
7. **Vocabulary interning for ``user_id`` reuses the same vectorized
   "map-then-python-loop-for-new-only" strategy** `kc_data.py` built for
   KDD's much larger item vocabulary (`_intern_new_only` there); Junyi's
   user vocabulary (247,606 students) is smaller but still large enough
   that a per-row Python dict lookup across 26M rows is worth avoiding.
   Item and KC ids need no such incremental interning (note 2: both
   vocabularies are fixed upfront from the small, 835-row exercise
   table), so the log-streaming side only ever does a vectorized
   ``pandas.Series.map`` against those two fixed dictionaries.
8. **Every raw-column parse failure is counted, not silently absorbed**
   (`kc_data.py` note 8's discipline applied to this bed's columns): a
   missing/unresolvable ``user_id``, an ``exercise`` value not found in
   the exercise-table vocabulary, a ``correct`` value that is not exactly
   ``"true"``/``"false"``, or a missing/unparseable ``time_done`` each get
   their own `LoadStats` counter. A row missing ``user_id`` or carrying an
   unresolvable ``exercise`` cannot be attributed to any `LearnerLog`
   entry (the former) or interned item id (the latter) and is excluded
   from the interaction stream entirely; an invalid ``correct`` value is
   likewise excluded (no natural fallback response). A missing
   ``time_done`` falls back to the sentinel ``0`` (sorts first, the same
   "missing sorts first" convention `kc_data.py` uses for KDD's empty-
   string timestamp fallback) and is separately counted.
9. **The prerequisite graph is exposed but not wired into the KC layer.**
   `load_prerequisite_graph` returns the (deduped) exercise-to-exercise
   edges from the ``prerequisites`` column, in both raw name and (when an
   item-name vocabulary is supplied) interned item-id form, for the later
   cross-KC-influence work the design flags Junyi15 for (its human-
   curated prerequisite graph is this bed's distinguishing asset per the
   triage). It is independent of `load_junyi_kc_traced` -- call order does
   not matter -- and never affects the LearnerLog/QMatrix outputs.
10. **`max_learners` deterministically subsamples LEARNERS, never KCs**
    (added for the Junyi real-bed pilot: 247,606 students is too many for
    a permutation-null replicate to rebuild every learner's slices in
    memory). When set, a memory-bounded pre-pass streams ONLY the
    ProblemLog's ``user_id`` column, in the same chunk size as the main
    pass, to collect the file's distinct raw ids (247,606 short strings,
    never the full 26M rows); it then keeps the first `max_learners` of
    those ids in lexicographic (string) sort order -- never cast to
    numeric, matching note 7's own treatment of ``user_id`` as an opaque
    string key. The main pass skips every row whose learner is not in
    that kept set exactly like any other excluded row (note 8's
    discipline: counted, not silently absorbed, under `LoadStats.
    n_rows_excluded_by_subsample`), so its row-array memory scales with
    the kept learners' share of the log, not the full cohort. `n_learners`
    /`LoadStats.n_learners_kept` are re-derived from the post-subsample
    interned vocabulary (learners actually surviving into the output,
    which can be fewer than `max_learners` if a kept id's own rows are all
    otherwise invalid); the item/KC catalog (`n_items`, `n_kcs`,
    `kc_names`) is untouched, since note 2 already fixes it from the
    exercise table before the log is streamed, independent of which
    learners are kept. ``None`` (default) skips the pre-pass entirely and
    is byte-identical to this parameter not existing.
11. **`subsample_seed` switches the kept-id SELECTION RULE from
    deterministic first-N to a seeded random draw** (added because
    first-N-by-sort is a non-random, potentially biased cohort: e.g. if
    ``user_id`` correlates with signup order, the first N ids in sort
    order are the earliest-signup students, not a representative sample).
    ``subsample_seed=None`` (default) keeps note 10's existing rule
    exactly (the first `max_learners` distinct ids in sorted order) --
    byte-identical to this parameter not existing. When `subsample_seed`
    is an int, `_collect_kept_user_ids` instead sorts the full distinct-id
    set (so the draw is insensitive to chunk size or set/dict iteration
    order, matching note 10's own reproducibility discipline) and draws
    `max_learners` of them via ``random.Random(subsample_seed).sample``,
    still fully deterministic for a fixed seed but no longer biased
    toward the lexicographically-first ids. Everything else about note
    10 (memory bound, the excluded-row accounting, the untouched item/KC
    catalog) is unaffected by which selection rule is in force.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kt_mirt.growth import bank as bank_mod
from kt_mirt.growth import qmatrix as qmatrix_mod
from kt_mirt.growth.bank import ItemHierarchy
from kt_mirt.growth.synth import LearnerLog

PROBLEM_LOG_COLS = ("user_id", "exercise", "correct", "time_done")


# ---------------------------------------------------------------------------
# Exercise table: full item + KC vocabulary (module docstring note 2)
# ---------------------------------------------------------------------------


@dataclass
class ExerciseTableInfo:
    item_names: list  # length n_items, index = item id, deduped catalog order
    item_name_to_id: dict
    kc_names: list  # length n_kcs, index = kc id, first-seen order among items
    kc_name_to_id: dict
    item_kc_id: np.ndarray  # (n_items,) int64, -1 if the item carries no topic
    n_exact_duplicate_rows_dropped: int
    n_exercises_missing_topic: int
    raw_table: "pd.DataFrame"  # deduped, kept for `load_prerequisite_graph`'s reuse


def _load_exercise_table(path) -> ExerciseTableInfo:
    ex_raw = pd.read_csv(path, encoding="utf-8-sig")
    n_rows_raw = len(ex_raw)
    ex = ex_raw.drop_duplicates(subset="name", keep="first").reset_index(drop=True)
    n_dup_dropped = n_rows_raw - len(ex)

    item_names = ex["name"].tolist()
    item_name_to_id = {name: i for i, name in enumerate(item_names)}

    kc_name_to_id: dict = {}
    kc_names: list = []
    item_kc_id = np.full(len(item_names), -1, dtype=np.int64)
    n_missing_topic = 0
    for i, topic in enumerate(ex["topic"]):
        if pd.isna(topic):
            n_missing_topic += 1
            continue
        kc_id = kc_name_to_id.get(topic)
        if kc_id is None:
            kc_id = len(kc_names)
            kc_name_to_id[topic] = kc_id
            kc_names.append(topic)
        item_kc_id[i] = kc_id

    return ExerciseTableInfo(
        item_names=item_names,
        item_name_to_id=item_name_to_id,
        kc_names=kc_names,
        kc_name_to_id=kc_name_to_id,
        item_kc_id=item_kc_id,
        n_exact_duplicate_rows_dropped=n_dup_dropped,
        n_exercises_missing_topic=n_missing_topic,
        raw_table=ex,
    )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LoadStats:
    n_rows_read: int = 0
    n_rows_missing_user_id: int = 0
    n_rows_exercise_not_in_table: int = 0  # includes a missing/blank exercise field
    n_rows_invalid_correct: int = 0
    n_rows_missing_timestamp: int = 0
    n_rows_used: int = 0  # kept as an interaction (note 8)
    n_rows_topic_unmapped: int = 0  # used rows whose item carries no topic (note 2)
    n_rows_excluded_by_subsample: int = 0  # has a learner, but dropped by max_learners (note 10)
    n_learners_kept: int = 0  # distinct learners actually in the output (note 10; == n_learners)


@dataclass
class JunyiLoadResult:
    learners: list  # list[LearnerLog], learner id = position (synth.py convention)
    n_learners: int
    n_kcs: int  # full exercise-table topic catalog size (note 2); sizes every per-KC array
    kc_names: list  # index = kc id, length n_kcs
    n_kcs_attempted: int  # topics with >=1 tagged log interaction (note 2)
    item_names: list  # index = item id, length n_items (full deduped exercise catalog)
    item_hierarchy: ItemHierarchy
    qmatrix: qmatrix_mod.QMatrix
    learner_timestamps: list  # list[np.ndarray], aligned with `learners`
    stats: LoadStats


# ---------------------------------------------------------------------------
# ProblemLog chunk processing
# ---------------------------------------------------------------------------


@dataclass
class _ChunkArrays:
    user_key: np.ndarray  # (n,) object str
    item_id: np.ndarray  # (n,) int64
    response: np.ndarray  # (n,) int8
    timestamp: np.ndarray  # (n,) int64
    row_pos: np.ndarray  # (n,) int64, global physical row order (tie-break, note 5)


def _process_chunk(
    chunk: pd.DataFrame, item_name_to_id: dict, row_offset: int,
    kept_user_ids: Optional[set] = None,
) -> tuple[Optional[_ChunkArrays], dict]:
    user_raw = chunk["user_id"]
    has_user = (user_raw.notna() & (user_raw.astype(str) != "")).to_numpy()

    # `max_learners` subsampling (module docstring note 10). ``None`` (the
    # default, no cap) makes `in_subsample` all-True, so `keep_mask` below
    # is untouched and this function is byte-identical to its pre-cap
    # behavior. Restricted to rows that already `has_user`, so a row with
    # no attributable learner at all is never double-counted as "excluded
    # by subsample" on top of "missing user id".
    if kept_user_ids is None:
        in_subsample = np.ones(len(chunk), dtype=bool)
        n_excluded_by_subsample = 0
    else:
        in_subsample = has_user & user_raw.astype(str).isin(kept_user_ids).to_numpy()
        n_excluded_by_subsample = int((has_user & ~in_subsample).sum())

    exercise_raw = chunk["exercise"].astype(str)
    item_id = exercise_raw.map(item_name_to_id)
    has_item = item_id.notna().to_numpy()

    correct_raw = chunk["correct"].astype(str)
    valid_correct = correct_raw.isin(["true", "false"]).to_numpy()
    response_all = np.where(correct_raw.to_numpy() == "true", 1, 0).astype(np.int8)

    ts_num = pd.to_numeric(chunk["time_done"], errors="coerce")
    missing_ts = ts_num.isna().to_numpy()
    ts_all = ts_num.fillna(0).to_numpy(dtype=np.int64)

    keep_mask = has_user & has_item & valid_correct & in_subsample
    n_used = int(keep_mask.sum())
    stats = {
        "n_rows": len(chunk),
        "n_missing_user": int((~has_user).sum()),
        "n_exercise_not_in_table": int((~has_item).sum()),
        "n_invalid_correct": int((~valid_correct).sum()),
        "n_used": n_used,
        "n_missing_timestamp": 0,
        "n_excluded_by_subsample": n_excluded_by_subsample,
    }
    if n_used == 0:
        return None, stats

    idx = np.where(keep_mask)[0]
    stats["n_missing_timestamp"] = int(missing_ts[idx].sum())

    arrays = _ChunkArrays(
        user_key=user_raw.astype(str).to_numpy()[idx],
        # Select the resolvable subset BEFORE casting to int64: `item_id`
        # still carries NaN at unresolved positions, and casting the whole
        # column first would be exactly the implementation-defined
        # NaN->int64 cast `kc_data.py` note 8 warns against, even though
        # those positions are discarded by `idx` immediately after.
        item_id=item_id.iloc[idx].to_numpy(dtype=np.int64),
        response=response_all[idx],
        timestamp=ts_all[idx],
        row_pos=(row_offset + idx).astype(np.int64),
    )
    return arrays, stats


# ---------------------------------------------------------------------------
# User-id interning (module docstring note 7; mirrors `kc_data._intern_new_only`)
# ---------------------------------------------------------------------------


def _intern_new_only(values: np.ndarray, vocab_to_id: dict, vocab_names: list) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Chronological ordering (module docstring note 5)
# ---------------------------------------------------------------------------


def _chronological_order(student_id: np.ndarray, timestamp: np.ndarray, row_pos: np.ndarray) -> np.ndarray:
    """Stable three-key sort chain: student id, ties broken by timestamp,
    ties broken by physical file row order (this bed's own analogue of
    `kc_data._chronological_order`, with `row_pos` standing in for KDD's
    ``Row`` column, which this release does not ship)."""
    order = np.argsort(row_pos, kind="stable")
    order = order[np.argsort(timestamp[order], kind="stable")]
    order = order[np.argsort(student_id[order], kind="stable")]
    return order


# ---------------------------------------------------------------------------
# Learner subsampling pre-pass (module docstring note 10)
# ---------------------------------------------------------------------------


def _collect_kept_user_ids(
    problem_log_path, max_learners: int, chunksize: int, nrows: Optional[int] = None,
    subsample_seed: Optional[int] = None,
) -> set:
    """Pre-pass for `max_learners` subsampling. Streams ONLY the
    ``user_id`` column of the ProblemLog file in chunks (same chunk size
    and ``nrows`` truncation as the main pass, so both passes see an
    identical view of the file), accumulating the set of distinct,
    non-missing raw ids -- memory bounded by the number of DISTINCT
    learners (247,606 short strings for the full Junyi15 file), never by
    the row count (26M), unlike holding every row would be.

    ``subsample_seed=None`` (default, module docstring note 10) returns
    the first `max_learners` of those ids in lexicographic (string) sort
    order, as a set for O(1) membership testing in the main pass below --
    byte-identical to this parameter not existing. When `subsample_seed`
    is an int (module docstring note 11), a SEEDED RANDOM sample of
    `max_learners` ids is returned instead: the full distinct-id set is
    sorted first (so the draw does not depend on chunk size or set/dict
    iteration order), then ``random.Random(subsample_seed).sample`` draws
    from that fixed sequence -- a representative cohort rather than a
    lexicographically-first one, still fully reproducible for a given
    seed (same seed -> identical returned set).
    """
    seen: set = set()
    reader = pd.read_csv(
        problem_log_path, usecols=["user_id"], dtype=str, chunksize=chunksize, nrows=nrows,
    )
    for chunk in reader:
        user_raw = chunk["user_id"]
        valid = (user_raw.notna() & (user_raw.astype(str) != "")).to_numpy()
        if valid.any():
            seen.update(user_raw.astype(str).to_numpy()[valid].tolist())
    sorted_ids = sorted(seen)
    if subsample_seed is None:
        return set(sorted_ids[:max_learners])
    return set(random.Random(subsample_seed).sample(sorted_ids, min(max_learners, len(sorted_ids))))


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_junyi_kc_traced(
    problem_log_path,
    exercise_table_path,
    chunksize: int = 2_000_000,
    nrows: Optional[int] = None,
    max_learners: Optional[int] = None,
    subsample_seed: Optional[int] = None,
) -> JunyiLoadResult:
    """Streams the Junyi15 problem-attempt log
    (`_planning/design/a4_design.md` v1.1) in chunks and builds the
    real-bed bridge: per-learner `LearnerLog`s (ordered chronologically,
    one interaction per valid, resolvable row), a flat `ItemHierarchy`,
    and the static `qmatrix.QMatrix` (item/KC vocabularies fixed upfront
    from the exercise table, module docstring note 2).

    ``nrows`` limits the total ProblemLog rows read (debug/test only,
    matching `kc_data.load_kdd_kc_traced`'s own convention); ``None``
    streams the full file.

    ``max_learners`` deterministically caps the number of DISTINCT
    learners the returned bed contains (module docstring note 10): a
    memory-bounded pre-pass over the ``user_id`` column alone picks the
    first `max_learners` distinct ids in sorted order, and the main pass
    then skips every row belonging to any other learner, so the pilot's
    per-replicate memory scales with the kept cohort, not the full
    247,606-student file. The KC/item catalog is unaffected either way
    (fixed from the exercise table, note 2). ``None`` (default) keeps
    every learner in the file, byte-identical to the pre-existing
    behavior.

    ``subsample_seed`` (module docstring note 11) chooses which
    `max_learners` ids that pre-pass keeps: ``None`` (default) keeps
    `max_learners`'s own first-N-by-sort rule unchanged; an int instead
    draws a SEEDED RANDOM sample of `max_learners` ids from the full
    distinct-id set, for a representative (rather than lexicographically-
    first, potentially biased) cohort, deterministically reproducible for
    a given seed. Has no effect when `max_learners` is ``None`` (no
    subsampling pre-pass runs at all).
    """
    ex_info = _load_exercise_table(exercise_table_path)
    n_items = len(ex_info.item_names)
    n_kcs = len(ex_info.kc_names)

    kept_user_ids: Optional[set] = None
    if max_learners is not None:
        kept_user_ids = _collect_kept_user_ids(
            problem_log_path, max_learners, chunksize, nrows=nrows, subsample_seed=subsample_seed,
        )

    student_vocab_to_id: dict = {}
    student_vocab_names: list = []

    chunk_student_ids: list = []
    chunk_item_ids: list = []
    chunk_responses: list = []
    chunk_row_pos: list = []
    chunk_timestamps: list = []

    stats = LoadStats()
    row_offset = 0
    topics_attempted: set = set()

    reader = pd.read_csv(
        problem_log_path, usecols=list(PROBLEM_LOG_COLS), dtype=str,
        chunksize=chunksize, nrows=nrows,
    )
    for chunk in reader:
        arrays, chunk_stats = _process_chunk(
            chunk, ex_info.item_name_to_id, row_offset, kept_user_ids=kept_user_ids,
        )
        row_offset += len(chunk)
        stats.n_rows_read += chunk_stats["n_rows"]
        stats.n_rows_missing_user_id += chunk_stats["n_missing_user"]
        stats.n_rows_exercise_not_in_table += chunk_stats["n_exercise_not_in_table"]
        stats.n_rows_invalid_correct += chunk_stats["n_invalid_correct"]
        stats.n_rows_used += chunk_stats["n_used"]
        stats.n_rows_missing_timestamp += chunk_stats["n_missing_timestamp"]
        stats.n_rows_excluded_by_subsample += chunk_stats["n_excluded_by_subsample"]
        if arrays is None:
            continue

        student_ids = _intern_new_only(arrays.user_key, student_vocab_to_id, student_vocab_names)
        item_kc = ex_info.item_kc_id[arrays.item_id]
        n_unmapped = int((item_kc < 0).sum())
        stats.n_rows_topic_unmapped += n_unmapped
        topics_attempted.update(item_kc[item_kc >= 0].tolist())

        chunk_student_ids.append(student_ids)
        chunk_item_ids.append(arrays.item_id)
        chunk_responses.append(arrays.response)
        chunk_row_pos.append(arrays.row_pos)
        chunk_timestamps.append(arrays.timestamp)

    stats.n_learners_kept = len(student_vocab_names)

    hierarchy = bank_mod.flat_hierarchy(n_items)
    item_kc_id_lists = [([int(k)] if k >= 0 else []) for k in ex_info.item_kc_id.tolist()]
    qmatrix = qmatrix_mod.build_qmatrix(
        item_keys=ex_info.item_names,
        item_kc_id_lists=item_kc_id_lists,
        kc_names=ex_info.kc_names,
        arity_max=1,
    )

    if not chunk_student_ids:
        return JunyiLoadResult(
            learners=[], n_learners=0, n_kcs=n_kcs, kc_names=list(ex_info.kc_names),
            n_kcs_attempted=0, item_names=list(ex_info.item_names),
            item_hierarchy=hierarchy, qmatrix=qmatrix, learner_timestamps=[], stats=stats,
        )

    student_all = np.concatenate(chunk_student_ids)
    item_all = np.concatenate(chunk_item_ids)
    response_all = np.concatenate(chunk_responses)
    row_pos_all = np.concatenate(chunk_row_pos)
    ts_all = np.concatenate(chunk_timestamps)

    order = _chronological_order(student_all, ts_all, row_pos_all)
    student_o = student_all[order]
    item_o = item_all[order]
    response_o = response_all[order]
    ts_o = ts_all[order]

    item_kc_o = ex_info.item_kc_id[item_o]
    tag_ids_o = item_kc_o.reshape(-1, 1)
    tag_mask_o = (tag_ids_o >= 0)

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
                tag_ids=tag_ids_o[s:e].copy(),
                tag_mask=tag_mask_o[s:e].copy(),
            )
        )
        learner_timestamps.append(ts_o[s:e].copy())

    return JunyiLoadResult(
        learners=learners,
        n_learners=len(student_vocab_names),
        n_kcs=n_kcs,
        kc_names=list(ex_info.kc_names),
        n_kcs_attempted=len(topics_attempted),
        item_names=list(ex_info.item_names),
        item_hierarchy=hierarchy,
        qmatrix=qmatrix,
        learner_timestamps=learner_timestamps,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Prerequisite graph (module docstring note 9; not wired into the KC layer)
# ---------------------------------------------------------------------------


@dataclass
class PrerequisiteGraph:
    edges: list  # list[tuple[str, str]], (prerequisite_name, requires_it_name), deduped
    self_loop_exercises: list  # names that list themselves as their own prerequisite
    n_exercise_rows_raw: int
    n_exact_duplicate_rows_dropped: int
    id_edges: Optional[list] = None  # list[tuple[int, int]] when `item_name_to_id` is supplied
    n_edges_endpoint_not_in_vocab: int = 0


def load_prerequisite_graph(exercise_table_path, item_name_to_id: Optional[dict] = None) -> PrerequisiteGraph:
    """Parses ``junyi_Exercise_table.csv``'s ``prerequisites`` cell into a
    directed edge list (edge = prerequisite -> exercise-that-requires-it,
    `triage_junyi15.py`'s own convention), independent of
    `load_junyi_kc_traced`. When `item_name_to_id` is given (e.g.
    `JunyiLoadResult.item_names`'s enumerate-dict), also returns the
    edges interned into that vocabulary's item ids, dropping (and
    counting) any edge with an endpoint outside it."""
    ex_raw = pd.read_csv(exercise_table_path, encoding="utf-8-sig")
    n_rows_raw = len(ex_raw)
    ex = ex_raw.drop_duplicates(subset="name", keep="first").reset_index(drop=True)
    n_dup_dropped = n_rows_raw - len(ex)

    edges: set = set()
    for exname, praw in zip(ex["name"], ex["prerequisites"]):
        if pd.isna(praw):
            continue
        for p in str(praw).split(","):
            p = p.strip()
            if p:
                edges.add((p, exname))
    edge_list = sorted(edges)
    self_loops = sorted(u for u, v in edge_list if u == v)

    id_edges = None
    n_missing_vocab = 0
    if item_name_to_id is not None:
        id_edges = []
        for u, v in edge_list:
            iu, iv = item_name_to_id.get(u), item_name_to_id.get(v)
            if iu is None or iv is None:
                n_missing_vocab += 1
                continue
            id_edges.append((iu, iv))

    return PrerequisiteGraph(
        edges=edge_list,
        self_loop_exercises=self_loops,
        n_exercise_rows_raw=n_rows_raw,
        n_exact_duplicate_rows_dropped=n_dup_dropped,
        id_edges=id_edges,
        n_edges_endpoint_not_in_vocab=n_missing_vocab,
    )


__all__ = [
    "PROBLEM_LOG_COLS",
    "ExerciseTableInfo",
    "LoadStats",
    "JunyiLoadResult",
    "load_junyi_kc_traced",
    "PrerequisiteGraph",
    "load_prerequisite_graph",
]
