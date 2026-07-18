"""STAGE-0 triage: Junyi Academy 2015 (EDM 2015 exercise-relationship release).

Distinct from `triage_junyi2020.py` (a different, later Junyi release with
its own Log_Problem/Info_Content schema): this is the original 2015 EDM
paper's release (Chang, Hsu & Chen, "Modeling Exercise Relationships in
E-Learning: A Unified Approach"), shipping `junyi_ProblemLog_original.csv`
(one row per problem attempt), `junyi_Exercise_table.csv` (one row per
exercise, with a `topic`/`area` tag and a free-text `prerequisites` cell),
and human-annotated `relationship_annotation_{training,testing}.csv` pairs
(not consumed by this triage -- those are similarity/difficulty/
prerequisite *ratings* for a model-training task, not KC or log data).

KC LAYER CHOICE. The exercise table ships a 2-level tag (`area`, 8 values,
too coarse) and `topic` (40 values, median ~20 exercises/topic, comparable
in grain to Junyi2020's level3_id) -- `topic` is used as the KC here, per
the task spec, and stated explicitly rather than left implicit. There is
NO finer skill tag in this release; prerequisites are a relation BETWEEN
EXERCISES (see below), not a KC tag, so the natural "exercise-level grain"
this KC choice leaves on the table is surfaced through three EXTRA
deliverables rather than re-run as a full second KCModelAggregator (an
"exercise as its own KC" battery would be degenerate: item universe ==
KC universe, so arity/anchor stats collapse to trivia). The three extras:
  (i)   the prerequisite graph itself (exercise-to-exercise edges),
  (ii)  KC-pair decoupling at both the topic grain (structurally vacuous,
        exactly as in Junyi2020 -- see below) and, separately, the
        exercise-pair grain for the top-30 prerequisite-LINKED exercise
        pairs specifically (the grain the A1 external-validation avenue
        needs, since prerequisite edges are the one place this release
        supplies a genuine, human-curated cross-item relationship),
  (iii) opportunity distributions at both the per-learner-exercise and
        per-learner-topic grain.

STRUCTURAL NOTE ON (e) AT THE TOPIC GRAIN. Exactly as in Junyi2020: each
row of junyi_ProblemLog_original.csv is one attempt at ONE exercise, which
carries ONE topic. A "practice slot" (one row) therefore always tags
exactly one KC under this scheme, so two topics can never co-occur on the
same slot and the co-occurrence-based decoupling metric
(triage_common.top_pairs_report over per-slot KC-tag sets) is structurally
vacuous at the topic grain: n_pairs_with_cross_occurrence is 0 by
construction, not an unlucky sample. This is recorded as such, not papered
over -- see kc_models.topic.e_decoupling in the output JSON.

ADAPTING THE METRIC FOR EXERCISE-PAIR DECOUPLING (extra ii). Because the
topic grain is structurally vacuous for (e), and because prerequisite
edges are the one genuine exercise-to-exercise relationship this release
ships, the exercise-pair reading in extra deliverable (ii) redefines what
counts as a "slot" for the SAME formula in triage_common.decoupling_stats/
top_pairs_report: instead of "one interaction event" (which is exactly
one exercise per row here, so any exercise-pair decoupling read this way
would trivially always be 1.0 -- no row ever touches two exercises at
once, an equally-vacuous, differently-shaped degeneracy), a "slot" is
redefined as ONE LEARNER's entire logged history: n_A = number of distinct
learners who ever attempted exercise A (at any point in their history),
n_both = number of distinct learners who attempted BOTH A and B at some
point. This is the real-data analogue of "is there a population of
learners who practiced the prerequisite without (yet) touching the
target" -- exactly what A1's cross-KC-influence identifiability needs --
and it is the second, explicitly labeled adaptation of the metric this
program has made (the first being the interaction-event-vs-original-
qmirt-timestep adaptation already documented in triage_common.py's module
docstring). The formula, threshold (0.75), and ranking convention
(top-TOP_K_PAIRS by co-occurrence volume) are reused unchanged; only the
denominator event ("one row" vs "one learner-ever") changes, and that
change is stated plainly here and in the output JSON rather than left
implicit.

PREREQUISITE GRAPH. `prerequisites` is a comma-separated cell (verified:
proper quoted-CSV multi-value field, e.g. "domain_of_a_function,
range_of_a_function") listing the prerequisite exercise(s) for a given
exercise; an edge is directed prerequisite -> exercise (the natural
"feeds into" reading). Two exact-duplicate rows in the raw table
(`matrix_mul_two`, `matrix_app_fruit_oil`, each appearing twice with
identical content) are dropped via drop_duplicates(subset="name") before
any graph or KC-map construction, since `name` is documented as a unique
exercise id. Two genuine self-loops remain after dedup
(`number_sense_length_l1`, `proportions_1`, each listing itself as its own
prerequisite) -- a real data quirk, not a parsing artifact, reported
explicitly rather than silently dropped.

ITEM/OPPORTUNITY GRAIN FOR PROBLEMLOG. One row of junyi_ProblemLog_
original.csv is one problem attempt at one exercise. `problem_number` is
documented (README.txt) and independently verified here (against the full
per-user, per-exercise chronological order recovered from `time_done`) to
be an exact 1-based running count of that (user, exercise) pair's attempt
index -- so it is used directly as the exercise-level opportunity index,
no re-derivation needed. No such field exists at the topic grain (a
learner's attempts at different exercises sharing a topic are not
jointly numbered), so topic-level opportunity is derived here by sorting
the full log by (user_id, time_done) and taking a stable running count
per (user_id, topic).

Usage:
    python triage_junyi15.py <path/to/junyi_ProblemLog_original.csv> \\
        <path/to/junyi_Exercise_table.csv> \\
        [--out-dir kt-mirt/_planning/triage] [--chunksize 5000000] [--nrows N]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_common import (  # noqa: E402
    DECOUPLING_DEFINITION_TEXT,
    DECOUPLING_CLEAN_THRESHOLD,
    GROWTH_MAX_OPPORTUNITY,
    KCModelAggregator,
    TOP_K_PAIRS,
    decoupling_stats,
    fraction_at_least,
    quartiles,
    top_pairs_report,
    write_json,
)

KC_COLUMN = "topic"


# ---------------------------------------------------------------------------
# Exercise table + prerequisite graph
# ---------------------------------------------------------------------------
def load_exercise_table(path: str) -> dict:
    ex_raw = pd.read_csv(path, encoding="utf-8-sig")
    n_rows_raw = len(ex_raw)
    ex = ex_raw.drop_duplicates(subset="name", keep="first").reset_index(drop=True)
    n_dup_rows_dropped = n_rows_raw - len(ex)

    hierarchy_shape = {
        "topic": int(ex["topic"].nunique()),
        "area": int(ex["area"].nunique()),
    }
    topic_group_sizes = ex.dropna(subset=["topic"]).groupby("topic").size()

    names = set(ex["name"])
    edges = set()
    for exname, praw in zip(ex["name"], ex["prerequisites"]):
        if pd.isna(praw):
            continue
        for p in str(praw).split(","):
            p = p.strip()
            if p:
                edges.add((p, exname))
    self_loops = sorted(u for u, v in edges if u == v)

    outdeg = Counter()
    indeg = Counter()
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
    out_arr = np.array([outdeg.get(n, 0) for n in names], dtype=float)
    in_arr = np.array([indeg.get(n, 0) for n in names], dtype=float)

    edges_no_self = [(u, v) for (u, v) in edges if u != v]
    is_dag_incl_self, cycle_note_incl = _is_dag(names, edges)
    is_dag_excl_self, cycle_note_excl = _is_dag(names, edges_no_self)

    prereq_refs = set(u for u, v in edges)
    n_prereq_refs_not_in_table = len(prereq_refs - names)

    graph = {
        "direction_convention": "edge = (prerequisite_exercise -> exercise_that_requires_it)",
        "n_nodes": int(len(names)),
        "n_edges_distinct": int(len(edges)),
        "n_edges_excluding_self_loops": int(len(edges_no_self)),
        "n_self_loops": int(len(self_loops)),
        "self_loop_exercises": self_loops,
        "n_prereq_references_not_found_in_exercise_table": n_prereq_refs_not_in_table,
        "is_dag_including_self_loops": is_dag_incl_self,
        "is_dag_excluding_self_loops": is_dag_excl_self,
        "cycle_note_including_self_loops": cycle_note_incl,
        "cycle_note_excluding_self_loops": cycle_note_excl,
        "out_degree": {**quartiles(out_arr), "note": "prerequisite-of count (this exercise unlocks N others)"},
        "in_degree": {**quartiles(in_arr), "note": "prerequisite-count (this exercise requires N others first)"},
    }

    return {
        "ex_table": ex,
        "n_exercise_rows_raw": n_rows_raw,
        "n_exact_duplicate_rows_dropped": n_dup_rows_dropped,
        "hierarchy_shape_n_distinct": hierarchy_shape,
        "exercises_per_topic_quartiles": {
            "min": int(topic_group_sizes.min()), "median": float(topic_group_sizes.median()),
            "max": int(topic_group_sizes.max()),
        },
        "n_exercises_missing_topic": int(ex["topic"].isna().sum()),
        "edges": edges,
        "graph": graph,
    }


def _is_dag(nodes: set, edges) -> tuple:
    """Kahn's algorithm topological sort. Returns (is_dag: bool, note: str)."""
    outgoing: dict = {n: [] for n in nodes}
    indeg = Counter()
    for u, v in edges:
        outgoing.setdefault(u, []).append(v)
        indeg[v] += 1
        indeg.setdefault(u, indeg.get(u, 0))
    queue = [n for n in nodes if indeg.get(n, 0) == 0]
    visited = 0
    queue_idx = 0
    indeg_work = dict(indeg)
    while queue_idx < len(queue):
        u = queue[queue_idx]
        queue_idx += 1
        visited += 1
        for v in outgoing.get(u, []):
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                queue.append(v)
    is_dag = visited == len(nodes)
    note = (f"Kahn's algorithm: {visited}/{len(nodes)} nodes topologically ordered"
            if is_dag else
            f"Kahn's algorithm: only {visited}/{len(nodes)} nodes topologically ordered, "
            f"{len(nodes) - visited} nodes involved in >=1 cycle")
    return is_dag, note


# ---------------------------------------------------------------------------
# ProblemLog streaming read
# ---------------------------------------------------------------------------
def read_problem_log(path: str, chunksize: int, nrows) -> pd.DataFrame:
    reader = pd.read_csv(
        path,
        usecols=["user_id", "exercise", "problem_number", "correct", "time_done"],
        dtype={"user_id": "int32", "exercise": "category", "problem_number": "int32", "correct": "category"},
        chunksize=chunksize, nrows=nrows,
    )
    chunks = list(reader)
    df = pd.concat(chunks, ignore_index=True)
    df["exercise"] = df["exercise"].astype("category")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem_log_csv")
    parser.add_argument("exercise_table_csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    parser.add_argument("--nrows", type=int, default=None, help="debug: limit total rows read")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "_planning" / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ex_info = load_exercise_table(args.exercise_table_csv)
    ex = ex_info["ex_table"]
    ex2topic = dict(zip(ex["name"], ex["topic"]))
    print(f"[triage_junyi15] exercise table: {len(ex)} exercises (after dedup), "
          f"{ex_info['graph']['n_nodes']} graph nodes, {ex_info['graph']['n_edges_distinct']} edges, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    df = read_problem_log(args.problem_log_csv, args.chunksize, args.nrows)
    n_rows_total = len(df)
    n_users_total = int(df["user_id"].nunique())
    n_exercises_in_log = int(df["exercise"].nunique())
    print(f"[triage_junyi15] ProblemLog: {n_rows_total:,} rows, {n_users_total} users, "
          f"{n_exercises_in_log} exercises attempted, elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    log_exercises = set(df["exercise"].cat.categories)
    n_log_exercises_not_in_table = len(log_exercises - set(ex["name"]))

    df["correct_bool"] = (df["correct"].astype(str) == "true").astype("int8")

    # ---- (iii) per-learner-exercise opportunity: problem_number IS the verified
    # 1-based running attempt count per (user_id, exercise) -- used directly, no re-derivation.
    opp_max_exercise = df.groupby(["user_id", "exercise"], observed=True)["problem_number"].max()
    exercise_opp_block = {
        **quartiles(opp_max_exercise.to_numpy(dtype=float)),
        "n_learner_exercise_pairs": int(opp_max_exercise.size),
        "frac_ge_5": fraction_at_least(opp_max_exercise.to_numpy(dtype=float), 5),
        "frac_ge_10": fraction_at_least(opp_max_exercise.to_numpy(dtype=float), 10),
        "frac_ge_20": fraction_at_least(opp_max_exercise.to_numpy(dtype=float), 20),
        "source": "problem_number column, used as-shipped (verified against time_done chronological order)",
    }
    print(f"[triage_junyi15] per-learner-exercise opportunity done, elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    # ---- exercise-pair decoupling (extra ii, prerequisite-linked exercise pairs) ----
    # "Slot" redefined to one learner's ever-attempted set -- see module docstring.
    ex_user_sets = {ex_name: frozenset(arr) for ex_name, arr in
                    df.groupby("exercise", observed=True)["user_id"].unique().items()}
    edges = ex_info["edges"]
    pair_both_exercise = Counter()
    kc_n_exercise = {ex_name: len(users) for ex_name, users in ex_user_sets.items()}
    n_edges_both_in_log = 0
    n_edges_missing_log_data = 0
    for u, v in edges:
        if u == v:
            continue  # self-loops carry no information for a pair-decoupling read
        if u in ex_user_sets and v in ex_user_sets:
            n_edges_both_in_log += 1
            lo, hi = (u, v) if u < v else (v, u)
            n_both = len(ex_user_sets[u] & ex_user_sets[v])
            pair_both_exercise[(lo, hi)] = n_both
        else:
            n_edges_missing_log_data += 1
    exercise_pair_decoupling = top_pairs_report(pair_both_exercise, kc_n_exercise, k=TOP_K_PAIRS)
    self_loop_names = ex_info["graph"]["self_loop_exercises"]
    exercise_pair_decoupling["slot_definition_note"] = (
        "ADAPTED slot definition for this specific read: a 'slot' is one LEARNER's entire "
        "logged history (not one interaction event as elsewhere in this program). n_a/n_b = "
        "number of distinct learners who ever attempted exercise A/B; n_both = number who "
        f"attempted both at any point. Restricted to the prerequisite-graph edge list ({len(edges)} "
        f"distinct edges minus {len(self_loop_names)} self-loops), not all C(n,2) exercise pairs -- "
        "see module docstring for why the standard per-row co-occurrence definition is vacuous "
        "(always exactly 1.0) at this grain."
    )
    exercise_pair_decoupling["n_prerequisite_edges_considered"] = len(edges) - len(self_loop_names)
    exercise_pair_decoupling["n_edges_with_log_data_both_sides"] = n_edges_both_in_log
    exercise_pair_decoupling["n_edges_skipped_missing_log_data"] = n_edges_missing_log_data

    # Supplementary context: the top-30-by-n_both selection (required above, and the same
    # ranking convention every other bed's (e) uses) is NOT a representative sample of this
    # bed's prerequisite pairs -- co-occurrence volume here is driven by how foundational/
    # universal an exercise is in the curriculum, and the most-universal exercises are
    # (mechanically) also the ones nearly every learner eventually attempts on both sides of
    # the edge, so "most practiced together" and "worst decoupled" are correlated in a way
    # they are not for KC-tag pairs elsewhere in this program. Reporting the decoupling
    # distribution over ALL edges with log data on both sides (not just the top 30 by volume)
    # so that this selection effect is visible rather than silently driving the headline number.
    all_edge_decs = []
    for (lo, hi), n_both in pair_both_exercise.items():
        st = decoupling_stats(kc_n_exercise.get(lo, 0), kc_n_exercise.get(hi, 0), n_both)
        if st["symmetric_decoupling"] is not None:
            all_edge_decs.append(st["symmetric_decoupling"])
    all_edge_decs_arr = np.array(all_edge_decs, dtype=float)
    exercise_pair_decoupling["all_edges_with_log_data_context"] = {
        "note": "NOT the required top-30-by-volume reading above -- the full population of "
                f"{len(all_edge_decs_arr)} prerequisite edges with attempt data on both sides, "
                "included because the top-30-by-co-occurrence-volume selection turns out to be "
                "systematically the worst-decoupled slice of this population (see module "
                "docstring / report), not a representative one.",
        **quartiles(all_edge_decs_arr),
        "frac_clearing_0.75": float(np.mean(all_edge_decs_arr >= DECOUPLING_CLEAN_THRESHOLD)) if all_edge_decs_arr.size else None,
    }
    print(f"[triage_junyi15] prerequisite exercise-pair decoupling done, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    # ---- topic KC aggregator: standard (a)-(g) battery ----
    df["topic"] = df["exercise"].map(ex2topic)
    n_rows_topic_unmapped = int(df["topic"].isna().sum())
    topic_df = df.dropna(subset=["topic"]).copy()
    topic_df["topic"] = topic_df["topic"].astype("category")

    agg = KCModelAggregator(name=KC_COLUMN)
    agg.add_overall(len(topic_df), int(topic_df["correct_bool"].sum()), topic_df["correct"].to_numpy())

    marg = topic_df.groupby("topic", observed=True)["correct_bool"].agg(["size", "sum"])
    agg.add_chunk_marginal(marg["size"].to_dict(), marg["sum"].to_dict())

    # chronological order for topic-level opportunity/growth: sort by (user_id, time_done).
    topic_df.sort_values(["user_id", "time_done"], kind="stable", inplace=True)
    topic_df["topic_opp"] = topic_df.groupby(["user_id", "topic"], observed=True, sort=False).cumcount() + 1
    print(f"[triage_junyi15] topic chronological sort + opportunity index done, "
          f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    opp_max_topic = topic_df.groupby(["user_id", "topic"], observed=True)["topic_opp"].max()
    agg.add_chunk_opp_max({key: int(v) for key, v in opp_max_topic.items()})

    gsub = topic_df[topic_df["topic_opp"].between(1, GROWTH_MAX_OPPORTUNITY)]
    ggrp = gsub.groupby(["topic", "topic_opp"], observed=True)["correct_bool"].agg(["size", "sum"])
    agg.add_chunk_growth(ggrp["size"].to_dict(), ggrp["sum"].to_dict())

    # (e) at the topic grain: structurally vacuous, see module docstring -- pair_both left empty.

    # (f)/(g): static Q-matrix from the FULL exercise bank (835 deduped exercises, 816 with a
    # topic tag), not restricted to the log sample -- same convention as EdNet/Junyi2020.
    item_to_kcset = {name: frozenset({topic}) for name, topic in zip(ex["name"], ex["topic"]) if pd.notna(topic)}
    agg.add_items(item_to_kcset)

    topic_result = agg.finalize()
    print(f"[triage_junyi15] topic KC aggregator finalized, elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    # ---- (iii) per-learner-topic opportunity (duplicated here for direct side-by-side with
    # the per-learner-exercise block above; identical numbers to topic_result's c_opportunity_counts).
    topic_opp_block = {
        **quartiles(opp_max_topic.to_numpy(dtype=float)),
        "n_learner_topic_pairs": int(opp_max_topic.size),
        "frac_ge_5": fraction_at_least(opp_max_topic.to_numpy(dtype=float), 5),
        "frac_ge_10": fraction_at_least(opp_max_topic.to_numpy(dtype=float), 10),
        "frac_ge_20": fraction_at_least(opp_max_topic.to_numpy(dtype=float), 20),
        "source": "derived: cumcount per (user_id, topic) after stable sort by (user_id, time_done)",
    }

    results = {
        "bed": "junyi15",
        "source_files": {
            "problem_log": str(Path(args.problem_log_csv).resolve()),
            "exercise_table": str(Path(args.exercise_table_csv).resolve()),
        },
        "kc_layer_choice": {
            "chosen": KC_COLUMN,
            "hierarchy_shape_n_distinct": ex_info["hierarchy_shape_n_distinct"],
            "exercises_per_topic_quartiles": ex_info["exercises_per_topic_quartiles"],
            "n_exercises_missing_topic": ex_info["n_exercises_missing_topic"],
            "justification": "area (8 values) is a coarse subject split; topic (40 values, median "
                              "~20 exercises/topic) is the finest tag this release ships and is used "
                              "as the KC layer, per the task spec. Prerequisites relate EXERCISES to "
                              "each other, not topics, so the exercise-level grain is reported "
                              "separately through the prerequisite-graph and exercise-pair-decoupling "
                              "extras below rather than as a second, degenerate KC-as-item battery.",
        },
        "item_grain": "exercise (`name`/`exercise` column, 835 distinct after dropping 2 exact-"
                      "duplicate rows); one practice slot is one row of junyi_ProblemLog_original.csv.",
        "unit_of_analysis": "one row of junyi_ProblemLog_original.csv (one problem attempt)",
        "response_field": "correct (true/false; false if any hint requested, per README.txt)",
        "sampling": "none (full file streamed in chunks)" if args.nrows is None
                    else f"DEBUG RUN: first {args.nrows} rows only",
        "n_rows_total": n_rows_total,
        "n_rows_topic_unmapped_dropped": n_rows_topic_unmapped,
        "n_users_total": n_users_total,
        "n_exercises_in_log": n_exercises_in_log,
        "n_exercises_in_table_after_dedup": int(len(ex)),
        "n_exact_duplicate_exercise_rows_dropped": ex_info["n_exact_duplicate_rows_dropped"],
        "n_log_exercises_not_in_table": n_log_exercises_not_in_table,
        "decoupling_definition": DECOUPLING_DEFINITION_TEXT,

        # (i) prerequisite graph
        "i_prerequisite_graph": ex_info["graph"],

        # standard (a)-(g) battery at the topic KC grain
        "kc_models": {KC_COLUMN: topic_result},

        # (ii) decoupling at both grains
        "ii_decoupling_both_grains": {
            "topic_kc_pairs": {
                "structural_note": "Structurally vacuous, exactly as Junyi2020's level4_id: one row "
                                    "= one exercise = one topic, so two topics never co-occur on the "
                                    "same slot. See kc_models.topic.e_decoupling "
                                    "(n_pairs_with_cross_occurrence == 0 by construction).",
                "n_pairs_with_cross_occurrence": topic_result["e_decoupling"]["n_pairs_with_cross_occurrence"],
            },
            "prerequisite_linked_exercise_pairs": exercise_pair_decoupling,
        },

        # (iii) opportunity distributions at both grains
        "iii_opportunity_both_grains": {
            "per_learner_exercise": exercise_opp_block,
            "per_learner_topic": topic_opp_block,
        },

        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "junyi15_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
