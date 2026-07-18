"""Shared helpers for kt-mirt STAGE-0 bed triage scripts.

STAGE-0 bed triage computes, from raw local files, the statistics every
downstream bed-selection choice in the kt-mirt program depends on
(saturation, growth headroom, KC-pair decoupling, pure-anchor coverage).
These numbers are not available in the literature for these specific
local extracts -- everything under kt-mirt/_planning/triage/*.json is
computed directly from the raw files, not estimated.

Each per-bed script (triage_kdd.py, triage_ednet.py, triage_slam.py,
triage_timss.py) streams its raw file(s) and feeds running counters into
a KCModelAggregator, then calls .finalize() to produce the (a)-(g) block
defined in the triage spec. This module holds the counter class and the
pure statistics functions shared across beds, so the per-bed scripts stay
short and only contain bed-specific parsing.

DECOUPLING METRIC -- exact definition
======================================
Adapted from the qmirt-archaeology "positivity condition"
(kt-mirt/_planning/research/qmirt-archaeology.md, section 2): in the
qmirt synthetic cross-concept-transfer generator, a source-concept ->
target-concept effect is identifiable only where the practice schedule
"decouples" the two concepts -- episodes where the source is practiced
without the target. That program quantified this as the fraction of
practice slots that are "source-only", reading >=0.75 as a clean
identification regime, 0.25-0.50 as weak/suggestive, and 0 as fully
collinear/unidentifiable (co-scheduled practice makes own-gain and
cross-concept gain collinear).

Real KC-tagged interaction logs (KDD Cup, EdNet) do not carry a labeled
source->target causal edge the way the synthetic generator did -- a KC
pair is just two skills that happen to co-occur on some items/steps. This
triage adapts the definition to be well-posed for an *unordered* KC pair
(A, B) that co-occurs at least once in the data:

  - A "practice slot" is one interaction event (one KDD step / one EdNet
    question attempt) whose tagged KC set includes A and/or B.
  - n_A, n_B  = number of slots tagging A (resp. B), regardless of the other.
  - n_both    = number of slots tagging BOTH A and B simultaneously (a
                single multi-KC-tagged interaction) -- the real-data
                analogue of "co-scheduled practice" in the qmirt generator.
  - n_A_only = n_A - n_both ; n_B_only = n_B - n_both.

  PRIMARY (symmetric) decoupling fraction -- reported as "the" decoupling
  number for a pair, and the one gated at >=0.75:

      decoupling(A,B) = (n_A_only + n_B_only) / (n_A_only + n_B_only + n_both)

  i.e. of all slots that touch the pair, the fraction that touch EXACTLY
  ONE of the two concepts. This is the direct real-data analogue of "the
  fraction of source-only practice slots": read A as the source, and
  n_A_only / n_A is exactly the qmirt quantity for that direction. The
  symmetric form reports that fraction generalized over both possible
  source assignments, since real KC-pair data has no privileged
  direction; the two one-directional readings (dir_a_source_only_frac,
  dir_b_source_only_frac) are reported alongside as supplementary detail
  for whichever direction a downstream user cares about.

  decoupling -> 0 as n_both -> min(n_A, n_B) (fully co-scheduled)
  reproduces the qmirt "completely unidentifiable" case. decoupling = 1
  would mean the pair never co-occurs, which cannot happen here since we
  only report pairs with n_both > 0 ("KC pairs WITH cross-occurrence").
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants (the thresholds named explicitly in the triage spec)
# ---------------------------------------------------------------------------
SATURATION_THRESHOLD = 0.85
OPPORTUNITY_THRESHOLDS = (5, 10, 20)
DECOUPLING_CLEAN_THRESHOLD = 0.75
GROWTH_MAX_OPPORTUNITY = 10
TOP_K_PAIRS = 30
MIN_PURE_ANCHORS = 3

DECOUPLING_DEFINITION_TEXT = (
    "decoupling(A,B) = (n_A_only + n_B_only) / (n_A_only + n_B_only + n_both), "
    "where a 'slot' is one interaction event whose tagged KC set includes A "
    "and/or B, n_both = slots tagging both simultaneously (co-scheduled "
    "practice), n_A_only/n_B_only = slots tagging exactly one. This is the "
    "real-data, direction-symmetric analogue of the qmirt-archaeology "
    "positivity condition ('fraction of source-only practice slots', "
    ">=0.75 clean / 0 unidentifiable) -- see triage_common.py module "
    "docstring for the full derivation and the two directional readings "
    "reported alongside it."
)


# ---------------------------------------------------------------------------
# Pure statistics helpers
# ---------------------------------------------------------------------------
def quartiles(values: Iterable[float]) -> dict:
    """n / min / q1 / median / q3 / max / mean for a 1D array-like."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None, "mean": None}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def fraction_above(values: Iterable[float], threshold: float) -> Optional[float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr > threshold))


def fraction_at_least(values: Iterable[float], threshold: float) -> Optional[float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr >= threshold))


def decoupling_stats(n_a: int, n_b: int, n_both: int) -> dict:
    """KC-pair decoupling metric. See module docstring for the definition."""
    n_a_only = n_a - n_both
    n_b_only = n_b - n_both
    union = n_a_only + n_b_only + n_both
    symmetric = (n_a_only + n_b_only) / union if union > 0 else None
    dir_a = n_a_only / n_a if n_a > 0 else None
    dir_b = n_b_only / n_b if n_b > 0 else None
    return {
        "n_a": int(n_a), "n_b": int(n_b), "n_both": int(n_both),
        "n_a_only": int(n_a_only), "n_b_only": int(n_b_only),
        "symmetric_decoupling": symmetric,
        "dir_a_source_only_frac": dir_a,
        "dir_b_source_only_frac": dir_b,
    }


def top_pairs_report(pair_both: Counter, kc_n: Counter, k: int = TOP_K_PAIRS,
                      clean_threshold: float = DECOUPLING_CLEAN_THRESHOLD) -> dict:
    """Top-k KC pairs by co-occurrence volume (n_both), with decoupling stats.

    Ranking: pairs are ranked by n_both (number of slots where the pair is
    co-scheduled), restricted to pairs with n_both > 0 ("cross-occurrence"),
    since pair_both only ever holds keys observed together at least once.
    """
    ranked = pair_both.most_common(k)
    pairs = []
    for (a, b), n_both in ranked:
        stats = decoupling_stats(kc_n.get(a, 0), kc_n.get(b, 0), n_both)
        stats["kc_a"] = a
        stats["kc_b"] = b
        pairs.append(stats)
    cleared = [p for p in pairs
               if p["symmetric_decoupling"] is not None and p["symmetric_decoupling"] >= clean_threshold]
    return {
        "definition": DECOUPLING_DEFINITION_TEXT,
        "ranking": "top pairs by n_both (co-occurring practice slots), restricted to n_both > 0",
        "clean_threshold": clean_threshold,
        "n_pairs_with_cross_occurrence": len(pair_both),
        "n_pairs_reported": len(pairs),
        "frac_reported_pairs_clearing_threshold": (len(cleared) / len(pairs)) if pairs else None,
        "pairs": pairs,
    }


def pure_anchor_and_arity_stats(item_to_kcset: Dict[str, frozenset], min_pure: int = MIN_PURE_ANCHORS) -> dict:
    """Item-KC arity and Gate-C-style pure-anchor coverage from a static item bank.

    item_to_kcset maps an item/step identifier to the set of KCs it is
    tagged with (its Q-matrix row). A "pure" item maps to exactly one KC.
    """
    nonempty = [s for s in item_to_kcset.values() if len(s) > 0]
    arities = np.array([len(s) for s in nonempty], dtype=float)
    pure_items_per_kc: Counter = Counter()
    kc_universe = set()
    for kcset in nonempty:
        kc_universe.update(kcset)
        if len(kcset) == 1:
            pure_items_per_kc[next(iter(kcset))] += 1
    n_kc = len(kc_universe)
    n_kc_with_ge_min = sum(1 for kc in kc_universe if pure_items_per_kc.get(kc, 0) >= min_pure)
    return {
        "n_items": int(len(nonempty)),
        "n_kc_in_item_bank": int(n_kc),
        "arity_mean": float(arities.mean()) if arities.size else None,
        "arity_max": int(arities.max()) if arities.size else None,
        "n_pure_items_total": int(sum(1 for s in nonempty if len(s) == 1)),
        "min_pure_items_gate": min_pure,
        "frac_kc_with_ge_min_pure_items": (n_kc_with_ge_min / n_kc) if n_kc else None,
    }


def pooled_growth_curve(growth_n: Counter, growth_correct: Counter, max_o: int = GROWTH_MAX_OPPORTUNITY) -> dict:
    """Pooled (all-KC, count-weighted) first-attempt correct rate at opportunity 1..max_o."""
    n_by_o: Counter = Counter()
    correct_by_o: Counter = Counter()
    for (kc, o), n in growth_n.items():
        n_by_o[o] += n
    for (kc, o), c in growth_correct.items():
        correct_by_o[o] += c
    curve = {}
    for o in range(1, max_o + 1):
        n = n_by_o.get(o, 0)
        curve[str(o)] = {"n": int(n), "correct_rate": (correct_by_o.get(o, 0) / n) if n > 0 else None}
    return curve


def per_kc_slopes(growth_n: Counter, growth_correct: Counter, max_o: int = GROWTH_MAX_OPPORTUNITY,
                   min_points: int = 3, min_n_per_point: int = 5) -> dict:
    """Median/sign of a per-KC OLS slope of first-attempt correct rate vs opportunity index.

    A KC is included only if it has at least `min_points` distinct
    opportunity indices (within 1..max_o) each backed by at least
    `min_n_per_point` observations, so the slope is not a fit to noise on
    tiny cells. This is the "median per-KC slope sign / rough magnitude"
    read requested for the descriptive, model-free growth signal.
    """
    by_kc: Dict[str, Dict[int, int]] = defaultdict(dict)
    for (kc, o), n in growth_n.items():
        if n >= min_n_per_point:
            by_kc[kc][o] = n
    slopes = {}
    for kc, opps in by_kc.items():
        xs = sorted(opps.keys())
        if len(xs) < min_points:
            continue
        x = np.array(xs, dtype=float)
        y = np.array([growth_correct[(kc, o)] / growth_n[(kc, o)] for o in xs], dtype=float)
        if np.all(x == x[0]):
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        slopes[kc] = slope
    if not slopes:
        return {
            "n_kc_with_slope": 0, "median_slope": None, "mean_slope": None,
            "frac_positive": None, "frac_negative": None, "frac_zero": None,
            "min_points_required": min_points, "min_n_per_opportunity_point": min_n_per_point,
        }
    vals = np.array(list(slopes.values()))
    return {
        "n_kc_with_slope": int(vals.size),
        "median_slope": float(np.median(vals)),
        "mean_slope": float(np.mean(vals)),
        "frac_positive": float(np.mean(vals > 0)),
        "frac_negative": float(np.mean(vals < 0)),
        "frac_zero": float(np.mean(vals == 0)),
        "min_points_required": min_points,
        "min_n_per_opportunity_point": min_n_per_point,
    }


def pair_counts_from_kc_lists(kc_list_series: "pd.Series") -> dict:
    """Co-occurrence counts for KC/tag pairs tagged together on the same slot.

    `kc_list_series` holds, per practice slot (one step / one question
    attempt), the (pre-explode) list of KC/tag strings for that slot.
    Vectorized for the common 2-KC case; a small Python loop handles the
    rare >=3-KC case (all C(k,2) pairs for that slot). Shared by the KDD
    (~~-separated KC cells) and EdNet (;-separated tag cells) triage
    scripts -- both reduce to "a slot has a list of KC/tag strings" before
    calling this.
    """
    counts: dict = {}
    lengths = kc_list_series.str.len()

    two = kc_list_series[lengths == 2]
    if len(two):
        arr = np.array(two.tolist())
        a, b = arr[:, 0], arr[:, 1]
        lo = np.where(a < b, a, b)
        hi = np.where(a < b, b, a)
        pair_df = pd.DataFrame({"lo": lo, "hi": hi})
        for (lo_v, hi_v), cnt in pair_df.groupby(["lo", "hi"]).size().items():
            key = (lo_v, hi_v)
            counts[key] = counts.get(key, 0) + int(cnt)

    many = kc_list_series[lengths >= 3]
    for kl in many:
        uniq = sorted(set(kl))
        for a, b in itertools.combinations(uniq, 2):
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1

    return counts


def category_distribution(counter: Counter) -> dict:
    total = sum(counter.values())
    return {
        str(k): {"n": int(v), "frac": (v / total) if total else None}
        for k, v in counter.most_common()
    }


def overall_block(n: int, n_correct: int, response_categories: Optional[Counter] = None) -> dict:
    rate = (n_correct / n) if n else None
    block = {"n": int(n), "n_correct": int(n_correct), "correct_rate": rate}
    if response_categories:
        block["response_category_distribution"] = category_distribution(response_categories)
    return block


# ---------------------------------------------------------------------------
# Streaming aggregator
# ---------------------------------------------------------------------------
@dataclass
class KCModelAggregator:
    """Running counters for one KC-tagging model (or one bed, for beds with
    a single tagging scheme like EdNet's `tags`). Bed scripts feed chunk-level
    aggregates in via the add_chunk_* methods (never the raw per-row stream,
    to keep memory bounded); .finalize() derives the (a)-(g) triage block.
    """
    name: str
    kc_n: Counter = field(default_factory=Counter)
    kc_correct: Counter = field(default_factory=Counter)
    pair_both: Counter = field(default_factory=Counter)
    opp_max: Dict = field(default_factory=dict)          # (student, kc) -> max opportunity seen
    growth_n: Counter = field(default_factory=Counter)    # (kc, opportunity) -> n
    growth_correct: Counter = field(default_factory=Counter)
    item_to_kcset: Dict[str, frozenset] = field(default_factory=dict)
    overall_n: int = 0
    overall_correct: int = 0
    response_categories: Counter = field(default_factory=Counter)

    def add_chunk_marginal(self, kc_counts: dict, kc_correct_counts: dict) -> None:
        for kc, n in kc_counts.items():
            self.kc_n[kc] += n
        for kc, c in kc_correct_counts.items():
            self.kc_correct[kc] += c

    def add_chunk_pairs(self, pair_counts: dict) -> None:
        for pair, n in pair_counts.items():
            self.pair_both[pair] += n

    def add_chunk_opp_max(self, opp_max_chunk: dict) -> None:
        opp_max = self.opp_max
        for key, v in opp_max_chunk.items():
            prev = opp_max.get(key, 0)
            if v > prev:
                opp_max[key] = v

    def add_chunk_growth(self, growth_n_chunk: dict, growth_correct_chunk: dict) -> None:
        for key, n in growth_n_chunk.items():
            self.growth_n[key] += n
        for key, c in growth_correct_chunk.items():
            self.growth_correct[key] += c

    def add_items(self, item_kc_map: Dict[str, frozenset]) -> None:
        # Q-matrix is a static property of the item/step; last-write-wins
        # (occurrences of the same item are expected to carry the same tag).
        self.item_to_kcset.update(item_kc_map)

    def add_overall(self, n: int, n_correct: int, response_values: Optional[Iterable] = None) -> None:
        self.overall_n += n
        self.overall_correct += n_correct
        if response_values is not None:
            self.response_categories.update(Counter(str(v) for v in response_values))

    def finalize(self, saturation_threshold: float = SATURATION_THRESHOLD,
                 opportunity_thresholds=OPPORTUNITY_THRESHOLDS,
                 top_k: int = TOP_K_PAIRS, min_pure: int = MIN_PURE_ANCHORS) -> dict:
        per_kc_rates = np.array(
            [self.kc_correct[kc] / self.kc_n[kc] for kc in self.kc_n if self.kc_n[kc] > 0],
            dtype=float,
        )
        opp_values = np.array(list(self.opp_max.values()), dtype=float)
        result = {
            "kc_model": self.name,
            "n_distinct_kc": int(len(self.kc_n)),
            "a_overall": overall_block(self.overall_n, self.overall_correct, self.response_categories),
            "b_per_kc_correct_rate": {
                **quartiles(per_kc_rates),
                "frac_kc_above_0.85": fraction_above(per_kc_rates, saturation_threshold),
            },
            "c_opportunity_counts": {
                **quartiles(opp_values),
                "n_learner_kc_pairs": int(opp_values.size),
                **{f"frac_ge_{t}": fraction_at_least(opp_values, t) for t in opportunity_thresholds},
            },
            "d_growth_signal": {
                "pooled_curve_opportunity_1_to_10": pooled_growth_curve(self.growth_n, self.growth_correct),
                "per_kc_slopes": per_kc_slopes(self.growth_n, self.growth_correct),
            },
            "e_decoupling": top_pairs_report(self.pair_both, self.kc_n, k=top_k),
        }
        if self.item_to_kcset:
            result["f_g_pure_anchors_and_arity"] = pure_anchor_and_arity_stats(self.item_to_kcset, min_pure=min_pure)
        else:
            result["f_g_pure_anchors_and_arity"] = None
        return result


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, frozenset):
            return sorted(o)
        return super().default(o)


def write_json(path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=NumpyJSONEncoder, sort_keys=False)
    print(f"[triage] wrote {path}")
