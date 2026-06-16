"""Network-free tests for the RQ1 judge pre-gate (deep_irt/rq1_essay/run_pregate.py).

Covers the two pieces that must be correct independent of Ollama:
  1. sample_labeled_pairs -- every sampled pair is non-tied with a human-score
     gap >= gap_min, the winner is the higher-scored essay, pairs are unique and
     deterministic given a seed, and the sampler spans the score range.
  2. compute_agreement -- fraction of DECIDED pairs where judge == human, with
     undecided (None) pairs counted as unparsed and excluded from the denominator.

Both use synthetic rows, so no ASAP TSV and no local LLM are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_RQ1 = Path(__file__).resolve().parents[1] / "rq1_essay"
if str(_RQ1) not in sys.path:
    sys.path.insert(0, str(_RQ1))

from run_pregate import sample_labeled_pairs, compute_agreement  # noqa: E402


def _synthetic_rows(n_per_score=8, lo=2, hi=12):
    """Build rows spanning the full set-1 score range, unique ascending ids."""
    rows = []
    eid = 1000
    for s in range(lo, hi + 1):
        for _ in range(n_per_score):
            rows.append({"essay_id": eid, "essay": f"essay {eid} score {s}",
                         "score": float(s)})
            eid += 1
    return rows


def test_pairs_are_non_tied_with_min_gap():
    rows = _synthetic_rows()
    pairs = sample_labeled_pairs(rows, n_pairs=50, seed=0, gap_min=2)
    assert pairs, "expected a non-empty sample"
    for p in pairs:
        gap = abs(p["score_a"] - p["score_b"])
        assert gap >= 2, f"gap {gap} below gap_min for pair {p}"
        assert p["score_a"] != p["score_b"], "tied pair leaked in"


def test_winner_is_higher_scored_essay():
    rows = _synthetic_rows()
    score_by_id = {r["essay_id"]: r["score"] for r in rows}
    pairs = sample_labeled_pairs(rows, n_pairs=50, seed=1, gap_min=2)
    for p in pairs:
        sa, sb = score_by_id[p["a"]], score_by_id[p["b"]]
        expected = p["a"] if sa > sb else p["b"]
        assert p["winner"] == expected, (
            f"winner {p['winner']} is not the higher-scored essay "
            f"(a={p['a']}:{sa}, b={p['b']}:{sb})")


def test_pairs_unique_and_deterministic():
    rows = _synthetic_rows()
    p1 = sample_labeled_pairs(rows, n_pairs=40, seed=7, gap_min=2)
    p2 = sample_labeled_pairs(rows, n_pairs=40, seed=7, gap_min=2)
    key = lambda lst: [(p["a"], p["b"], p["winner"]) for p in lst]
    assert key(p1) == key(p2), "same seed must reproduce the same pairs"
    keys = {(p["a"], p["b"]) for p in p1}
    assert len(keys) == len(p1), "pairs must be unique (unordered)"
    # different seed should generally differ
    p3 = sample_labeled_pairs(rows, n_pairs=40, seed=8, gap_min=2)
    assert key(p1) != key(p3), "different seed should change the sample"


def test_sampler_spans_score_range():
    rows = _synthetic_rows()
    pairs = sample_labeled_pairs(rows, n_pairs=80, seed=0, gap_min=2)
    lows = {int(min(p["score_a"], p["score_b"])) for p in pairs}
    # stratified-by-lower-score buckets should cover several distinct bands
    assert len(lows) >= 4, f"expected spread across buckets, got {sorted(lows)}"


def test_gap_min_too_large_yields_empty():
    rows = _synthetic_rows(lo=2, hi=12)
    # gap_min beyond the achievable range -> no valid pairs, no infinite loop
    pairs = sample_labeled_pairs(rows, n_pairs=10, seed=0, gap_min=99)
    assert pairs == []


def test_compute_agreement_basic():
    # human winners: 1,1,2,2 ; judge: right, right, wrong, undecided(None)
    decisions = [(1, 1), (1, 1), (2, 1), (2, None)]
    agreement, n, unparsed = compute_agreement(decisions)
    assert n == 3, "None pair excluded from denominator"
    assert unparsed == 1
    assert agreement == pytest.approx(2 / 3)


def test_compute_agreement_all_undecided_is_nan():
    decisions = [(1, None), (2, None)]
    agreement, n, unparsed = compute_agreement(decisions)
    assert n == 0
    assert unparsed == 2
    assert agreement != agreement  # NaN


def test_compute_agreement_perfect_and_zero():
    perfect = [(1, 1), (2, 2), (3, 3)]
    agr, n, unp = compute_agreement(perfect)
    assert agr == 1.0 and n == 3 and unp == 0
    zero = [(1, 2), (2, 1)]
    agr, n, unp = compute_agreement(zero)
    assert agr == 0.0 and n == 2 and unp == 0
