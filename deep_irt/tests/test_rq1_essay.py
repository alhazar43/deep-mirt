"""Tests for the RQ1 essay shared-scale pilot (deep_irt/rq1_essay).

Network-free and Ollama-free: uses handcrafted features and synthetic
comparisons so the model + metric machinery is covered in CI without the
local LLM or sentence-transformers download.

Test IDs
--------
1. test_sign_convention_aligned -- with ORACLE comparisons (winner = higher
   human score), the joint model's quality CORRELATES POSITIVELY with the
   human score on pairwise-only essays. This guards the sign-convention fix:
   both heads must agree that higher d = higher quality, else the joint
   transfer sign-flips on the minority slice.
2. test_trap_detector_punishes_noise -- with RANDOM comparisons, the
   genuine-transfer gap (joint minus content-only on pairwise-only essays) is
   NOT positive. The metric must not manufacture transfer from noise.
3. test_graded_head_monotone -- the graded head maps higher d to higher
   expected score (the ordinal direction is correct).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

_RQ1 = Path(__file__).resolve().parents[1] / "rq1_essay"
if str(_RQ1) not in sys.path:
    sys.path.insert(0, str(_RQ1))

ASAP = Path("rl/data/asap/training_set_rel3.tsv")
pytestmark = pytest.mark.skipif(
    not ASAP.exists(), reason="ASAP TSV not present"
)


def _build(seed=0, n=60):
    from essay_data import build_split
    # prefer_st=False -> handcrafted features, no download
    return build_split(essay_set=1, n_essays=n, seed=seed,
                       prefer_st=False, max_chars=4000)


def _oracle_pairs(split, noise=0.15, n_pairs=300, seed=1):
    rng = np.random.default_rng(seed)
    pe = split.pairwise_idx
    pairs, seen = [], set()
    guard = 0
    while len(pairs) < n_pairs and guard < n_pairs * 60:
        guard += 1
        i, j = rng.choice(pe, size=2, replace=False)
        i, j = int(i), int(j)
        k = (min(i, j), max(i, j))
        if k in seen:
            continue
        seen.add(k)
        si, sj = split.raw_scores[i], split.raw_scores[j]
        if si == sj:
            o = int(rng.integers(0, 2))
        else:
            true = 1 if si > sj else 0
            o = true if rng.random() > noise else 1 - true
        pairs.append((i, j, o))
    return pairs


def _random_pairs(split, n_pairs=300, seed=2):
    rng = np.random.default_rng(seed)
    pe = split.pairwise_idx
    pairs, seen = [], set()
    guard = 0
    while len(pairs) < n_pairs and guard < n_pairs * 60:
        guard += 1
        i, j = rng.choice(pe, size=2, replace=False)
        i, j = int(i), int(j)
        k = (min(i, j), max(i, j))
        if k in seen:
            continue
        seen.add(k)
        pairs.append((i, j, int(rng.integers(0, 2))))
    return pairs


def test_sign_convention_aligned():
    from essay_train import train_joint
    from essay_metrics import compute_metrics
    torch.manual_seed(0)
    split = _build(seed=0)
    pairs = _oracle_pairs(split)
    joint = train_joint(split, pairs, n_epochs=400, seed=0, verbose=False)
    # Need indep readouts only to call compute_metrics; reuse joint quality for
    # the slices we assert on.
    q = joint.get_quality().numpy()
    m = compute_metrics(split, pairs, q, q, q)
    # The joint quality must POSITIVELY track human score on pairwise-only
    # essays (whose human score was held out). align_sign is global; the model
    # being internally sign-consistent is what makes this hold on the slice.
    assert m["joint_on_pairwise_only"] > 0.3, m["joint_on_pairwise_only"]
    assert m["joint_overall"] > 0.5, m["joint_overall"]


def test_trap_detector_punishes_noise():
    from essay_train import train_joint, train_indep_graded, train_indep_bt
    from essay_metrics import compute_metrics
    torch.manual_seed(0)
    split = _build(seed=0)
    pairs = _random_pairs(split)
    joint = train_joint(split, pairs, n_epochs=400, seed=0, verbose=False)
    ig = train_indep_graded(split, n_epochs=400, seed=0, verbose=False)
    ib = train_indep_bt(split, pairs, n_epochs=400, seed=0, verbose=False)
    m = compute_metrics(split, pairs,
                        joint.get_quality().numpy(),
                        ig.get_quality().numpy(),
                        ib.get_quality().numpy())
    # Random comparisons must not yield a positive genuine-transfer gap.
    assert m["pairwise_only_gap_vs_content"] <= 0.10, \
        m["pairwise_only_gap_vs_content"]


def test_graded_head_monotone():
    from essay_model import GradedHead
    head = GradedHead(K=5)
    d = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    lp = head.log_probs(d)            # (5, 5)
    probs = lp.exp()
    cats = torch.arange(5, dtype=torch.float32)
    exp_score = (probs * cats).sum(dim=-1)   # expected category per d
    # Expected score must be non-decreasing in d.
    assert torch.all(exp_score[1:] - exp_score[:-1] > -1e-4), exp_score
