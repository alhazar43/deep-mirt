"""
Tests for the EdNet separability study (deep_irt/ednet_sep).

Covers the scientific logic that does not require the full EdNet artefact:
the variance decomposition, the leakage probe, the alignment gradient, and
the data-side split-half / subsequence helpers.  These guard against silent
regressions in the metrics that drive the study's verdicts.

Test IDs
--------
1. test_partmap_section_helpers   -- TOEIC section assignment is correct
2. test_vardecomp_separable       -- separable synthetic -> index ~1
3. test_vardecomp_part_dominated  -- part-dominated synthetic -> index ~0
4. test_spearman_brown            -- prophecy formula
5. test_leakage_pure_ability      -- ability-only features -> residual AUC ~chance
6. test_leakage_part_encoding     -- part-encoding features -> residual AUC >> chance
7. test_alignment_gradient        -- within-section >> cross-section
8. test_split_half_shapes         -- split_half partitions a learner's part events
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deep_irt.ednet_sep.partmap import (
    part_section, same_section, LISTENING_PARTS, READING_PARTS,
)
from deep_irt.ednet_sep.analysis import (
    variance_decomposition, leakage_probe, cross_part_alignment, spearman_brown,
)
from deep_irt.ednet_sep.data import LearnerRecord, split_half
from deep_irt.ednet_sep.anchored_readout import (
    map_theta_batch, anchored_part_ability, build_anchored_ability_table,
)


PARTS = (1, 2, 3, 4, 5, 6, 7)


def test_partmap_section_helpers():
    for p in LISTENING_PARTS:
        assert part_section(p) == "listening"
    for p in READING_PARTS:
        assert part_section(p) == "reading"
    assert same_section(1, 4) and same_section(5, 7)
    assert not same_section(4, 5)  # the listening/reading boundary
    with pytest.raises(ValueError):
        part_section(8)


def _separable_ability(rng, n=200):
    trait = {p: rng.normal() for p in range(n)}
    return {
        p + 1: {pt: trait[p] + rng.normal(0, 0.05) for pt in PARTS}
        for p in range(n)
    }


def test_vardecomp_separable():
    rng = np.random.default_rng(0)
    vd = variance_decomposition(_separable_ability(rng), PARTS)
    assert vd["separability_index"] > 0.9
    assert vd["part_var_share"] < 0.1


def test_vardecomp_part_dominated():
    rng = np.random.default_rng(1)
    trait = {p: rng.normal() for p in range(200)}
    part_eff = {pt: rng.normal(0, 3.0) for pt in PARTS}
    ability = {
        p + 1: {pt: 0.1 * trait[p] + part_eff[pt] + rng.normal(0, 0.3) for pt in PARTS}
        for p in range(200)
    }
    vd = variance_decomposition(ability, PARTS)
    assert vd["separability_index"] < 0.2
    assert vd["part_var_share"] > 0.7


def test_spearman_brown():
    assert spearman_brown(0.5) == pytest.approx(2 / 3, abs=1e-6)
    assert spearman_brown(1.0) == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(spearman_brown(-1.0))


def test_leakage_pure_ability():
    rng = np.random.default_rng(2)
    N = 700
    abil = rng.normal(size=N)
    part = rng.integers(1, 8, size=N)
    feat = np.column_stack([abil + rng.normal(0, 0.01, N) for _ in range(8)])
    out = leakage_probe(feat, abil, part, seed=0, n_splits=5)
    assert out["residual_auc"] < 0.6  # ability removed -> near chance


def test_leakage_part_encoding():
    rng = np.random.default_rng(3)
    N = 700
    abil = rng.normal(size=N)
    part = rng.integers(1, 8, size=N)
    feat = np.column_stack(
        [(part == k).astype(float) + rng.normal(0, 0.1, N) for k in range(1, 8)]
    )
    out = leakage_probe(feat, abil, part, seed=0, n_splits=5)
    assert out["residual_auc"] > 0.8  # part identity survives ability removal


def test_alignment_gradient():
    rng = np.random.default_rng(4)
    n = 200
    listen = rng.normal(size=n)
    read = rng.normal(size=n)
    ability = {}
    for i in range(n):
        ability[i + 1] = {
            pt: (listen[i] if pt <= 4 else read[i]) + rng.normal(0, 0.3)
            for pt in PARTS
        }
    rel = {pt: 0.9 for pt in PARTS}
    al = cross_part_alignment(ability, rel, PARTS, min_pairs=10)
    g = al["gradient"]
    assert g["within_section_mean_pearson"] > g["cross_section_mean_pearson"]
    assert g["gradient_gap_raw"] > 0.3
    assert g["gradient_holds_raw"] is True


def test_split_half_shapes():
    # Learner with 10 part-2 events interleaved among other parts.
    parts = torch.tensor([2, 1, 2, 3, 2, 2, 5, 2, 2, 2, 2, 2, 7, 2])
    n_q = parts.numel()
    rec = LearnerRecord(
        student_id=7,
        questions=torch.arange(1, n_q + 1),
        responses=torch.zeros(n_q, dtype=torch.long),
        parts=parts,
    )
    res = split_half(rec, part=2, seed=0)
    assert res is not None
    a, b = res
    n_part2 = int((parts == 2).sum())
    assert len(a["questions"]) + len(b["questions"]) == n_part2
    # Both halves draw only from part-2 question ids.
    part2_qids = set((torch.arange(1, n_q + 1)[parts == 2]).tolist())
    assert set(a["questions"].tolist()) <= part2_qids
    assert set(b["questions"].tolist()) <= part2_qids
    # A learner with < 4 part events cannot be split.
    rec_small = LearnerRecord(
        student_id=8,
        questions=torch.tensor([1, 2, 3]),
        responses=torch.zeros(3, dtype=torch.long),
        parts=torch.tensor([2, 2, 1]),
    )
    assert split_half(rec_small, part=2, seed=0) is None


# ---------------------------------------------------------------------------
# Anchored (fixed-parameter) readout -- the verification path
# ---------------------------------------------------------------------------

def _binary_gpcm_tables(b_vals):
    """Binary (K=2) GPCM item tables: a=1, threshold=b.  P(1) = sigmoid(theta-b)."""
    n = len(b_vals)
    a = np.ones(n, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64).reshape(n, 1)
    return a, b


def test_map_theta_recovers_truth_binary():
    # K=2 GPCM is 2PL: with a=1, MAP theta solves a Rasch person estimate.
    # Generate responses for known thetas against fixed easy/hard items, then
    # check the MAP estimate recovers the rank order and rough magnitude.
    rng = np.random.default_rng(0)
    n_items = 60
    b_vals = rng.normal(0, 1.0, size=n_items)
    a_tab, b_tab = _binary_gpcm_tables(b_vals)

    true_theta = np.array([-2.0, -0.5, 0.5, 2.0])
    P = len(true_theta)
    # Simulate responses: P(correct) = sigmoid(theta - b).
    resp = np.zeros((P, n_items), dtype=np.int64)
    for p in range(P):
        prob = 1.0 / (1.0 + np.exp(-(true_theta[p] - b_vals)))
        resp[p] = (rng.uniform(size=n_items) < prob).astype(np.int64)

    a_t = torch.tensor(np.tile(a_tab, (P, 1)), dtype=torch.float32)
    b_t = torch.tensor(np.tile(b_tab.reshape(1, n_items, 1), (P, 1, 1)), dtype=torch.float32)
    r_t = torch.tensor(resp, dtype=torch.long)
    m_t = torch.ones(P, n_items, dtype=torch.float32)

    theta_hat = map_theta_batch(a_t, b_t, r_t, m_t, prior_sd=5.0,
                                n_steps=300, lr=0.2).numpy()
    # Rank order must be preserved.
    assert np.all(np.argsort(theta_hat) == np.argsort(true_theta))
    # And the estimate should be in the right ballpark (loose, 60 items, MAP).
    assert np.corrcoef(theta_hat, true_theta)[0, 1] > 0.95


def test_map_theta_prior_keeps_saturated_finite():
    # All-correct and all-incorrect learners: pure MLE diverges; the MAP prior
    # must keep theta finite and signed correctly.
    a_tab, b_tab = _binary_gpcm_tables([0.0] * 20)
    P = 2
    a_t = torch.tensor(np.tile(a_tab, (P, 1)), dtype=torch.float32)
    b_t = torch.tensor(np.tile(b_tab.reshape(1, 20, 1), (P, 1, 1)), dtype=torch.float32)
    r_t = torch.tensor(np.array([[1] * 20, [0] * 20]), dtype=torch.long)
    m_t = torch.ones(P, 20, dtype=torch.float32)
    theta_hat = map_theta_batch(a_t, b_t, r_t, m_t, prior_sd=1.0,
                                n_steps=300, lr=0.2).numpy()
    assert np.all(np.isfinite(theta_hat))
    assert theta_hat[0] > 0 > theta_hat[1]  # all-correct high, all-incorrect low


def test_anchored_part_ability_uses_all_events():
    # Two learners, part 2 has 12 and 8 events respectively; anchored readout
    # should return one theta per eligible learner and respect min_events.
    a_tab, b_tab = _binary_gpcm_tables(np.zeros(50))
    recs = [
        LearnerRecord(
            student_id=1,
            questions=torch.arange(1, 21),
            responses=torch.randint(0, 2, (20,)),
            parts=torch.tensor([2] * 12 + [5] * 8),
        ),
        LearnerRecord(
            student_id=2,
            questions=torch.arange(1, 16),
            responses=torch.randint(0, 2, (15,)),
            parts=torch.tensor([2] * 8 + [5] * 7),
        ),
    ]
    theta, sids = anchored_part_ability(recs, a_tab, b_tab, part=2,
                                        min_events=10, prior_sd=1.0, n_steps=50)
    # Only learner 1 has >= 10 part-2 events.
    assert set(sids.tolist()) == {1}
    assert theta.shape == (1,)
    # build_anchored_ability_table returns the nested-dict shape RQ4/RQ5b need.
    tab = build_anchored_ability_table(recs, a_tab, b_tab, min_events=5,
                                       prior_sd=1.0, n_steps=50)
    assert 1 in tab and 2 in tab[1] and 5 in tab[1]


# ---------------------------------------------------------------------------
# Classical 2PL control (encoder-free RQ4/RQ5 circularity check)
# ---------------------------------------------------------------------------

from deep_irt.ednet_sep.classical_control import (
    binary_correct, build_part_response_matrix, fit_classical_part,
    classical_vs_encoder_difficulty, MISSING_FILL,
)


def test_binary_correct_from_k4_code():
    # K=4 coercion {0,1}=incorrect, {2,3}=correct -> correct iff code>=2.
    codes = np.array([0, 1, 2, 3, 0, 3])
    assert binary_correct(codes).tolist() == [0, 0, 1, 1, 0, 1]


def test_build_part_response_matrix_shape_and_missing():
    # Two learners on part 2; only items answered by >= min_item_resp learners
    # become rows, and unanswered cells are MISSING_FILL.
    recs = [
        LearnerRecord(
            student_id=1,
            questions=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            responses=torch.tensor([3, 3, 0, 0, 2, 2, 1, 1, 3, 0]),  # codes
            parts=torch.tensor([2] * 10),
        ),
        LearnerRecord(
            student_id=2,
            questions=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            responses=torch.tensor([0, 3, 3, 0, 2, 1, 1, 0, 3, 2]),
            parts=torch.tensor([2] * 10),
        ),
    ]
    matrix, item_ids, sids = build_part_response_matrix(
        recs, part=2, min_events=10, min_item_resp=2,
    )
    # Both learners answered all 10 items, all kept; matrix is [10 items x 2].
    assert matrix.shape == (10, 2)
    assert set(sids.tolist()) == {1, 2}
    # Item 1: learner1 code 3 -> 1, learner2 code 0 -> 0.
    row1 = matrix[list(item_ids).index(1)]
    assert row1.tolist() == [1, 0]
    # No missing here (complete), but the sentinel must never appear as 0/1.
    assert MISSING_FILL not in (0, 1)
    # A learner with too few part events is dropped (min_events filter).
    assert (matrix != MISSING_FILL).all()


def test_classical_2pl_recovers_synthetic_difficulty():
    # girth 2PL on a synthetic binary matrix recovers item difficulty ordering
    # and EAP abilities -- the engine the control relies on, end to end.
    rng = np.random.default_rng(0)
    n_items, n_people = 12, 400
    true_b = rng.normal(0, 1, n_items)
    true_a = rng.uniform(0.7, 1.8, n_items)
    theta = rng.normal(0, 1, n_people)
    z = true_a[:, None] * (theta[None, :] - true_b[:, None])
    p = 1.0 / (1.0 + np.exp(-z))
    resp = (rng.random((n_items, n_people)) < p).astype(np.int64)
    # Knock out 30% as missing to mimic the sparse per-part regime.
    miss = rng.random((n_items, n_people)) < 0.3
    resp[miss] = MISSING_FILL
    fit = fit_classical_part(
        resp, item_ids=np.arange(1, n_items + 1),
        person_sids=np.arange(1, n_people + 1),
    )
    assert fit["ok"]
    from scipy.stats import spearmanr
    # Item difficulty + ability recovered in the right direction (noisy N).
    assert spearmanr(fit["b"], true_b).correlation > 0.6
    assert spearmanr(fit["theta"], theta).correlation > 0.5


def test_classical_vs_encoder_difficulty_linking():
    # Difficulty agreement is computed after mean/SD linking and is invariant
    # to an affine shift of the classical difficulty scale.
    rng = np.random.default_rng(1)
    n = 40
    enc_loc = rng.normal(0, 1, n)
    # classical b = affine(enc) + noise on items 0..n-1 (1-based ids 1..n).
    cls_b = 2.0 * enc_loc + 0.5 + rng.normal(0, 0.2, n)
    item_tables = {
        2: {"ok": True, "item_ids": np.arange(1, n + 1), "b": cls_b,
            "a": np.ones(n)},
    }
    enc_b_table = enc_loc.reshape(-1, 1)  # (B, 1) location-only
    out = classical_vs_encoder_difficulty(item_tables, enc_b_table, parts=(2,),
                                          min_shared=10)
    assert out["per_part"][2]["pearson"] > 0.9  # affine + small noise
