"""Tests for `kt_mirt.transfer.ct0`: the per-edge sign-F1 scorer and the
CT1-bar / power-curve verdict logic (`_planning/design/a1_design.md` v1.1,
sections 4.1, 5.1, 2.1.1). These cover the pure-function scoring on
hand-built matrices (no fitting); the end-to-end fit is exercised by the
slow model test and the standalone smoke run."""

from __future__ import annotations

import numpy as np
import pytest

from kt_mirt.transfer import ct0
from kt_mirt.transfer.synth import default_G_true


G_TRUE = default_G_true(3, g_pos=0.05, g_neg=0.02)  # + at (0,1), - at (1,2)


# ---------------------------------------------------------------------------
# edge_signs / band thresholding
# ---------------------------------------------------------------------------


def test_edge_signs_bands_out_small_cells_and_zeros_diagonal():
    G = np.array([[9.9, 0.10, 0.001], [-0.10, 9.9, -0.20], [0.02, -0.005, 9.9]])
    s = ct0.edge_signs(G, band=0.05)
    assert s[0, 1] == 1 and s[1, 0] == -1 and s[1, 2] == -1
    assert s[0, 2] == 0  # 0.001 below band
    assert s[2, 1] == 0  # -0.005 below band
    assert np.allclose(np.diag(s), 0.0)  # diagonal forced to 0 despite the 9.9s


# ---------------------------------------------------------------------------
# sign_f1 on the trivial ground-truth cases (design section 4.1)
# ---------------------------------------------------------------------------


def test_sign_f1_perfect_recovery():
    r = ct0.sign_f1(G_TRUE * 3.0, G_TRUE, band=0.01)
    assert r["sign_f1"] == pytest.approx(1.0)
    assert r["sign_accuracy_true_edges"] == pytest.approx(1.0)
    assert r["false_edge_rate"] == pytest.approx(0.0)
    assert r["pos_f1"] == pytest.approx(1.0)
    assert r["neg_f1"] == pytest.approx(1.0)


def test_sign_f1_all_zero_predictor_scores_zero():
    """The degenerate all-zero baseline: F1 = 0 (no recall), false-edge
    rate 0 (design section 4.1 -- why the headline metric is recall-bearing
    F1, not false-edge-rate alone)."""
    r = ct0.sign_f1(np.zeros((3, 3)), G_TRUE, band=0.01)
    assert r["sign_f1"] == pytest.approx(0.0)
    assert r["false_edge_rate"] == pytest.approx(0.0)
    assert r["baseline_allzero_f1"] == pytest.approx(0.0)


def test_sign_f1_flipped_signs_score_zero():
    """A sign flip is maximally penalized (both a false positive and a
    false negative), so fully-flipped recovery scores F1 = 0."""
    r = ct0.sign_f1(-G_TRUE * 3.0, G_TRUE, band=0.01)
    assert r["sign_f1"] == pytest.approx(0.0)
    assert r["sign_accuracy_true_edges"] == pytest.approx(0.0)


def test_sign_f1_false_edge_on_zero_cell():
    """One spurious edge on a true-zero cell: recall stays 1 (both true
    edges found), precision drops to 2/3, F1 = 0.8, false-edge rate 1/4."""
    G_hat = G_TRUE * 3.0
    G_hat[2, 0] = 0.5  # spurious edge on the zero-row control KC
    r = ct0.sign_f1(G_hat, G_TRUE, band=0.01)
    assert r["sign_f1"] == pytest.approx(0.8)
    assert r["false_edge_rate"] == pytest.approx(0.25)


def test_sign_f1_missing_negative_edge_hits_neg_recall():
    """Zeroing the negative edge (L1 over-shrinkage) leaves the positive
    half perfect but tanks negative-half recall -- the gate clause that
    stops L1 passing by silently killing the weaker negative edges."""
    G_hat = G_TRUE * 3.0
    G_hat[1, 2] = 0.0  # negative edge shrunk to zero
    r = ct0.sign_f1(G_hat, G_TRUE, band=0.01)
    assert r["pos_f1"] == pytest.approx(1.0)
    assert r["neg_recall"] == pytest.approx(0.0)
    assert r["neg_f1"] == pytest.approx(0.0)


def test_sign_f1_beats_random_sign_baseline_when_perfect():
    r = ct0.sign_f1(G_TRUE * 3.0, G_TRUE, band=0.01)
    assert r["sign_f1"] > r["baseline_randsign_f1"] + 0.15


# ---------------------------------------------------------------------------
# pooled null band (design section 4.2)
# ---------------------------------------------------------------------------


def test_pooled_null_band_is_offdiagonal_quantile():
    ng0 = np.array([[0.0, 0.01, 0.02], [0.03, 0.0, 0.01], [0.02, 0.04, 0.0]])
    ng1 = np.array([[0.0, 0.02, 0.01], [0.01, 0.0, 0.05], [0.03, 0.01, 0.0]])
    band = ct0.pooled_null_band([ng0, ng1], quantile=1.0)  # max of pooled off-diagonal
    assert band == pytest.approx(0.05)


def test_pooled_null_band_ignores_diagonal():
    ng = np.array([[9.9, 0.01, 0.02], [0.01, 9.9, 0.02], [0.01, 0.02, 9.9]])
    band = ct0.pooled_null_band([ng], quantile=1.0)
    assert band == pytest.approx(0.02)  # the 9.9 diagonal is excluded


# ---------------------------------------------------------------------------
# CT1 bar / verdict logic (design section 5.1, 5.4)
# ---------------------------------------------------------------------------


def test_clears_ct1_true_on_strong_metrics():
    metrics = {"sign_f1": 0.95, "sign_accuracy_true_edges": 1.0, "false_edge_rate": 0.0,
               "neg_f1": 0.9, "baseline_allzero_f1": 0.0, "baseline_randsign_f1": 0.25}
    assert ct0.clears_ct1(metrics, sign_consistent=True)


def test_clears_ct1_fails_on_high_false_edge_rate():
    metrics = {"sign_f1": 0.95, "sign_accuracy_true_edges": 1.0, "false_edge_rate": 0.5,
               "neg_f1": 0.9, "baseline_allzero_f1": 0.0, "baseline_randsign_f1": 0.25}
    assert not ct0.clears_ct1(metrics, sign_consistent=True)


def test_clears_ct1_fails_on_weak_negative_half():
    metrics = {"sign_f1": 0.85, "sign_accuracy_true_edges": 0.9, "false_edge_rate": 0.0,
               "neg_f1": 0.5, "baseline_allzero_f1": 0.0, "baseline_randsign_f1": 0.25}
    assert not ct0.clears_ct1(metrics, sign_consistent=True)


def test_clears_ct1_fails_without_sign_consistency():
    metrics = {"sign_f1": 0.95, "sign_accuracy_true_edges": 1.0, "false_edge_rate": 0.0,
               "neg_f1": 0.9, "baseline_allzero_f1": 0.0, "baseline_randsign_f1": 0.25}
    assert not ct0.clears_ct1(metrics, sign_consistent=False)


def test_clears_ct1_fails_when_not_beating_random_baseline():
    metrics = {"sign_f1": 0.35, "sign_accuracy_true_edges": 0.9, "false_edge_rate": 0.0,
               "neg_f1": 0.8, "baseline_allzero_f1": 0.0, "baseline_randsign_f1": 0.25}
    assert not ct0.clears_ct1(metrics, sign_consistent=True)  # 0.35 < 0.25 + 0.15
