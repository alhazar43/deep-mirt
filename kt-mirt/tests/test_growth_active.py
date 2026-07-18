"""Tests for `kt_mirt.growth.active`: the ACT transition (practice-gated,
ceiling-gated, response-blind), ACT-P0/ACT-P1, the ceiling-fixed CG1
fallback, the closed-form implied-trajectory/rise statistics, and RB-A's
firing verdict (`_planning/design/a4_design.md` v1.1, section 2.3, 5.2).
Also covers `kt_mirt.growth.recognition` (ACT's amortized ``u_i``/
``lambda_i`` heads), which has no dedicated test file of its own since it
exists only to serve ACT."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kt_mirt.growth import active, recognition, tracker
from kt_mirt.growth.synth import LearnerLog


def _log(learner, item_ids, responses, tag_ids, tag_mask=None):
    item_ids = np.asarray(item_ids)
    tag_ids = np.asarray(tag_ids)
    tag_mask = np.asarray(tag_mask) if tag_mask is not None else np.ones_like(tag_ids, dtype=bool)
    return LearnerLog(
        learner=learner, item_ids=item_ids, responses=np.asarray(responses, dtype=np.int8),
        tag_ids=tag_ids, tag_mask=tag_mask,
    )


# ---------------------------------------------------------------------------
# recognition.py
# ---------------------------------------------------------------------------


def test_recognition_network_p0_emits_only_u():
    torch.manual_seed(0)
    cfg = recognition.RecognitionConfig(hidden_dim=8, emb_dim=4, predict_lambda=False)
    net = recognition.RecognitionNetwork(num_items=2, cfg=cfg)
    item_ids = torch.tensor([[0, 1, 0]])
    responses = torch.tensor([[1, 0, 1]])
    seq_lens = torch.tensor([3])
    u_i, lam = net(item_ids, responses, seq_lens)
    assert u_i.shape == (1,)
    assert lam is None


def test_recognition_network_p1_emits_positive_lambda():
    torch.manual_seed(0)
    cfg = recognition.RecognitionConfig(hidden_dim=8, emb_dim=4, predict_lambda=True)
    net = recognition.RecognitionNetwork(num_items=2, cfg=cfg)
    item_ids = torch.tensor([[0, 1, 0], [1, 1, 0]])
    responses = torch.tensor([[1, 0, 1], [0, 0, 1]])
    seq_lens = torch.tensor([3, 2])
    u_i, lam = net(item_ids, responses, seq_lens)
    assert lam.shape == (2,)
    assert (lam > 0).all()  # softplus positivity


def test_recognition_network_reads_full_window_including_last_step():
    """Module docstring's interpretation: `state_for_prediction` would
    exclude the learner's final response, but this network deliberately
    reads the RAW final hidden state instead. Changing only the LAST
    response (with seq_lens unchanged) must therefore change u_i."""
    torch.manual_seed(0)
    cfg = recognition.RecognitionConfig(hidden_dim=8, emb_dim=4)
    net = recognition.RecognitionNetwork(num_items=2, cfg=cfg)
    item_ids = torch.tensor([[0, 1, 0]])
    seq_lens = torch.tensor([3])
    u_a, _ = net(item_ids, torch.tensor([[1, 0, 1]]), seq_lens)
    u_b, _ = net(item_ids, torch.tensor([[1, 0, 0]]), seq_lens)
    assert not torch.allclose(u_a, u_b)


def test_recognition_network_uses_seq_lens_to_pick_final_position():
    torch.manual_seed(0)
    cfg = recognition.RecognitionConfig(hidden_dim=8, emb_dim=4)
    net = recognition.RecognitionNetwork(num_items=2, cfg=cfg)
    # Two learners, padded to T=4; learner 0 has true length 2, learner 1 length 4.
    item_ids = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 1]])
    responses = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1]])
    seq_lens = torch.tensor([2, 4])
    u_i, _ = net(item_ids, responses, seq_lens)
    assert u_i.shape == (2,)
    assert torch.isfinite(u_i).all()


# ---------------------------------------------------------------------------
# ceiling_init
# ---------------------------------------------------------------------------


def test_ceiling_init_matches_formula():
    b_hat = np.array([-1.0, 0.0, 1.0, 2.0, 3.0] * 20)  # 100 values
    expected = float(np.percentile(b_hat, 95) + 2.0)
    assert active.ceiling_init(b_hat) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# run_transition: response-blindness, practice-gating, ceiling-gating
# ---------------------------------------------------------------------------


def _batch_from_logs(logs, b_true):
    return tracker.build_learner_batch(logs, b_true=b_true)


def test_run_transition_is_response_blind():
    l0 = _log(0, [0, 1, 0, 0, 1], [1, 0, 1, 1, 0], [[0], [1], [0], [0], [1]])
    l1 = _log(1, [0, 0, 0], [1, 1, 1], [[0], [0], [0]])
    batch_a = _batch_from_logs([l0, l1], b_true=np.array([0.0, 0.0]))

    l0_flip = _log(0, [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [[0], [1], [0], [0], [1]])
    l1_flip = _log(1, [0, 0, 0], [0, 0, 0], [[0], [0], [0]])
    batch_b = _batch_from_logs([l0_flip, l1_flip], b_true=np.array([0.0, 0.0]))

    z0 = torch.tensor([[0.5, -0.2], [0.1, 0.0]])
    lam = torch.tensor([1.0, 1.0])
    g_c = torch.tensor([0.3, 0.1])
    M = torch.tensor(3.0)
    logits_a, z_a = active.run_transition(z0, lam, g_c, M, batch_a.kc_ids, batch_a.kc_mask, batch_a.b_j)
    logits_b, z_b = active.run_transition(z0, lam, g_c, M, batch_b.kc_ids, batch_b.kc_mask, batch_b.b_j)
    assert torch.allclose(z_a, z_b)
    assert torch.allclose(logits_a, logits_b)


def test_run_transition_practice_gating_untouched_kc_stays_at_z0():
    l1 = _log(1, [0, 0, 0], [1, 1, 1], [[0], [0], [0]])  # only practices KC0
    batch = _batch_from_logs([l1], b_true=np.array([0.0]))
    z0 = torch.tensor([[0.1, 0.0]])
    lam = torch.tensor([1.0])
    g_c = torch.tensor([0.3, 0.1])
    M = torch.tensor(3.0)
    _, z_final = active.run_transition(z0, lam, g_c, M, batch.kc_ids, batch.kc_mask, batch.b_j)
    assert z_final[0, 1].item() == pytest.approx(z0[0, 1].item())  # KC1 never practiced
    assert z_final[0, 0].item() > z0[0, 0].item()  # KC0 grew


def test_run_transition_ceiling_gating_saturates_and_never_exceeds_M():
    T = 200
    l0 = _log(0, [0] * T, [1] * T, [[0]] * T)
    batch = _batch_from_logs([l0], b_true=np.array([0.0]))
    z0 = torch.tensor([[0.0]])
    lam = torch.tensor([1.0])
    g_c = torch.tensor([0.5])
    M = torch.tensor(2.0)
    _, z_final = active.run_transition(z0, lam, g_c, M, batch.kc_ids, batch.kc_mask, batch.b_j)
    assert z_final[0, 0].item() <= 2.0 + 1e-4
    assert z_final[0, 0].item() > 1.9  # should approach the ceiling after 200 practices


def test_run_transition_zero_gain_never_grows():
    l0 = _log(0, [0, 0, 0], [1, 1, 1], [[0], [0], [0]])
    batch = _batch_from_logs([l0], b_true=np.array([0.0]))
    z0 = torch.tensor([[0.5]])
    lam = torch.tensor([1.0])
    g_c = torch.tensor([0.0])  # zero gain
    M = torch.tensor(3.0)
    _, z_final = active.run_transition(z0, lam, g_c, M, batch.kc_ids, batch.kc_mask, batch.b_j)
    assert z_final[0, 0].item() == pytest.approx(0.5)


def test_run_transition_multi_tag_readout_is_masked_mean_of_pre_update_z():
    l0 = _log(0, [0], [1], [[0, 1]], [[True, True]])
    batch = _batch_from_logs([l0], b_true=np.array([0.0]))
    z0 = torch.tensor([[1.0, 3.0]])
    lam = torch.tensor([1.0])
    g_c = torch.tensor([0.0, 0.0])  # zero gain isolates the READOUT check from any update effect
    M = torch.tensor(5.0)
    logits, _ = active.run_transition(z0, lam, g_c, M, batch.kc_ids, batch.kc_mask, batch.b_j)
    assert logits[0, 0].item() == pytest.approx((1.0 + 3.0) / 2.0)


# ---------------------------------------------------------------------------
# ActiveModel: construction, no free per-learner parameters, m_fixed
# ---------------------------------------------------------------------------


def test_active_model_forward_shapes_p0():
    torch.manual_seed(0)
    l0 = _log(0, [0, 1, 0], [1, 0, 1], [[0], [1], [0]])
    l1 = _log(1, [0, 0], [1, 1], [[0], [0]])
    batch = _batch_from_logs([l0, l1], b_true=np.array([0.0, 0.0]))
    cfg = active.ActiveConfig(variant="act_p0", hidden_dim=8, emb_dim=4)
    model = active.ActiveModel(num_items=2, n_kcs=2, cfg=cfg, m_init=3.0)
    seq_lens = active.seq_lens_from_mask(batch.seq_mask)
    logits, z_final, u_i, lam = model(batch, seq_lens)
    assert logits.shape == (2, 3)
    assert z_final.shape == (2, 2)
    assert lam is None


def test_active_model_forward_shapes_p1_has_positive_lambda():
    torch.manual_seed(0)
    l0 = _log(0, [0, 1, 0], [1, 0, 1], [[0], [1], [0]])
    batch = _batch_from_logs([l0], b_true=np.array([0.0, 0.0]))
    cfg = active.ActiveConfig(variant="act_p1", hidden_dim=8, emb_dim=4)
    model = active.ActiveModel(num_items=2, n_kcs=2, cfg=cfg, m_init=3.0)
    seq_lens = active.seq_lens_from_mask(batch.seq_mask)
    _, _, _, lam = model(batch, seq_lens)
    assert lam.shape == (1,)
    assert (lam > 0).all()


def test_active_model_rejects_unknown_variant():
    cfg = active.ActiveConfig(variant="act_p2")
    with pytest.raises(ValueError):
        active.ActiveModel(num_items=2, n_kcs=2, cfg=cfg, m_init=3.0)


def test_active_model_g_c_always_nonnegative():
    torch.manual_seed(0)
    cfg = active.ActiveConfig()
    model = active.ActiveModel(num_items=2, n_kcs=5, cfg=cfg, m_init=3.0)
    with torch.no_grad():
        model.g_raw.copy_(torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0]))
    assert (model.g_c >= 0).all()


def test_active_model_no_free_per_learner_parameters():
    """Gate B: parameter count must not scale with the number of learners
    in a batch (only population-level and recognition-network weights are
    learnable)."""
    torch.manual_seed(0)
    cfg = active.ActiveConfig(hidden_dim=8, emb_dim=4)
    model = active.ActiveModel(num_items=3, n_kcs=4, cfg=cfg, m_init=3.0)
    n_params = sum(p.numel() for p in model.parameters())
    # Constructing a second model with identical config must have the SAME
    # parameter count regardless of how many learners will ever be batched
    # through it (there is no learner axis in any nn.Parameter shape).
    model2 = active.ActiveModel(num_items=3, n_kcs=4, cfg=cfg, m_init=3.0)
    assert sum(p.numel() for p in model2.parameters()) == n_params
    for p in model.parameters():
        assert p.dim() == 0 or p.shape[0] != 10_000  # sanity: no accidentally-learner-shaped tensor


def test_active_model_m_fixed_true_freezes_ceiling():
    cfg = active.ActiveConfig(m_fixed=True)
    model = active.ActiveModel(num_items=2, n_kcs=2, cfg=cfg, m_init=3.0)
    assert not isinstance(model.M, torch.nn.Parameter)
    assert float(model.M) == pytest.approx(3.0)
    param_names = [n for n, _ in model.named_parameters()]
    assert "_M" not in param_names


def test_active_model_m_fixed_false_is_trainable():
    cfg = active.ActiveConfig(m_fixed=False)
    model = active.ActiveModel(num_items=2, n_kcs=2, cfg=cfg, m_init=3.0)
    assert isinstance(model.M, torch.nn.Parameter)
    param_names = [n for n, _ in model.named_parameters()]
    assert "_M" in param_names


# ---------------------------------------------------------------------------
# Training / forecast NLL sanity
# ---------------------------------------------------------------------------


def test_train_active_reduces_loss_on_tiny_overfit_batch():
    torch.manual_seed(0)
    n = 40
    l0 = _log(0, [0] * n, [1, 0] * (n // 2), [[0]] * n)
    batch = _batch_from_logs([l0], b_true=np.array([0.0]))
    cfg = active.ActiveConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=30)
    model = active.ActiveModel(num_items=1, n_kcs=1, cfg=cfg, m_init=3.0)
    trace = active.train_active(model, batch, cfg)
    assert trace[-1] < trace[0]


def test_forecast_nll_matches_active_loss_no_grad():
    torch.manual_seed(0)
    l0 = _log(0, [0, 0], [1, 0], [[0], [0]])
    batch = _batch_from_logs([l0], b_true=np.array([0.0]))
    cfg = active.ActiveConfig(hidden_dim=4, emb_dim=2)
    model = active.ActiveModel(num_items=1, n_kcs=1, cfg=cfg, m_init=3.0)
    direct = active.active_loss(model, batch).item()
    via_forecast = active.forecast_nll(model, batch)
    assert direct == pytest.approx(via_forecast)


# ---------------------------------------------------------------------------
# implied_z_trajectory / implied_score_rise / implied_z_rise
# ---------------------------------------------------------------------------


def test_implied_z_trajectory_monotonic_nondecreasing_with_positive_gain():
    traj = active.implied_z_trajectory(np.array([0.0]), np.array([1.0]), np.array([0.3]), M=3.0, n_opportunities=10)
    diffs = np.diff(traj.ravel())
    assert (diffs >= -1e-9).all()


def test_implied_z_trajectory_zero_gain_is_flat():
    traj = active.implied_z_trajectory(np.array([0.5]), np.array([1.0]), np.array([0.0]), M=3.0, n_opportunities=10)
    assert np.allclose(traj.ravel(), 0.5)


def test_implied_z_trajectory_already_at_ceiling_stays_put():
    traj = active.implied_z_trajectory(np.array([3.0]), np.array([1.0]), np.array([0.5]), M=3.0, n_opportunities=10)
    assert np.allclose(traj.ravel(), 3.0)


def test_implied_score_rise_zero_gain_is_zero():
    rise = active.implied_score_rise(np.array([0.0]), np.array([1.0]), np.array([0.0]), M=3.0, b_ref=np.array([0.0]))
    assert rise[0] == pytest.approx(0.0)


def test_implied_score_rise_positive_gain_gives_positive_rise():
    rise = active.implied_score_rise(np.array([-1.0]), np.array([1.0]), np.array([0.2]), M=3.0, b_ref=np.array([0.0]))
    assert rise[0] > 0


def test_implied_z_rise_matches_trajectory_endpoints():
    z0, lam, g, M = np.array([0.0]), np.array([1.0]), np.array([0.2]), 3.0
    traj = active.implied_z_trajectory(z0, lam, g, M, 10)
    rise = active.implied_z_rise(z0, lam, g, M, n_from=1, n_to=10)
    assert rise[0] == pytest.approx(traj[9, 0] - traj[0, 0])


# ---------------------------------------------------------------------------
# RB-A firing verdict
# ---------------------------------------------------------------------------


def test_rb_a_firing_fires_when_all_three_conditions_hold():
    bed_rise = np.array([0.06, 0.07, 0.055])
    kc_rise = np.array([[0.06, 0.001], [0.07, 0.002], [0.055, -0.001]])
    v = active.rb_a_firing(bed_rise, kc_rise, null_bed_rise=0.005)
    assert v.bed_fires is True
    assert v.kc_fires.tolist() == [True, False]


def test_rb_a_firing_fails_below_magnitude_bar():
    bed_rise = np.array([0.02, 0.03, 0.025])  # below 0.05 bar
    kc_rise = np.array([[0.02, 0.0], [0.03, 0.0], [0.025, 0.0]])
    v = active.rb_a_firing(bed_rise, kc_rise, null_bed_rise=0.001)
    assert v.bed_fires is False


def test_rb_a_firing_fails_below_null_multiple():
    bed_rise = np.array([0.06, 0.07, 0.055])
    kc_rise = np.array([[0.06, 0.0], [0.07, 0.0], [0.055, 0.0]])
    v = active.rb_a_firing(bed_rise, kc_rise, null_bed_rise=0.02)  # 5x*0.02=0.1 > mean rise
    assert v.bed_fires is False


def test_rb_a_firing_fails_on_sign_inconsistency():
    bed_rise = np.array([0.06, -0.07, 0.055])  # sign flips across seeds
    kc_rise = np.array([[0.06, 0.0], [-0.07, 0.0], [0.055, 0.0]])
    v = active.rb_a_firing(bed_rise, kc_rise, null_bed_rise=0.001)
    assert v.bed_fires is False


def test_rb_a_firing_kc_never_fires_if_bed_does_not():
    bed_rise = np.array([0.02, 0.03])  # bed does not fire
    kc_rise = np.array([[0.9, 0.9], [0.9, 0.9]])  # KC-level rise is huge, irrelevant
    v = active.rb_a_firing(bed_rise, kc_rise, null_bed_rise=0.001)
    assert v.bed_fires is False
    assert not v.kc_fires.any()


def test_rb_a_firing_empty_kc_array_is_defensive():
    v = active.rb_a_firing(np.array([0.06, 0.07]), np.zeros((2, 0)), null_bed_rise=0.001)
    assert v.kc_fires.shape == (0,)


# ---------------------------------------------------------------------------
# Saturation reporting rule
# ---------------------------------------------------------------------------


def test_report_gain_withholds_saturated_kcs():
    g_c = np.array([0.1, 0.2, 0.3])
    is_unsaturated = np.array([True, False, True])
    out = active.report_gain(g_c, is_unsaturated)
    assert out[0] == pytest.approx(0.1)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(0.3)
