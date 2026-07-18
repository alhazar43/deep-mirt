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


def test_train_active_epoch_ceiling_is_floored_against_legacy_small_values():
    """The ACT-P0 fabrication repair (module docstring note 8): a caller
    still passing the pre-fix fixed count (20) must NOT get a 20-epoch
    run -- ``n_epochs`` is a ceiling floored at ``ACT_MIN_EPOCHS_CEILING``
    and the convergence gate governs. The earliest the windowed rule can
    fire is ``2*window_epochs + patience_epochs`` epochs."""
    torch.manual_seed(0)
    n = 40
    l0 = _log(0, [0] * n, [1, 0] * (n // 2), [[0]] * n)
    batch = _batch_from_logs([l0], b_true=np.array([0.0]))
    cfg = active.ActiveConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=20)
    model = active.ActiveModel(num_items=1, n_kcs=1, cfg=cfg, m_init=3.0)
    trace = active.train_active(model, batch, cfg)
    assert len(trace) >= 2 * cfg.window_epochs + cfg.patience_epochs
    assert len(trace) <= active.ACT_MIN_EPOCHS_CEILING


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


# ---------------------------------------------------------------------------
# ACT-P0 fabrication regression (the repaired train_active; module
# docstring note 8; `_planning/research/act_p0_diagnosis.md`)
# ---------------------------------------------------------------------------
#
# The diagnosis in miniature, after the convergence-gated repair. Data
# construction follows the diagnosis's own ablation scripts
# (`kt-mirt/scripts/probe/_diag_act_p0_mechanism.py` / `_diag_act_p0_
# fix_check.py`): single-tag items (item id == KC id, so item difficulty
# IS KC difficulty), z0_true = xi_i + eta_c + noise, interleaved practice
# order; the no-growth twin draws every opportunity from the SAME flat
# z0_true, the known-growth twin advances z through EXACTLY the recurrence
# ACT itself uses (z <- z + g_true*(M_true - z)+, matched family).
#
# lr = 0.1 (not the campaign's 0.05) purely to halve epochs-to-convergence
# and keep this section within its wall-time budget: the convergence gate
# under test is lr-agnostic, and the OLD broken 20-epoch loop still
# fabricates hard at lr = 0.1 (diagnosis section 3.4: no-growth
# pop_mean_rise 0.325), so the discrimination is preserved. drift_tol is
# raised to 0.1 alongside (the guard bounds per-epoch parameter movement,
# whose scale is Adam's normalized step ~ lr; the default 5e-2 is
# calibrated to the campaign's lr = 0.05).
#
# The assertion levels encode the estimator's TRUE CONVERGED behavior at
# this scale, measured on this exact configuration (stationarity study,
# 2026-07-18), with margin for cross-platform torch numerics -- they are
# not tuned to any certification bar. For reference: the broken
# fixed-20-epoch loop reads no-growth pop ~0.33 / p95 ~0.47 at this lr.


def _twin_logs(n_learners: int, n_kcs: int, seed: int, g_true: float):
    """Toy twin builder (see section comment). Returns (logs, b_c)."""
    rng = np.random.default_rng(seed)
    xi_i = rng.normal(0.0, 1.0, size=n_learners)
    eta_c = rng.normal(0.0, 0.5, size=n_kcs)
    b_c = rng.normal(0.0, 1.0, size=n_kcs)
    m_true = float(np.percentile(b_c, 95) + 2.0)
    # heterogeneous practice counts, KDD-like anchors (the diagnosis
    # scripts' own values; shorter sequences weaken the flatness evidence
    # enough that the no-growth fit no longer converges within the ceiling)
    p_anchor = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    v_anchor = np.array([1.0, 4.0, 8.0, 16.0, 20.0])

    logs = []
    for i in range(n_learners):
        u = rng.uniform(0.0, 1.0, size=n_kcs)
        Ts = np.clip(np.round(np.interp(u, p_anchor, v_anchor)).astype(int), 1, 20)
        z_slice = xi_i[i] + eta_c + rng.normal(0.0, 0.3, size=n_kcs)
        entries = []
        for c in range(n_kcs):
            for t in np.sort(rng.random(int(Ts[c]))):
                entries.append((t, c))
        entries.sort(key=lambda e: e[0])
        item_ids = np.array([c for (_, c) in entries], dtype=np.int64)
        z_running = z_slice.copy()
        responses = np.zeros(len(item_ids), dtype=np.int8)
        for pos, c in enumerate(item_ids):
            p_correct = 1.0 / (1.0 + np.exp(-(z_running[c] - b_c[c])))
            responses[pos] = int(rng.random() < p_correct)
            if g_true > 0:
                z_running[c] = z_running[c] + g_true * max(m_true - z_running[c], 0.0)
        logs.append(_log(i, item_ids, responses, item_ids.reshape(-1, 1)))
    return logs, b_c


def _fit_act_p0(logs, b_c, seed: int = 0):
    """One repaired-trainer ACT-P0 fit plus the closed-form population
    read (`run.py:_act_implied_rises`'s logic, specialized to this dense
    every-learner-practices-every-KC toy). Intra-op threads are pinned to
    2 for the fit (restored after): the tiny per-timestep tensors make
    the default pool's synchronization overhead dominate, roughly 2x
    wall time at this scale."""
    n_kcs = len(b_c)
    batch = _batch_from_logs(logs, b_true=b_c)
    m_init = float(np.percentile(b_c, 95) + 2.0)
    b_ref = float(np.median(b_c))
    cfg = active.ActiveConfig(
        variant="act_p0", hidden_dim=16, emb_dim=8, lr=0.1, drift_tol=1e-1, seed=seed
    )
    torch.manual_seed(seed)
    model = active.ActiveModel(num_items=n_kcs, n_kcs=n_kcs, cfg=cfg, m_init=m_init)
    n_threads_before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        trace = active.train_active(model, batch, cfg)
    finally:
        torch.set_num_threads(n_threads_before)

    with torch.no_grad():
        seq_lens = active.seq_lens_from_mask(batch.seq_mask)
        u_i, _ = model.recognition(batch.item_ids, batch.responses, seq_lens)
        u_i = u_i.cpu().numpy()
        g_c = model.g_c.detach().cpu().numpy()
        v_c = model.v_c.detach().cpu().numpy()
        M = float(model.M.detach().cpu().item())
    per_learner = np.array([
        active.implied_score_rise(
            u_i[i] + v_c, np.ones(n_kcs), g_c, M, np.full(n_kcs, b_ref)
        ).mean()
        for i in range(len(logs))
    ])
    return {
        "pop_mean": float(per_learner.mean()),
        "p95_abs": float(np.percentile(np.abs(per_learner), 95)),
        "g_c_mean": float(g_c.mean()),
        "epochs_run": len(trace),
    }


@pytest.fixture(scope="module")
def act_p0_no_growth_fit():
    logs, b_c = _twin_logs(n_learners=80, n_kcs=6, seed=0, g_true=0.0)
    return _fit_act_p0(logs, b_c, seed=0)


@pytest.fixture(scope="module")
def act_p0_known_growth_fit():
    logs, b_c = _twin_logs(n_learners=80, n_kcs=6, seed=0, g_true=0.15)
    return _fit_act_p0(logs, b_c, seed=0)


def test_act_p0_repair_no_growth_twin_is_silent_at_convergence(act_p0_no_growth_fit):
    """The fabrication regression proper: on a no-growth twin the
    CONVERGED ACT-P0 implied population rise must sit at its measured
    converged level (well under the CG1 KDD silence bar, 0.01/0.01, at
    this scale), not at the untrained-init fabrication level (~0.33
    pop / ~0.47 p95 with the old fixed 20-epoch loop at this lr)."""
    r = act_p0_no_growth_fit
    assert r["pop_mean"] <= 0.01
    assert r["p95_abs"] <= 0.015
    assert r["g_c_mean"] <= 0.05  # driven far down from softplus(0) ~ 0.69


def test_act_p0_repair_no_growth_convergence_takes_far_beyond_legacy_epochs(act_p0_no_growth_fit):
    """The mechanism guard: unlearning the softplus(0) init on no-growth
    data NEEDS hundreds of epochs (diagnosis section 3.2); if this fit
    ever stops near the legacy 20-epoch count again, the gate was lost."""
    assert act_p0_no_growth_fit["epochs_run"] > 300


def test_act_p0_repair_known_growth_twin_recovers_true_gain(act_p0_known_growth_fit):
    """The positive control (diagnosis section 3.3's mirror-image trap):
    the SAME stopping rule must still detect real growth and recover the
    true gain (g_true = 0.15) within tolerance -- a repair that silences
    both twins is as broken as one that fabricates on both."""
    r = act_p0_known_growth_fit
    assert 0.10 <= r["g_c_mean"] <= 0.20
    assert r["pop_mean"] >= 0.30  # confidently fires on real growth
    assert r["epochs_run"] < active.ACT_MIN_EPOCHS_CEILING  # gate, not ceiling, stopped it
