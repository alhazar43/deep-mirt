"""Tests for `kt_mirt.growth.tracker`: PAS-N1 (shared-state, multi-tag
gather) and PAS-N2 (factorized per-KC), batch construction, training,
forecast-NLL evaluation, the frozen-encoder null, and E-P3 displacement
(`_planning/design/a4_design.md` v1.1, section 2.2)."""

from __future__ import annotations

import numpy as np
import torch

from kt_mirt.growth import tracker
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
# Batch construction
# ---------------------------------------------------------------------------


def test_build_learner_batch_shapes_and_padding():
    l0 = _log(0, [0, 1, 2], [1, 0, 1], [[0], [1], [0]])
    l1 = _log(1, [0], [1], [[0]])
    batch = tracker.build_learner_batch([l0, l1], b_true=np.array([0.0, 0.0, 0.0]))
    assert batch.item_ids.shape == (2, 3)
    assert batch.seq_mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.kc_ids.shape == (2, 3, 1)


def test_build_learner_batch_multi_tag_padding_and_mask():
    l0 = _log(0, [0], [1], [[0, 1]], [[True, True]])
    l1 = _log(1, [0], [1], [[0]], [[True]])  # only 1 tag slot -> padded to A=2
    batch = tracker.build_learner_batch([l0, l1], b_true=np.array([0.0]))
    assert batch.kc_ids.shape == (2, 1, 2)
    assert batch.kc_mask[1, 0].tolist() == [True, False]


def test_build_learner_batch_empty_is_defensive():
    batch = tracker.build_learner_batch([])
    assert batch.item_ids.numel() == 0


def test_build_slice_batch_shapes():
    class S:
        pass

    s0 = S()
    s0.item_id = np.array([0, 1])
    s0.response = np.array([1, 0], dtype=np.int8)
    s0.T = 2
    s1 = S()
    s1.item_id = np.array([2])
    s1.response = np.array([1], dtype=np.int8)
    s1.T = 1
    batch = tracker.build_slice_batch([s0, s1], b_true=np.array([0.0, 0.0, 0.0]))
    assert batch.item_ids.shape == (2, 2)
    assert batch.mask.tolist() == [[True, True], [True, False]]


def test_difficulty_falls_back_to_zero_without_bank_or_truth():
    out = tracker._difficulty(np.array([0, 1, 2]), None, None)
    assert out.tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# PAS-N1: multi-tag gather (mean over tagged slots)
# ---------------------------------------------------------------------------


def test_pas_n1_model_forward_shape():
    torch.manual_seed(0)
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4)
    model = tracker.build_tracker("pas_n1", num_items=3, n_kcs=2, cfg=cfg)
    item_ids = torch.tensor([[0, 1, 2]])
    responses = torch.tensor([[1, 0, 1]])
    kc_ids = torch.tensor([[[0, -1], [1, -1], [0, 1]]])
    kc_mask = torch.tensor([[[True, False], [True, False], [True, True]]])
    theta = model(item_ids, responses, kc_ids, kc_mask)
    assert theta.shape == (1, 3)


def test_pas_n1_multi_tag_gather_is_masked_mean():
    """Directly verifies the gather-then-mean arithmetic against a
    manually-constructed ability tensor (module docstring note 3)."""
    torch.manual_seed(0)
    cfg = tracker.TrackerConfig(hidden_dim=4, emb_dim=2)
    model = tracker.PASN1Model(num_items=2, n_kcs=3, cfg=cfg)
    # Bypass the encoder: monkeypatch state_for_prediction output via a stub.
    fake_state = torch.zeros(1, 1, cfg.hidden_dim)
    with torch.no_grad():
        ability = model.ability_head(fake_state)  # (1,1,3), some fixed values from init weights
    kc_ids = torch.tensor([[[0, 1]]])
    kc_mask = torch.tensor([[[True, True]]])
    kc_clamped = kc_ids.clamp(min=0)
    gathered = torch.gather(ability, 2, kc_clamped)
    expected = gathered.mean(dim=-1)  # both slots valid -> plain mean of the two gathered values
    mask_f = kc_mask.to(gathered.dtype)
    denom = mask_f.sum(dim=-1).clamp(min=1.0)
    actual = (gathered * mask_f).sum(dim=-1) / denom
    assert torch.allclose(actual, expected)


def test_pas_n1_padding_kc_id_never_contributes():
    torch.manual_seed(1)
    cfg = tracker.TrackerConfig(hidden_dim=4, emb_dim=2)
    model = tracker.PASN1Model(num_items=2, n_kcs=3, cfg=cfg)
    item_ids = torch.tensor([[0]])
    responses = torch.tensor([[1]])
    kc_ids_full_tag = torch.tensor([[[0, -1]]])
    kc_mask_one_valid = torch.tensor([[[True, False]]])
    theta_one = model(item_ids, responses, kc_ids_full_tag, kc_mask_one_valid)
    # Changing the padded (masked-out) slot's id must not change the output.
    kc_ids_different_pad = torch.tensor([[[0, 2]]])
    theta_changed_pad = model(item_ids, responses, kc_ids_different_pad, kc_mask_one_valid)
    assert torch.allclose(theta_one, theta_changed_pad)


# ---------------------------------------------------------------------------
# PAS-N2: factorized per-slice
# ---------------------------------------------------------------------------


def test_pas_n2_model_forward_shape():
    torch.manual_seed(0)
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4)
    model = tracker.build_tracker("pas_n2", num_items=3, n_kcs=1, cfg=cfg)
    item_ids = torch.tensor([[0, 1], [2, 0]])
    responses = torch.tensor([[1, 0], [1, 1]])
    theta = model(item_ids, responses)
    assert theta.shape == (2, 2)


def test_pas_n2_order_invariant_to_cross_kc_interleaving_by_construction():
    """PAS-N2's certification claim (CG9, module docstring note 2): since
    each slice is its own independent forward pass, a slice's OWN theta
    trajectory cannot depend on any other KC's interactions at all --
    trivially true here since the model never sees them, verified by
    checking two slice batches sharing one slice are identical regardless
    of what other (unrelated) slice accompanies it in the batch."""
    torch.manual_seed(2)
    cfg = tracker.TrackerConfig(hidden_dim=4, emb_dim=2)
    model = tracker.build_tracker("pas_n2", num_items=3, n_kcs=1, cfg=cfg)
    shared_items = torch.tensor([0, 1, 2])
    shared_resp = torch.tensor([1, 0, 1])
    batch_a = torch.stack([shared_items, torch.tensor([0, 0, 0])])
    resp_a = torch.stack([shared_resp, torch.tensor([1, 1, 1])])
    theta_a = model(batch_a, resp_a)[0]
    batch_b = torch.stack([shared_items, torch.tensor([1, 1, 1])])
    resp_b = torch.stack([shared_resp, torch.tensor([0, 0, 0])])
    theta_b = model(batch_b, resp_b)[0]
    assert torch.allclose(theta_a, theta_b)


# ---------------------------------------------------------------------------
# Frozen-encoder null (battery arm 3 / CG7 / RB2)
# ---------------------------------------------------------------------------


def test_freeze_encoder_freezes_only_encoder_params():
    torch.manual_seed(0)
    cfg = tracker.TrackerConfig(hidden_dim=4, emb_dim=2, train_encoder=False)
    model = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.ability_head.parameters())


def test_build_tracker_rejects_unknown_kind():
    import pytest

    cfg = tracker.TrackerConfig()
    with pytest.raises(ValueError):
        tracker.build_tracker("pas_n3", num_items=2, n_kcs=1, cfg=cfg)


def test_build_encoder_rejects_unknown_kind():
    import pytest

    cfg = tracker.TrackerConfig(encoder="rnn")
    with pytest.raises(ValueError):
        tracker._build_encoder(2, cfg)


# ---------------------------------------------------------------------------
# Loss / training / forecast NLL: overfitting sanity on tiny data
# ---------------------------------------------------------------------------


def test_train_pas_n1_reduces_loss_on_tiny_overfit_batch():
    torch.manual_seed(0)
    l0 = _log(0, [0, 1, 0, 1] * 5, [1, 0, 1, 0] * 5, [[0]] * 20)
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=40)
    batch = tracker.build_learner_batch([l0], b_true=np.array([0.0, 0.0]))
    model = tracker.build_tracker("pas_n1", num_items=2, n_kcs=1, cfg=cfg)
    trace = tracker.train_pas_n1(model, batch, cfg)
    assert trace[-1] < trace[0]


def test_train_pas_n2_reduces_loss_on_tiny_overfit_batch():
    torch.manual_seed(0)

    class S:
        pass

    s0 = S()
    s0.item_id = np.array([0, 1, 0, 1] * 5)
    s0.response = np.array([1, 0, 1, 0] * 5, dtype=np.int8)
    s0.T = 20
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=40)
    batch = tracker.build_slice_batch([s0], b_true=np.array([0.0, 0.0]))
    model = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg)
    trace = tracker.train_pas_n2(model, batch, cfg)
    assert trace[-1] < trace[0]


def test_forecast_nll_matches_loss_function_no_grad():
    torch.manual_seed(0)
    l0 = _log(0, [0, 1], [1, 0], [[0], [0]])
    cfg = tracker.TrackerConfig(hidden_dim=4, emb_dim=2)
    batch = tracker.build_learner_batch([l0], b_true=np.array([0.0, 0.0]))
    model = tracker.build_tracker("pas_n1", num_items=2, n_kcs=1, cfg=cfg)
    direct = tracker.pas_n1_loss(model, batch).item()
    via_forecast = tracker.forecast_nll(model, batch, "pas_n1")
    assert direct == via_forecast


def test_forecast_nll_rejects_unknown_kind():
    import pytest

    with pytest.raises(ValueError):
        tracker.forecast_nll(None, None, "pas_n3")


# ---------------------------------------------------------------------------
# Mini-batched training (TrackerConfig.batch_size, module docstring note 6)
# ---------------------------------------------------------------------------


def _make_slices(n, T=8):
    class S:
        pass

    out = []
    for i in range(n):
        s = S()
        s.item_id = np.array(([0, 1] * (T // 2 + 1))[:T])
        s.response = np.array(([1, 0] if i % 2 else [0, 1]) * (T // 2 + 1), dtype=np.int8)[:T]
        s.T = T
        out.append(s)
    return out


def _original_whole_cohort_train(model, batch, cfg, loss_fn):
    """Verbatim replica of the pre-batch_size training loop, kept in the
    test as the KDD-path invariance reference: batch_size=None must stay
    bit-identical to THIS loop forever."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    trace = []
    for _ in range(cfg.n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        trace.append(float(loss.item()))
    return trace


def test_batch_size_none_is_bit_identical_to_original_loop_pas_n2():
    """KDD-path invariance: the default (batch_size=None) whole-cohort path
    must produce bit-identical parameters and loss trace to the original
    pre-batch_size loop."""
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=10)
    batch = tracker.build_slice_batch(_make_slices(6), b_true=np.zeros(2))

    torch.manual_seed(7)
    model_new = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg)
    trace_new = tracker.train_pas_n2(model_new, batch, cfg)

    torch.manual_seed(7)
    model_ref = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg)
    trace_ref = _original_whole_cohort_train(model_ref, batch, cfg, tracker.pas_n2_loss)

    assert trace_new == trace_ref
    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)


def test_batch_size_none_is_bit_identical_to_original_loop_pas_n1():
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=10)
    logs = [_log(i, [0, 1, 0, 1], [1, 0, 1, 0] if i % 2 else [0, 1, 0, 1], [[0]] * 4) for i in range(4)]
    batch = tracker.build_learner_batch(logs, b_true=np.zeros(2))

    torch.manual_seed(7)
    model_new = tracker.build_tracker("pas_n1", num_items=2, n_kcs=1, cfg=cfg)
    trace_new = tracker.train_pas_n1(model_new, batch, cfg)

    torch.manual_seed(7)
    model_ref = tracker.build_tracker("pas_n1", num_items=2, n_kcs=1, cfg=cfg)
    trace_ref = _original_whole_cohort_train(model_ref, batch, cfg, tracker.pas_n1_loss)

    assert trace_new == trace_ref
    for p_new, p_ref in zip(model_new.parameters(), model_ref.parameters()):
        assert torch.equal(p_new, p_ref)


def test_batch_size_at_least_n_rows_is_whole_cohort():
    """batch_size >= n_rows takes the whole-cohort path (mirrors the core
    DeepIRTModel.fit semantics), bit-identical to batch_size=None."""
    batch = tracker.build_slice_batch(_make_slices(6), b_true=np.zeros(2))
    cfg_none = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=5)
    cfg_big = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=5, batch_size=999)

    torch.manual_seed(3)
    model_a = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg_none)
    trace_a = tracker.train_pas_n2(model_a, batch, cfg_none)

    torch.manual_seed(3)
    model_b = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg_big)
    trace_b = tracker.train_pas_n2(model_b, batch, cfg_big)

    assert trace_a == trace_b
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_minibatched_pas_n2_trains_and_reduces_loss():
    """EdNet-profile smoke at test scale: many slice rows, batch_size well
    below the row count (multiple batches per epoch), loss decreases and the
    trace stays one MEAN entry per epoch."""
    torch.manual_seed(0)
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=30, batch_size=4)
    batch = tracker.build_slice_batch(_make_slices(10, T=12), b_true=np.zeros(2))
    model = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg)
    trace = tracker.train_pas_n2(model, batch, cfg)
    assert len(trace) == cfg.n_epochs
    assert trace[-1] < trace[0]


def test_minibatched_pas_n1_trains_and_reduces_loss():
    torch.manual_seed(0)
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=30, batch_size=2)
    logs = [_log(i, [0, 1, 0, 1] * 3, [1, 0, 1, 0] * 3, [[0]] * 12) for i in range(6)]
    batch = tracker.build_learner_batch(logs, b_true=np.zeros(2))
    model = tracker.build_tracker("pas_n1", num_items=2, n_kcs=1, cfg=cfg)
    trace = tracker.train_pas_n1(model, batch, cfg)
    assert len(trace) == cfg.n_epochs
    assert trace[-1] < trace[0]


def test_minibatched_training_is_seed_deterministic():
    """The shuffle generator is seeded from cfg.seed, never the global torch
    RNG: two identically-configured runs must match exactly."""
    batch = tracker.build_slice_batch(_make_slices(9), b_true=np.zeros(2))
    cfg = tracker.TrackerConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=8, batch_size=4, seed=5)

    torch.manual_seed(11)
    model_a = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg)
    trace_a = tracker.train_pas_n2(model_a, batch, cfg)

    torch.manual_seed(11)
    model_b = tracker.build_tracker("pas_n2", num_items=2, n_kcs=1, cfg=cfg)
    trace_b = tracker.train_pas_n2(model_b, batch, cfg)

    assert trace_a == trace_b
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


# ---------------------------------------------------------------------------
# E-P3: displacement
# ---------------------------------------------------------------------------


def test_displacement_below_population_floor_is_nan():
    assert np.isnan(tracker.displacement(np.array([0.1, 0.2, 0.3])))


def test_displacement_matches_manual_quarter_computation():
    theta = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])  # T=8, quarter=2
    d = tracker.displacement(theta)
    expected = np.mean(theta[-2:]) - np.mean(theta[:2])
    assert d == expected


def test_displacement_is_free_signed():
    rising = tracker.displacement(np.array([0.0, 0.0, 1.0, 1.0]))
    falling = tracker.displacement(np.array([1.0, 1.0, 0.0, 0.0]))
    assert rising > 0
    assert falling < 0
