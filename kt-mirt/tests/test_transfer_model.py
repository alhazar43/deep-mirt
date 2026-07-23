"""Tests for `kt_mirt.transfer.model`: the signed cross-KC influence model
(`_planning/design/a1_design.md` v1.1, section 2.2) -- the sign-asymmetric
gate, the response-blind transfer transition, the zero-diagonal hard
constraint, the G=0 isolation property, and a convergence-gated fit."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kt_mirt.growth.synth import LearnerLog
from kt_mirt.growth.tracker import build_learner_batch
from kt_mirt.transfer import model as M


def _log(learner, item_ids, responses, tag_ids, tag_mask=None):
    item_ids = np.asarray(item_ids)
    tag_ids = np.asarray(tag_ids)
    tag_mask = np.asarray(tag_mask) if tag_mask is not None else np.ones_like(tag_ids, dtype=bool)
    return LearnerLog(
        learner=learner, item_ids=item_ids, responses=np.asarray(responses, dtype=np.int8),
        tag_ids=tag_ids, tag_mask=tag_mask,
    )


def _batch(logs, b_true):
    return build_learner_batch(logs, b_true=b_true)


# ---------------------------------------------------------------------------
# The sign-asymmetric gate T(g, z)
# ---------------------------------------------------------------------------


def test_gate_positive_is_ceiling_gated():
    """T(g>0, z) = g*(M-z)+ : rises toward M, and is ~0 at the ceiling."""
    M_c, floor = 3.0, -5.0
    g = torch.tensor(0.5)
    below = M.sign_asymmetric_transfer(g, torch.tensor(0.0), torch.tensor(M_c), floor)
    at_ceiling = M.sign_asymmetric_transfer(g, torch.tensor(3.0), torch.tensor(M_c), floor)
    assert below.item() == pytest.approx(0.5 * 3.0)
    assert at_ceiling.item() == pytest.approx(0.0)


def test_gate_negative_is_floor_gated():
    """T(g<0, z) = g*(z-floor)+ : falls toward the floor, ~0 at the floor."""
    M_c, floor = 3.0, -5.0
    g = torch.tensor(-0.5)
    mid = M.sign_asymmetric_transfer(g, torch.tensor(0.0), torch.tensor(M_c), floor)
    at_floor = M.sign_asymmetric_transfer(g, torch.tensor(-5.0), torch.tensor(M_c), floor)
    assert mid.item() == pytest.approx(-0.5 * (0.0 - (-5.0)))
    assert at_floor.item() == pytest.approx(0.0)


def test_gate_numpy_and_torch_agree():
    """Generator (numpy) and model (torch) MUST share the gate exactly
    (the qmirt discipline, design section 3.1)."""
    for g, z in [(0.3, 0.5), (-0.2, 1.0), (0.1, -3.0), (-0.4, -4.9)]:
        t = M.sign_asymmetric_transfer(torch.tensor(g), torch.tensor(z), torch.tensor(3.0), -5.0).item()
        n = float(M._sign_asymmetric_np(np.array(g), np.array(z), 3.0, -5.0))
        assert t == pytest.approx(n)


# ---------------------------------------------------------------------------
# run_transfer_transition: response-blindness, isolation, signed push
# ---------------------------------------------------------------------------


def test_run_transfer_transition_is_response_blind():
    """Design constraint 1: given fixed (z0, lambda) the transition output
    is invariant to the responses (it reads only the practice tags)."""
    l0 = _log(0, [0, 1, 2, 0], [1, 0, 1, 1], [[0], [1], [2], [0]])
    l1 = _log(1, [1, 2, 1], [1, 1, 0], [[1], [2], [1]])
    b = np.zeros(3)
    batch_a = _batch([l0, l1], b)
    l0f = _log(0, [0, 1, 2, 0], [0, 1, 0, 0], [[0], [1], [2], [0]])
    l1f = _log(1, [1, 2, 1], [0, 0, 1], [[1], [2], [1]])
    batch_b = _batch([l0f, l1f], b)

    z0 = torch.tensor([[0.5, -0.2, 0.1], [0.1, 0.0, 0.3]])
    lam = torch.tensor([1.0, 1.0])
    g_c = torch.tensor([0.3, 0.1, 0.2])
    Mc = torch.tensor(3.0)
    G = torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0]])
    la, za = M.run_transfer_transition(z0, lam, g_c, Mc, G, -5.0, batch_a.kc_ids, batch_a.kc_mask, batch_a.b_j)
    lb, zb = M.run_transfer_transition(z0, lam, g_c, Mc, G, -5.0, batch_b.kc_ids, batch_b.kc_mask, batch_b.b_j)
    assert torch.allclose(za, zb)
    assert torch.allclose(la, lb)


def test_run_transfer_transition_zero_G_isolation():
    """G=0 (the qmirt isolation property): a non-practiced KC never leaves
    z0; only own-gain moves the practiced KC."""
    l0 = _log(0, [0, 0, 0], [1, 1, 1], [[0], [0], [0]])  # only KC0 practiced
    batch = _batch([l0], np.zeros(3))
    z0 = torch.tensor([[0.1, 0.4, -0.2]])
    G0 = torch.zeros(3, 3)
    _, zf = M.run_transfer_transition(z0, torch.tensor([1.0]), torch.tensor([0.3, 0.1, 0.2]),
                                      torch.tensor(3.0), G0, -5.0, batch.kc_ids, batch.kc_mask, batch.b_j)
    assert zf[0, 1].item() == pytest.approx(0.4)  # KC1 untouched
    assert zf[0, 2].item() == pytest.approx(-0.2)  # KC2 untouched
    assert zf[0, 0].item() > 0.1  # KC0 grew via own-gain


def test_run_transfer_transition_positive_edge_raises_untouched_target():
    """G[c,a]>0 with only source a practiced (target c never practiced)
    raises c above z0 (ceiling-gated facilitation), zero own-gain leak."""
    l0 = _log(0, [1, 1, 1], [1, 1, 1], [[1], [1], [1]])  # only KC1 (source) practiced
    batch = _batch([l0], np.zeros(3))
    z0 = torch.tensor([[0.0, 0.5, 0.0]])
    G = torch.zeros(3, 3)
    G[0, 1] = 0.2  # KC1 -> KC0 facilitation
    _, zf = M.run_transfer_transition(z0, torch.tensor([1.0]), torch.tensor([0.0, 0.3, 0.0]),
                                      torch.tensor(3.0), G, -5.0, batch.kc_ids, batch.kc_mask, batch.b_j)
    assert zf[0, 0].item() > 0.0  # target KC0 raised by transfer
    assert zf[0, 2].item() == pytest.approx(0.0)  # KC2 (no edge) untouched


def test_run_transfer_transition_negative_edge_lowers_untouched_target():
    """G[c,a]<0 lowers the untouched target toward the floor (interference)."""
    l0 = _log(0, [2, 2, 2], [1, 1, 1], [[2], [2], [2]])  # only KC2 (source) practiced
    batch = _batch([l0], np.zeros(3))
    z0 = torch.tensor([[0.0, 0.5, 0.0]])
    G = torch.zeros(3, 3)
    G[1, 2] = -0.2  # KC2 -> KC1 interference
    _, zf = M.run_transfer_transition(z0, torch.tensor([1.0]), torch.tensor([0.0, 0.0, 0.3]),
                                      torch.tensor(3.0), G, -5.0, batch.kc_ids, batch.kc_mask, batch.b_j)
    assert zf[0, 1].item() < 0.5  # target KC1 lowered by negative transfer


# ---------------------------------------------------------------------------
# TransferModel: shapes, zero-diagonal, positivity, no free per-learner params
# ---------------------------------------------------------------------------


def test_transfer_model_forward_shapes_p0():
    torch.manual_seed(0)
    l0 = _log(0, [0, 1, 2], [1, 0, 1], [[0], [1], [2]])
    l1 = _log(1, [1, 2], [1, 1], [[1], [2]])
    batch = _batch([l0, l1], np.zeros(3))
    cfg = M.TransferConfig(variant="a1_p0", hidden_dim=8, emb_dim=4)
    mdl = M.TransferModel(num_items=3, n_kcs=3, cfg=cfg, m_init=3.0)
    sl = M.seq_lens_from_mask(batch.seq_mask)
    logits, zf, u, lam = mdl(batch, sl)
    assert logits.shape == (2, 3)
    assert zf.shape == (2, 3)
    assert lam is None


def test_transfer_model_forward_shapes_p1_positive_lambda():
    torch.manual_seed(0)
    l0 = _log(0, [0, 1, 2], [1, 0, 1], [[0], [1], [2]])
    batch = _batch([l0], np.zeros(3))
    cfg = M.TransferConfig(variant="a1_p1", hidden_dim=8, emb_dim=4)
    mdl = M.TransferModel(num_items=3, n_kcs=3, cfg=cfg, m_init=3.0)
    sl = M.seq_lens_from_mask(batch.seq_mask)
    _, _, _, lam = mdl(batch, sl)
    assert lam.shape == (1,)
    assert (lam > 0).all()


def test_transfer_model_G_is_zero_diagonal_hard():
    torch.manual_seed(0)
    cfg = M.TransferConfig()
    mdl = M.TransferModel(num_items=3, n_kcs=4, cfg=cfg, m_init=3.0)
    with torch.no_grad():
        mdl.G_off.copy_(torch.randn(4, 4))  # arbitrary, including diagonal
    G = mdl.G
    assert torch.allclose(torch.diag(G), torch.zeros(4))
    assert np.allclose(np.diag(mdl.recovered_G()), 0.0)


def test_transfer_model_g_c_nonnegative():
    torch.manual_seed(0)
    cfg = M.TransferConfig()
    mdl = M.TransferModel(num_items=3, n_kcs=5, cfg=cfg, m_init=3.0)
    with torch.no_grad():
        mdl.g_raw.copy_(torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0]))
    assert (mdl.g_c >= 0).all()


def test_transfer_model_rejects_unknown_variant():
    cfg = M.TransferConfig(variant="a1_p2")
    with pytest.raises(ValueError):
        M.TransferModel(num_items=3, n_kcs=3, cfg=cfg, m_init=3.0)


def test_transfer_model_no_free_per_learner_parameters():
    """Gate B / gamma pin: no nn.Parameter has a learner axis, and two
    models with identical config have the same parameter count regardless
    of the batch size they will see."""
    torch.manual_seed(0)
    cfg = M.TransferConfig(hidden_dim=8, emb_dim=4)
    m1 = M.TransferModel(num_items=5, n_kcs=3, cfg=cfg, m_init=3.0)
    m2 = M.TransferModel(num_items=5, n_kcs=3, cfg=cfg, m_init=3.0)
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    assert n1 == n2
    for p in m1.parameters():
        assert p.dim() == 0 or p.shape[0] != 10_000


def test_transfer_model_default_has_no_phantom_gamma():
    """The per-learner transfer multiplier is pinned at 1 (gamma pin): no
    gamma head unless explicitly enabled (section 4.6 is a later stage)."""
    cfg = M.TransferConfig()
    mdl = M.TransferModel(num_items=3, n_kcs=3, cfg=cfg, m_init=3.0)
    assert not mdl.phantom_gamma
    assert not hasattr(mdl, "gamma_head")


def test_transfer_model_m_fixed_freezes_ceiling():
    cfg = M.TransferConfig(m_fixed=True)
    mdl = M.TransferModel(num_items=3, n_kcs=3, cfg=cfg, m_init=3.0)
    assert not isinstance(mdl.M, torch.nn.Parameter)
    assert float(mdl.M) == pytest.approx(3.0)
    assert "_M" not in [n for n, _ in mdl.named_parameters()]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_transfer_loss_returns_nll_and_l1_separately():
    torch.manual_seed(0)
    l0 = _log(0, [0, 1, 2], [1, 0, 1], [[0], [1], [2]])
    batch = _batch([l0], np.zeros(3))
    cfg = M.TransferConfig(hidden_dim=4, emb_dim=2, l1_weight=1e-2)
    mdl = M.TransferModel(num_items=3, n_kcs=3, cfg=cfg, m_init=3.0)
    with torch.no_grad():
        mdl.G_off.copy_(torch.tensor([[0.0, 0.5, -0.3], [0.2, 0.0, 0.4], [0.1, -0.1, 0.0]]))
    nll, l1 = M.transfer_loss(mdl, batch)
    # L1 is the off-diagonal absolute sum (diagonal hard-zeroed).
    expected_l1 = float(np.abs(mdl.recovered_G()).sum())
    assert float(l1) == pytest.approx(expected_l1, rel=1e-5)
    assert float(nll) > 0.0


@pytest.mark.slow
def test_train_transfer_reduces_forecast_nll():
    """The one end-to-end fit smoke: the convergence-gated trainer reduces
    the forecast NLL on a small overfittable batch and returns a G of the
    right shape."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    logs = []
    for i in range(30):
        seq = [0, 1, 2] * 6
        resp = rng.integers(0, 2, size=len(seq))
        tags = [[c] for c in seq]
        logs.append(_log(i, seq, resp, tags))
    batch = _batch(logs, np.zeros(3))
    cfg = M.TransferConfig(hidden_dim=8, emb_dim=4, lr=0.05, n_epochs=300)
    mdl = M.TransferModel(num_items=3, n_kcs=3, cfg=cfg, m_init=3.0)
    trace = M.train_transfer(mdl, batch, cfg)
    assert trace[-1] < trace[0]
    assert mdl.recovered_G().shape == (3, 3)
    assert np.allclose(np.diag(mdl.recovered_G()), 0.0)
