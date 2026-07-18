"""Direct probability-normalization and shape tests for kt_mirt.core.decoders.

Complements test_alpha_link.py (alpha-map / item_params contracts) and
test_nrm_decoder.py (NRM). This file pins the GPCM/Binary2PL log-probability
normalization directly and covers BradleyTerryDecoder, which none of the
ported kt-irt tests exercise at the decoder level (it is only reachable
through DeepIRTModel's decoder="bt" path in the source repo, never
unit-tested directly).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kt_mirt.core.decoders import (
    Binary2PLDecoder,
    BradleyTerryDecoder,
    GPCMDecoder,
)


# ---------------------------------------------------------------------------
# GPCMDecoder
# ---------------------------------------------------------------------------

def test_gpcm_log_probs_sum_to_one():
    torch.manual_seed(0)
    dec = GPCMDecoder(emb_dim=6, n_cats=5)
    emb = torch.randn(9, 6)
    theta = torch.randn(9)
    p = dec.item_params(emb)
    lp = dec.log_probs(theta, p["alpha"], p["beta"])
    assert lp.shape == (9, 5)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(9), atol=1e-5)


def test_gpcm_logits_psi0_is_always_zero():
    torch.manual_seed(1)
    dec = GPCMDecoder(emb_dim=4, n_cats=4)
    emb = torch.randn(5, 4)
    theta = torch.randn(5)
    p = dec.item_params(emb)
    logits = dec.logits(theta, p["alpha"], p["beta"])
    assert torch.allclose(logits[:, 0], torch.zeros(5))


def test_gpcm_item_params_sorted_sorts_beta_ascending():
    torch.manual_seed(2)
    dec = GPCMDecoder(emb_dim=4, n_cats=5)
    emb = torch.randn(6, 4)
    sorted_params = dec.item_params_sorted(emb)
    beta = sorted_params["beta"]
    assert torch.all(torch.diff(beta, dim=-1) >= 0)


def test_gpcm_alpha_always_positive_under_default_softplus():
    torch.manual_seed(3)
    dec = GPCMDecoder(emb_dim=4, n_cats=4)
    emb = torch.randn(20, 4) * 5.0  # wide spread, including large negative raw
    a = dec.item_params(emb)["alpha"]
    assert (a > 0).all()


def test_gpcm_nll_gradient_flow():
    torch.manual_seed(4)
    dec = GPCMDecoder(emb_dim=4, n_cats=4)
    emb = torch.randn(7, 4, requires_grad=True)
    theta = torch.randn(7)
    resp = torch.randint(0, 4, (7,))
    loss = dec.nll(theta, emb, resp)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(emb.grad).all()
    assert dec.fc_a.weight.grad is not None
    assert dec.fc_b.weight.grad is not None


def test_gpcm_k2_matches_binary_structure():
    """GPCM at K=2 has one threshold and a 2-way normalized distribution."""
    torch.manual_seed(5)
    dec = GPCMDecoder(emb_dim=4, n_cats=2)
    emb = torch.randn(5, 4)
    theta = torch.randn(5)
    p = dec.item_params(emb)
    assert p["beta"].shape == (5, 1)
    lp = dec.log_probs(theta, p["alpha"], p["beta"])
    assert lp.shape == (5, 2)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(5), atol=1e-5)


# ---------------------------------------------------------------------------
# Binary2PLDecoder
# ---------------------------------------------------------------------------

def test_binary_log_probs_sum_to_one():
    torch.manual_seed(6)
    dec = Binary2PLDecoder(emb_dim=4)
    emb = torch.randn(8, 4)
    theta = torch.randn(8)
    p = dec.item_params(emb)
    lp = dec.log_probs(theta, p["alpha"], p["beta"])
    assert torch.allclose(lp.exp().sum(-1), torch.ones(8), atol=1e-5)


def test_binary_logit_matches_log_prob_difference():
    torch.manual_seed(7)
    dec = Binary2PLDecoder(emb_dim=4)
    emb = torch.randn(6, 4)
    theta = torch.randn(6)
    z = dec.binary_logit(theta, emb)
    p = dec.item_params(emb)
    lp = dec.log_probs(theta, p["alpha"], p["beta"])
    assert torch.allclose(z, lp[:, 1] - lp[:, 0], atol=1e-5)
    # sigmoid(z) must equal P(y=1) = exp(lp[:,1])
    assert torch.allclose(torch.sigmoid(z), lp[:, 1].exp(), atol=1e-5)


def test_binary_nll_matches_bce_with_logits():
    torch.manual_seed(8)
    dec = Binary2PLDecoder(emb_dim=4)
    emb = torch.randn(10, 4)
    theta = torch.randn(10)
    resp = torch.randint(0, 2, (10,))
    nll = dec.nll(theta, emb, resp)
    z = dec.binary_logit(theta, emb)
    ref = F.binary_cross_entropy_with_logits(z, resp.float())
    assert torch.allclose(nll, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# BradleyTerryDecoder
# ---------------------------------------------------------------------------

def test_bt_item_strength_shape():
    torch.manual_seed(9)
    dec = BradleyTerryDecoder(emb_dim=5)
    emb = torch.randn(4, 5)
    s = dec.item_strength(emb)
    assert s.shape == (4, 1)
    assert dec.item_params(emb)["strength"].shape == (4, 1)


def test_bt_nll_pairs_gradient_flow():
    torch.manual_seed(10)
    dec = BradleyTerryDecoder(emb_dim=4)
    emb_i = torch.randn(6, 4, requires_grad=True)
    emb_j = torch.randn(6, 4, requires_grad=True)
    outcome = torch.randint(0, 2, (6,))
    loss = dec.nll_pairs(emb_i, emb_j, outcome)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(emb_i.grad).all()
    assert torch.isfinite(emb_j.grad).all()
    assert dec.fc_s.weight.grad is not None


def test_bt_confident_correct_prediction_costs_less_than_confident_wrong():
    dec = BradleyTerryDecoder(emb_dim=1)
    with torch.no_grad():
        dec.fc_s.weight.fill_(1.0)
        dec.fc_s.bias.zero_()
    emb_i = torch.tensor([[3.0]])   # strength = 3
    emb_j = torch.tensor([[-3.0]])  # strength = -3
    correct = dec.nll_pairs(emb_i, emb_j, torch.tensor([1]), reg_strength=0.0)
    wrong = dec.nll_pairs(emb_i, emb_j, torch.tensor([0]), reg_strength=0.0)
    assert correct.item() < wrong.item()
