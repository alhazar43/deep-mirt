"""Tests for the swappable Transformer and DKVMN backbones.

Both subclass ``BaseSeqEncoder`` and must honour the same interface as
``LSTMEncoder``: identical public accessors, identical embedding-table layout
(item_emb / resp_emb / alpha_item_emb), the single-shift causal alignment, and
a hidden width equal to ``hidden_dim``.  These tests pin that contract for both
backbones at once, plus the end-to-end DeepIRTModel paths and the decoupled
alpha contract (mirroring test_decoupled_alpha.py).

Causality
---------
``_direct_hidden`` must be strictly causal: h_t depends only on inputs at
positions <= t.  The base single-shift then guarantees the PREDICTION theta at
step t never sees r_t.  Both are asserted below.
"""

from __future__ import annotations

import math

import pytest
import torch

from deep_irt.core import DeepIRTModel
from deep_irt.core.transformer_encoder import TransformerEncoder
from deep_irt.core.dkvmn_encoder import DKVMNEncoder


_SEED = 42
_DEVICE = torch.device("cpu")
_BACKBONES = ("transformer", "dkvmn")


def _data(num_items=10, n_cats=4, n=16, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    it = torch.randint(0, num_items, (n, t), generator=g)
    rp = torch.randint(0, n_cats, (n, t), generator=g)
    return it, rp


def _enc(kind, num_items=10, n_cats=4, emb_dim=4, hidden_dim=8,
         separate_theta=False, alpha_emb_dim=None, seed=_SEED):
    torch.manual_seed(seed)
    common = dict(num_items=num_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
                  n_cats=n_cats, separate_theta=separate_theta,
                  alpha_emb_dim=alpha_emb_dim)
    if kind == "transformer":
        return TransformerEncoder(**common, n_heads=2, n_layers=2)
    return DKVMNEncoder(**common, memory_size=8)


# ---------------------------------------------------------------------------
# Contract 1: shapes of the public accessors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _BACKBONES)
def test_accessor_shapes(kind):
    it, rp = _data()
    enc = _enc(kind, hidden_dim=8)
    B, T = it.shape
    assert enc.encode(it, rp).shape == (B, T)
    assert enc.theta_for_prediction(it, rp).shape == (B, T)
    assert enc.state_for_prediction(it, rp).shape == (B, T, 8)
    assert enc.get_final_theta(it, rp).shape == (B,)


# ---------------------------------------------------------------------------
# Contract 2: causality of _direct_hidden + the prediction-theta shift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _BACKBONES)
def test_direct_hidden_is_causal(kind):
    """Perturbing inputs at positions > t must not change h at positions <= t."""
    it, rp = _data()
    enc = _enc(kind)
    enc.eval()
    t_cut = it.size(1) // 2
    it2, rp2 = it.clone(), rp.clone()
    g = torch.Generator().manual_seed(7)
    it2[:, t_cut:] = torch.randint(0, enc.num_items, it2[:, t_cut:].shape,
                                   generator=g)
    rp2[:, t_cut:] = torch.randint(0, enc.n_cats, rp2[:, t_cut:].shape,
                                   generator=g)
    with torch.no_grad():
        h1 = enc._direct_hidden(it, rp)
        h2 = enc._direct_hidden(it2, rp2)
    assert torch.allclose(h1[:, :t_cut], h2[:, :t_cut], atol=1e-6), \
        f"{kind}: future inputs leaked into past hidden"
    # And the future was actually perturbed (guard against a no-op test).
    assert not torch.allclose(h1[:, t_cut:], h2[:, t_cut:], atol=1e-6)


@pytest.mark.parametrize("kind", _BACKBONES)
def test_prediction_theta_does_not_see_current_or_future_responses(kind):
    """Changing responses[:, t:] must not move theta_for_prediction[:, :t].

    The base single-shift guarantees the step-t prediction theta excludes r_t;
    causal _direct_hidden guarantees it excludes later responses too.
    """
    it, rp = _data()
    for sep in (False, True):
        enc = _enc(kind, separate_theta=sep)
        enc.eval()
        t = it.size(1) // 2
        rp2 = rp.clone()
        rp2[:, t:] = (rp2[:, t:] + 1) % enc.n_cats
        with torch.no_grad():
            a = enc.theta_for_prediction(it, rp)[:, :t]
            b = enc.theta_for_prediction(it, rp2)[:, :t]
        assert torch.allclose(a, b, atol=1e-6), f"{kind} sep={sep}"


# ---------------------------------------------------------------------------
# Contract 3: embedding tables exist with correct shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _BACKBONES)
def test_tables_present(kind):
    enc = _enc(kind, num_items=12, emb_dim=4)
    assert enc.item_emb.weight.shape == (12, 4)
    assert enc.resp_emb.weight.shape == (enc.n_cats, 4)
    # No alpha key without alpha_emb_dim.
    assert not hasattr(enc, "alpha_item_emb")


@pytest.mark.parametrize("kind", _BACKBONES)
def test_alpha_table_only_when_requested(kind):
    enc = _enc(kind, num_items=12, emb_dim=4, hidden_dim=8, alpha_emb_dim=16)
    assert hasattr(enc, "alpha_item_emb")
    assert enc.alpha_item_emb.weight.shape == (12, 16)
    # The wide key grows the decoder's alpha head input, never the state width.
    it, rp = _data(num_items=12)
    assert enc.state_for_prediction(it, rp).shape[-1] == 8


@pytest.mark.parametrize("kind", _BACKBONES)
def test_state_width_is_hidden_dim_regardless_of_alpha(kind):
    it, rp = _data()
    plain = _enc(kind, hidden_dim=8, alpha_emb_dim=None)
    wide = _enc(kind, hidden_dim=8, alpha_emb_dim=16)
    assert plain.state_for_prediction(it, rp).shape[-1] == 8
    assert wide.state_for_prediction(it, rp).shape[-1] == 8


# ---------------------------------------------------------------------------
# Contract 4: end-to-end via DeepIRTModel (tiny smoke fit)
# ---------------------------------------------------------------------------

def _fit_and_check_recovery(model, it, rp, num_items, n_cats):
    model.fit(it, rp, n_epochs=5, verbose=False)
    rec = model.recover_item_params(it, rp)
    assert rec["a"].shape == (num_items,)
    assert rec["b"].shape == (num_items, n_cats - 1)
    assert rec["seen"].shape == (num_items,)
    assert bool(rec["seen"].all())
    assert torch.isfinite(torch.tensor(rec["a"])).all()
    assert torch.isfinite(torch.tensor(rec["b"])).all()
    assert (rec["a"] > 0).all()


@pytest.mark.parametrize("kind", _BACKBONES)
def test_end_to_end_state_alpha_decoupled(kind):
    """The decoupled config: wide alpha key + exp transform."""
    num_items, n_cats = 10, 4
    it, rp = _data(num_items=num_items, n_cats=n_cats, n=20, t=15)
    m = DeepIRTModel(
        num_items=num_items, emb_dim=4, hidden_dim=8, n_cats=n_cats,
        decoder="gpcm", encoder=kind, state_alpha=True, alpha_emb_dim=16,
        alpha_log_scale=1.0, device=_DEVICE, seed=_SEED,
    )
    _fit_and_check_recovery(m, it, rp, num_items, n_cats)


@pytest.mark.parametrize("kind", _BACKBONES)
def test_end_to_end_state_alpha_cheap(kind):
    """The cheap state_alpha config (no wide key)."""
    num_items, n_cats = 10, 4
    it, rp = _data(num_items=num_items, n_cats=n_cats, n=20, t=15)
    m = DeepIRTModel(
        num_items=num_items, emb_dim=4, hidden_dim=8, n_cats=n_cats,
        decoder="gpcm", encoder=kind, state_alpha=True,
        device=_DEVICE, seed=_SEED,
    )
    _fit_and_check_recovery(m, it, rp, num_items, n_cats)


@pytest.mark.parametrize("kind", _BACKBONES)
def test_end_to_end_plain(kind):
    """The plain config (no state_alpha): static item-only recovery."""
    num_items, n_cats = 10, 4
    it, rp = _data(num_items=num_items, n_cats=n_cats, n=20, t=15)
    m = DeepIRTModel(
        num_items=num_items, emb_dim=4, hidden_dim=8, n_cats=n_cats,
        decoder="gpcm", encoder=kind, decouple=False, device=_DEVICE, seed=_SEED,
    )
    r = m.fit(it, rp, n_epochs=5, verbose=False)
    assert math.isfinite(r["final_nll"])
    rec = m.recover_item_params()                     # static: all items
    assert rec["a"].shape == (num_items,)
    assert rec["b"].shape == (num_items, n_cats - 1)
    assert (rec["a"] > 0).all()
    assert torch.isfinite(torch.tensor(rec["a"])).all()


# ---------------------------------------------------------------------------
# Contract 5: decoupling -- the wide key moves alpha, never theta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _BACKBONES)
def test_wide_key_moves_alpha_not_theta(kind):
    it, rp = _data(num_items=10)
    m = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=4, decoder="gpcm",
        encoder=kind, state_alpha=True, alpha_emb_dim=16,
        device=_DEVICE, seed=_SEED,
    )
    m.fit(it, rp, n_epochs=5, verbose=False)

    theta_before = m.track(it, rp).clone()
    rec_before = m.recover_item_params(it, rp)

    with torch.no_grad():
        m.encoder.alpha_item_emb.weight.add_(
            torch.randn_like(m.encoder.alpha_item_emb.weight))
    theta_after = m.track(it, rp)
    rec_after = m.recover_item_params(it, rp)

    assert torch.allclose(theta_before, theta_after, atol=1e-6), \
        f"{kind}: theta moved when only the alpha key changed"
    assert not torch.allclose(
        torch.tensor(rec_before["a"]), torch.tensor(rec_after["a"]),
        atol=1e-4), f"{kind}: discrimination did not move with the alpha key"
