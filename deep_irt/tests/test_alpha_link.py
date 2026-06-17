"""Tests for the discrimination positivity transform (``alpha_log_scale``).

The GPCM/binary decoder pushes the raw discrimination-head output through a
positive map (GPCM needs ``a > 0``), one of two transforms:

  softplus (default, ``alpha_log_scale=None``) -- the original transform.
  exponential (``alpha_log_scale=s``)          -- ``exp(s * raw)``, ma-irt's
      transform; with ``s = 1.0`` it is plain ``exp(raw)``.

Contracts pinned here
---------------------
1. ``alpha_log_scale=None`` is bit-for-bit identical to before (softplus): same
   weights, same trained NLL, same recovered alpha.
2. The exp transform is exactly ``exp(s * raw)`` on the static head.
3. The exp transform changes the model (different trained NLL from softplus) --
   it is a genuinely different positivity map, not a no-op.
4. Validation: ``alpha_log_scale <= 0`` raises; the transform is GPCM/binary-only.
5. The exp transform composes with the decoupled wide item key.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from deep_irt.core import DeepIRTModel
from deep_irt.core.decoders import GPCMDecoder, Binary2PLDecoder


_SEED = 42
_DEVICE = torch.device("cpu")


def _data(num_items=10, n_cats=4, n=20, t=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    it = torch.randint(0, num_items, (n, t), generator=g)
    rp = torch.randint(0, n_cats, (n, t), generator=g)
    return it, rp


# ---------------------------------------------------------------------------
# Contract 1: alpha_log_scale=None is bit-for-bit identical (softplus)
# ---------------------------------------------------------------------------

def test_link_none_is_bit_identical():
    it, rp = _data()
    common = dict(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                  decoder="gpcm", state_alpha=True, device=_DEVICE, seed=_SEED)
    m_default = DeepIRTModel(**common)
    m_none = DeepIRTModel(alpha_log_scale=None, **common)
    r1 = m_default.fit(it, rp, n_epochs=15, verbose=False)
    r2 = m_none.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isclose(r1["final_nll"], r2["final_nll"], rel_tol=1e-12)
    rec1 = m_default.recover_item_params(it, rp)
    rec2 = m_none.recover_item_params(it, rp)
    assert torch.allclose(torch.tensor(rec1["a"]), torch.tensor(rec2["a"]),
                          atol=1e-12)


# ---------------------------------------------------------------------------
# Contract 2: the exp link is exactly exp(s * raw) on the static head
# ---------------------------------------------------------------------------

def test_exp_link_is_exact_on_static_head():
    s = 0.3
    dec = GPCMDecoder(emb_dim=4, n_cats=4, alpha_log_scale=s)
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(7, 4, generator=g)
    raw = dec.fc_a(emb)                       # (7, 1) raw discrimination
    a = dec.item_params(emb)["a"]             # static path: state=None
    assert torch.allclose(a, torch.exp(s * raw), atol=1e-7)
    # and NOT softplus
    assert not torch.allclose(a, F.softplus(raw), atol=1e-4)


def test_softplus_default_on_static_head():
    dec = GPCMDecoder(emb_dim=4, n_cats=4, alpha_log_scale=None)
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(7, 4, generator=g)
    raw = dec.fc_a(emb)
    a = dec.item_params(emb)["a"]
    assert torch.allclose(a, F.softplus(raw), atol=1e-7)


def test_exp_link_is_exact_on_state_head():
    s = 0.3
    dec = GPCMDecoder(emb_dim=4, n_cats=4, state_dim=8, alpha_log_scale=s)
    g = torch.Generator().manual_seed(2)
    emb = torch.randn(5, 4, generator=g)
    state = torch.randn(5, 8, generator=g)
    raw = dec.fc_a_state(torch.cat([state, emb], dim=-1))
    a = dec.item_params(emb, state=state)["a"]
    assert torch.allclose(a, torch.exp(s * raw), atol=1e-7)


# ---------------------------------------------------------------------------
# Contract 3: the exp link is a genuinely different model
# ---------------------------------------------------------------------------

def test_exp_link_changes_training():
    it, rp = _data()
    common = dict(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                  decoder="gpcm", state_alpha=True, device=_DEVICE, seed=_SEED)
    m_soft = DeepIRTModel(alpha_log_scale=None, **common)
    m_exp = DeepIRTModel(alpha_log_scale=0.3, **common)
    r_soft = m_soft.fit(it, rp, n_epochs=20, verbose=False)
    r_exp = m_exp.fit(it, rp, n_epochs=20, verbose=False)
    assert not math.isclose(r_soft["final_nll"], r_exp["final_nll"], rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Contract 4: validation
# ---------------------------------------------------------------------------

def test_nonpositive_log_scale_raises():
    for bad in (0.0, -0.3):
        try:
            GPCMDecoder(emb_dim=4, n_cats=4, alpha_log_scale=bad)
        except ValueError as e:
            assert "alpha_log_scale" in str(e)
        else:
            raise AssertionError(f"expected ValueError for alpha_log_scale={bad}")


def test_link_is_gpcm_binary_only():
    try:
        DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                       decoder="nrm", alpha_log_scale=0.3)
    except ValueError as e:
        assert "alpha_log_scale" in str(e)
    else:
        raise AssertionError("expected ValueError for alpha_log_scale on nrm")


def test_binary_decoder_threads_link():
    s = 0.3
    dec = Binary2PLDecoder(emb_dim=4, alpha_log_scale=s)
    g = torch.Generator().manual_seed(3)
    emb = torch.randn(6, 4, generator=g)
    raw = dec._gpcm.fc_a(emb)
    a = dec.item_params(emb)["a"]
    assert torch.allclose(a, torch.exp(s * raw), atol=1e-7)


# ---------------------------------------------------------------------------
# Contract 5: the exp link composes with the decoupled wide item key
# ---------------------------------------------------------------------------

def test_exp_link_composes_with_decoupled_key():
    it, rp = _data(num_items=10)
    m = DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                       decoder="gpcm", state_alpha=True, item_key_dim=16,
                       alpha_log_scale=0.3, device=_DEVICE, seed=_SEED)
    r = m.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isfinite(r["final_nll"])
    rec = m.recover_item_params(it, rp)
    assert rec["a"].shape == (10,)
    assert (rec["a"] > 0).all()        # exp link is strictly positive


# ---------------------------------------------------------------------------
# Stage B contract: decoupling is the DEFAULT (pins the flip so it cannot regress)
# ---------------------------------------------------------------------------

def _bare(decoder="gpcm", **kw):
    return DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                        decoder=decoder, device=_DEVICE, seed=_SEED, **kw)


def test_decouple_is_the_default():
    # bare gpcm -> the decoupled deep-irt s_0 (state_alpha + own wide item key + exp)
    m = _bare()
    assert m.decouple is True
    assert m.state_alpha is True
    assert m.item_key_dim == 64
    assert m.alpha_log_scale == 1.0
    assert hasattr(m.encoder, "item_key_emb")


def test_decouple_false_is_the_plain_path():
    m = _bare(decouple=False)
    assert m.state_alpha is False
    assert m.item_key_dim is None and m.alpha_log_scale is None
    assert not hasattr(m.encoder, "item_key_emb")


def test_explicit_alpha_knobs_defer_decouple():
    # explicit state_alpha=True (no item key) -> plain state_alpha, NOT auto-decoupled
    ms = _bare(state_alpha=True)
    assert ms.state_alpha is True and ms.item_key_dim is None
    assert not hasattr(ms.encoder, "item_key_emb")
    # explicit state_alpha=False is respected (the None-sentinel, not auto-decoupled)
    mf = _bare(state_alpha=False)
    assert mf.state_alpha is False and mf.item_key_dim is None


def test_decouple_is_noop_for_non_alpha_decoders():
    # nrm/bt have no alpha head; default decouple must be a no-op, not a raise
    for dec in ("nrm", "bt"):
        m = _bare(decoder=dec)
        assert m.state_alpha is False and m.item_key_dim is None
