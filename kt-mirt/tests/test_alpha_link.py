"""Tests for the discrimination map (``alpha_log_scale`` / ``alpha_pos_map``).

The GPCM/binary decoder pushes the raw discrimination-head output through a
map, historically one of two transforms:

  softplus (default, ``alpha_log_scale=None``) -- the original transform.
  exponential (``alpha_log_scale=s``)          -- ``exp(s * raw)``, ma-irt's
      transform; with ``s = 1.0`` it is plain ``exp(raw)``.

The named ``alpha_pos_map`` API extends this for the E5 positivity-geometry
ablation while preserving the old ``alpha_log_scale`` path exactly.

Contracts pinned here
---------------------
1. ``alpha_log_scale=None`` is bit-for-bit identical to before (softplus): same
   weights, same trained NLL, same recovered alpha.
2. The exp transform is exactly ``exp(s * raw)`` on the static head.
3. The exp transform changes the model (different trained NLL from softplus) --
   it is a genuinely different positivity map, not a no-op.
4. Validation: ``alpha_log_scale <= 0`` raises; the transform is GPCM/binary-only.
5. The exp transform composes with the decoupled wide item key.
6. Named alpha maps implement their exact formulas and thread through the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kt_mirt.core import DeepIRTModel
from kt_mirt.core.decoders import GPCMDecoder, Binary2PLDecoder


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
    assert torch.allclose(torch.tensor(rec1["alpha"]), torch.tensor(rec2["alpha"]),
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
    a = dec.item_params(emb)["alpha"]             # static path: state=None
    assert torch.allclose(a, torch.exp(s * raw), atol=1e-7)
    # and NOT softplus
    assert not torch.allclose(a, F.softplus(raw), atol=1e-4)


def test_softplus_default_on_static_head():
    dec = GPCMDecoder(emb_dim=4, n_cats=4, alpha_log_scale=None)
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(7, 4, generator=g)
    raw = dec.fc_a(emb)
    a = dec.item_params(emb)["alpha"]
    assert torch.allclose(a, F.softplus(raw), atol=1e-7)


def test_named_alpha_maps_are_exact():
    raw = torch.tensor([[-2.0], [-0.5], [0.0], [0.5], [2.0]])
    cases = [
        ("identity", {}, raw),
        ("raw", {}, raw),
        ("relu", {}, F.relu(raw)),
        ("softplus", {}, F.softplus(raw)),
        ("softplus_eps", {"epsilon": 0.2}, F.softplus(raw) + 0.2),
        ("scaled_softplus", {"scale": 2.5}, 2.5 * F.softplus(raw)),
        ("temperature_softplus", {"temperature": 0.3}, F.softplus(0.3 * raw)),
        ("sigmoid", {}, torch.sigmoid(raw)),
        ("scaled_sigmoid", {"scale": 3.0}, 3.0 * torch.sigmoid(raw)),
        ("square", {"epsilon": 0.1}, raw.square() + 0.1),
        ("exp", {"scale": 0.3}, torch.exp(0.3 * raw)),
        (
            "clipped_exp",
            {"scale": 2.0, "clip_min": -1.0, "clip_max": 1.0},
            torch.exp(torch.clamp(2.0 * raw, min=-1.0, max=1.0)),
        ),
        (
            "clipped_identity",
            {"clip_min": -0.7, "clip_max": 1.1},
            torch.clamp(raw, min=-0.7, max=1.1),
        ),
        (
            "clipped_relu",
            {"clip_min": 0.0, "clip_max": 1.1},
            torch.clamp(F.relu(raw), min=0.0, max=1.1),
        ),
    ]
    for name, kwargs, expected in cases:
        dec = GPCMDecoder(
            emb_dim=4, n_cats=4,
            alpha_pos_map=name, alpha_pos_kwargs=kwargs,
        )
        actual = dec._alpha_pos(raw)
        assert torch.allclose(actual, expected, atol=1e-7), name


def test_named_exp_map_uses_scale_kwarg():
    raw = torch.tensor([[-2.0], [0.0], [2.0]])
    dec = GPCMDecoder(
        emb_dim=4, n_cats=4,
        alpha_pos_map="exp", alpha_pos_kwargs={"scale": 0.4},
    )
    assert torch.allclose(dec._alpha_pos(raw), torch.exp(0.4 * raw), atol=1e-7)


def test_exp_link_is_exact_on_state_head():
    s = 0.3
    dec = GPCMDecoder(emb_dim=4, n_cats=4, state_dim=8, alpha_log_scale=s)
    g = torch.Generator().manual_seed(2)
    emb = torch.randn(5, 4, generator=g)
    state = torch.randn(5, 8, generator=g)
    raw = dec.fc_a_state(torch.cat([state, emb], dim=-1))
    a = dec.item_params(emb, state=state)["alpha"]
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


def test_bad_named_map_config_raises():
    bad_configs = [
        dict(alpha_pos_map="unknown"),
        dict(alpha_pos_map="scaled_softplus", alpha_pos_kwargs={"scale": 0.0}),
        dict(alpha_pos_map="softplus", alpha_pos_kwargs={"typo": 1.0}),
        dict(alpha_pos_map="temperature_softplus",
             alpha_pos_kwargs={"temperature": -1.0}),
        dict(alpha_pos_map="square", alpha_pos_kwargs={"epsilon": 0.0}),
        dict(alpha_pos_map="clipped_exp",
             alpha_pos_kwargs={"clip_min": 2.0, "clip_max": 1.0}),
        dict(alpha_pos_map="clipped_identity",
             alpha_pos_kwargs={"clip_min": 2.0, "clip_max": 1.0}),
        dict(alpha_pos_map="clipped_relu",
             alpha_pos_kwargs={"clip_min": 2.0, "clip_max": 1.0}),
    ]
    for kw in bad_configs:
        try:
            GPCMDecoder(emb_dim=4, n_cats=4, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kw}")


def test_named_map_rejects_alpha_log_scale_mix():
    try:
        GPCMDecoder(
            emb_dim=4, n_cats=4,
            alpha_pos_map="exp", alpha_log_scale=0.4,
        )
    except ValueError as e:
        assert "alpha_log_scale" in str(e)
    else:
        raise AssertionError("expected ValueError when mixing alpha map APIs")


def test_link_is_gpcm_binary_only():
    try:
        DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                       decoder="nrm", alpha_log_scale=0.3)
    except ValueError as e:
        assert "alpha_log_scale" in str(e)
    else:
        raise AssertionError("expected ValueError for alpha_log_scale on nrm")


def test_named_map_is_gpcm_binary_only():
    try:
        DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                       decoder="nrm", alpha_pos_map="softplus")
    except ValueError as e:
        assert "alpha_pos_map" in str(e)
    else:
        raise AssertionError("expected ValueError for alpha_pos_map on nrm")


def test_binary_decoder_threads_link():
    s = 0.3
    dec = Binary2PLDecoder(emb_dim=4, alpha_log_scale=s)
    g = torch.Generator().manual_seed(3)
    emb = torch.randn(6, 4, generator=g)
    raw = dec._gpcm.fc_a(emb)
    a = dec.item_params(emb)["alpha"]
    assert torch.allclose(a, torch.exp(s * raw), atol=1e-7)


def test_binary_decoder_threads_named_map():
    dec = Binary2PLDecoder(
        emb_dim=4,
        alpha_pos_map="scaled_softplus",
        alpha_pos_kwargs={"scale": 2.0},
    )
    g = torch.Generator().manual_seed(4)
    emb = torch.randn(6, 4, generator=g)
    raw = dec._gpcm.fc_a(emb)
    a = dec.item_params(emb)["alpha"]
    assert torch.allclose(a, 2.0 * F.softplus(raw), atol=1e-7)


def test_deepirt_model_threads_named_map():
    it, rp = _data()
    m = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
        decoder="gpcm", state_alpha=True,
        alpha_pos_map="scaled_sigmoid",
        alpha_pos_kwargs={"scale": 2.0},
        device=_DEVICE, seed=_SEED,
    )
    assert m.alpha_pos_map == "scaled_sigmoid"
    assert m.decoder.alpha_pos_map == "scaled_sigmoid"
    r = m.fit(it, rp, n_epochs=10, verbose=False)
    assert math.isfinite(r["final_nll"])
    rec = m.recover_item_params(it, rp)
    assert rec["alpha"].shape == (10,)


def test_grad_clip_norm_validation():
    it, rp = _data()
    m = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
        decoder="gpcm", state_alpha=True,
        device=_DEVICE, seed=_SEED,
    )
    try:
        m.fit(it, rp, n_epochs=1, verbose=False, grad_clip_norm=0.0)
    except ValueError as e:
        assert "grad_clip_norm" in str(e)
    else:
        raise AssertionError("expected ValueError for non-positive grad_clip_norm")


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
    assert rec["alpha"].shape == (10,)
    assert (rec["alpha"] > 0).all()        # exp link is strictly positive


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
