"""Tests for the DECOUPLED variant (item_key_dim).

The decoupled variant gives the static item params their OWN wide item key -- a
separate ``item_key_emb`` table feeding the state-conditioned alpha head and the
step-threshold head -- while the theta-encoder (LSTM input, theta_proj) stays on
the thin ``item_val_emb`` value.  This buys alpha/beta capacity without widening
the encoder that produces theta.

Contracts pinned here
---------------------
1. ``item_key_dim=None`` is bit-for-bit identical to plain ``state_alpha`` (no
   extra table, no RNG drift, same NLL + tracked theta).
2. The wide item key is built and wired: the encoder grows an ``item_key_emb``
   table of the requested width and the decoder's alpha head input is
   ``hidden_dim + item_key_dim`` (not ``hidden_dim + emb_dim``).
3. Decoupling is real: the LSTM input embedding (``item_val_emb``) and theta are
   driven by the CHEAP table only; perturbing the wide item key never moves theta
   but DOES move the recovered discrimination.
4. ``item_key_dim`` requires ``state_alpha`` (the key only feeds that head).
5. Recovery works: occurrence-averaged alpha over the wide key, shapes correct.
6. Beta rides the wide key by default when decoupled (ma-irt's shared-key path).
"""

from __future__ import annotations

import math

import torch

from deep_irt.core import DeepIRTModel


_SEED = 42
_DEVICE = torch.device("cpu")


def _data(num_items=10, n_cats=4, n=20, t=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    it = torch.randint(0, num_items, (n, t), generator=g)
    rp = torch.randint(0, n_cats, (n, t), generator=g)
    return it, rp


def _model(num_items=10, n_cats=4, state_alpha=True, item_key_dim=None,
           emb_dim=4, hidden_dim=8, seed=_SEED):
    return DeepIRTModel(
        num_items=num_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
        n_cats=n_cats, decoder="gpcm", state_alpha=state_alpha,
        item_key_dim=item_key_dim, device=_DEVICE, seed=seed,
    )


# ---------------------------------------------------------------------------
# Contract 1: item_key_dim=None is bit-for-bit identical to plain state_alpha
# ---------------------------------------------------------------------------

def test_item_key_none_is_bit_identical():
    it, rp = _data()
    m_plain = _model(state_alpha=True, item_key_dim=None, seed=_SEED)
    m_explicit = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=4, decoder="gpcm",
        state_alpha=True, device=_DEVICE, seed=_SEED,
    )
    for (n1, p1), (n2, p2) in zip(
        m_plain.encoder.named_parameters(),
        m_explicit.encoder.named_parameters(),
    ):
        assert n1 == n2 and torch.allclose(p1, p2), n1
    for (n1, p1), (n2, p2) in zip(
        m_plain.decoder.named_parameters(),
        m_explicit.decoder.named_parameters(),
    ):
        assert n1 == n2 and torch.allclose(p1, p2), n1

    r1 = m_plain.fit(it, rp, n_epochs=15, verbose=False)
    r2 = m_explicit.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isclose(r1["final_nll"], r2["final_nll"], rel_tol=1e-12)
    assert torch.allclose(m_plain.track(it, rp), m_explicit.track(it, rp),
                          atol=1e-12)
    assert not hasattr(m_plain.encoder, "item_key_emb")


# ---------------------------------------------------------------------------
# Contract 2: the wide item key is built and wired
# ---------------------------------------------------------------------------

def test_wide_item_key_is_built_and_wired():
    m = _model(num_items=12, emb_dim=4, hidden_dim=8, item_key_dim=16)
    assert hasattr(m.encoder, "item_key_emb")
    assert m.encoder.item_key_emb.weight.shape == (12, 16)
    # alpha head reads [state(hidden), item_key(16)] not [state, item_val(4)].
    assert m.decoder.fc_a_state.in_features == 8 + 16
    # beta rides the wide item key by default when decoupled (16, not 4).
    assert m.decoder.fc_b.in_features == 16
    # the LSTM input is still the cheap 2*emb_dim.
    assert m.encoder.lstm.input_size == 2 * 4


# ---------------------------------------------------------------------------
# Contract 3: decoupling is real -- the wide key drives alpha, never theta
# ---------------------------------------------------------------------------

def test_wide_key_moves_alpha_not_theta():
    it, rp = _data(num_items=10)
    m = _model(num_items=10, item_key_dim=16)
    m.fit(it, rp, n_epochs=20, verbose=False)

    theta_before = m.track(it, rp).clone()
    rec_before = m.recover_item_params(it, rp)

    # Perturb ONLY the wide item key.  Theta (driven by the cheap item_val_emb +
    # LSTM) must be invariant; recovered discrimination must move.
    with torch.no_grad():
        m.encoder.item_key_emb.weight.add_(
            torch.randn_like(m.encoder.item_key_emb.weight))
    theta_after = m.track(it, rp)
    rec_after = m.recover_item_params(it, rp)

    assert torch.allclose(theta_before, theta_after, atol=1e-7), \
        "theta moved when only the item key changed"
    assert not torch.allclose(
        torch.tensor(rec_before["a"]), torch.tensor(rec_after["a"]), atol=1e-4), \
        "discrimination did not move when the item key changed"


def test_cheap_item_val_drives_theta():
    """Perturbing the cheap item_val_emb moves theta; this is the encoder table."""
    it, rp = _data(num_items=10)
    m = _model(num_items=10, item_key_dim=16)
    m.fit(it, rp, n_epochs=15, verbose=False)
    theta_before = m.track(it, rp).clone()
    with torch.no_grad():
        m.encoder.item_val_emb.weight.add_(
            torch.randn_like(m.encoder.item_val_emb.weight))
    theta_after = m.track(it, rp)
    assert not torch.allclose(theta_before, theta_after, atol=1e-5)


# ---------------------------------------------------------------------------
# Contract 4: item_key_dim requires state_alpha
# ---------------------------------------------------------------------------

def test_item_key_dim_requires_state_alpha():
    try:
        DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                       decoder="gpcm", state_alpha=False, item_key_dim=16)
    except ValueError as e:
        assert "state_alpha" in str(e)
    else:
        raise AssertionError("expected ValueError for item_key_dim w/o state_alpha")


# ---------------------------------------------------------------------------
# Contract 5: recovery works (occurrence-averaged over the wide key)
# ---------------------------------------------------------------------------

def test_decoupled_recovery_shapes_and_variation():
    it, rp = _data(num_items=10, n_cats=4, n=24, t=20)
    m = _model(num_items=10, n_cats=4, item_key_dim=16)
    m.fit(it, rp, n_epochs=25, verbose=False)
    rec = m.recover_item_params(it, rp)
    assert rec["a"].shape == (10,)
    assert rec["b"].shape == (10, 3)
    assert rec["seen"].shape == (10,) and rec["seen"].all()
    assert float(rec["a"].std()) > 0.0


def test_decoupled_trains_and_predicts():
    it, rp = _data(num_items=10)
    m = _model(num_items=10, item_key_dim=16)
    r = m.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isfinite(r["final_nll"])
    # different objective from the cheap baseline (a genuinely different model)
    m_cheap = _model(num_items=10, item_key_dim=None)
    r_cheap = m_cheap.fit(it, rp, n_epochs=15, verbose=False)
    assert not math.isclose(r["final_nll"], r_cheap["final_nll"], rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Wide-key beta: when decoupled, step thresholds ride the wide item key by
# DEFAULT (ma-irt's shared-key beta path; no flag, retired beta_wide).
# ---------------------------------------------------------------------------

def test_decoupled_wires_fc_b_to_wide_key():
    """fc_b reads the wide item key whenever the key exists (decoupled)."""
    m = _model(num_items=12, emb_dim=4, hidden_dim=8, item_key_dim=16)
    # fc_b reads the wide item key (16), not the narrow item value (4).
    assert m.decoder.fc_b.in_features == 16
    # alpha head still reads [state(8), item_key(16)].
    assert m.decoder.fc_a_state.in_features == 8 + 16
    # LSTM input still cheap.
    assert m.encoder.lstm.input_size == 2 * 4


def test_non_decoupled_keeps_narrow_fc_b():
    """Without the wide key (plain state_alpha) beta stays on the thin value."""
    m = _model(num_items=12, emb_dim=4, hidden_dim=8, item_key_dim=None)
    assert m.decoder.fc_b.in_features == 4   # narrow item value read


def test_wide_key_moves_beta():
    """Perturbing the wide item key moves recovered beta in the default decoupled
    model (beta rides the wide key)."""
    it, rp = _data(num_items=10, n=24, t=20)
    m = _model(num_items=10, item_key_dim=16)
    m.fit(it, rp, n_epochs=20, verbose=False)
    rec_before = m.recover_item_params(it, rp)
    with torch.no_grad():
        m.encoder.item_key_emb.weight.add_(
            torch.randn_like(m.encoder.item_key_emb.weight))
    rec_after = m.recover_item_params(it, rp)
    assert not torch.allclose(
        torch.tensor(rec_before["b"]), torch.tensor(rec_after["b"]), atol=1e-4), \
        "beta did not move when the wide item key changed"


def test_wide_beta_recovery_shapes():
    it, rp = _data(num_items=10, n_cats=4, n=24, t=20)
    m = _model(num_items=10, n_cats=4, item_key_dim=16)
    m.fit(it, rp, n_epochs=20, verbose=False)
    rec = m.recover_item_params(it, rp)
    assert rec["a"].shape == (10,)
    assert rec["b"].shape == (10, 3)
    assert rec["seen"].shape == (10,) and rec["seen"].all()


def test_wide_beta_binary_decoder():
    """Wide-key beta threads through the binary (K=2) decoder too."""
    it, rp = _data(num_items=10, n_cats=2, n=20, t=15)
    m = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=2, decoder="binary",
        state_alpha=True, item_key_dim=16,
        device=_DEVICE, seed=_SEED,
    )
    assert m.decoder._gpcm.fc_b.in_features == 16
    r = m.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isfinite(r["final_nll"])
    rec = m.recover_item_params(it, rp)
    assert rec["b"].shape == (10, 1)
