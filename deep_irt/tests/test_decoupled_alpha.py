"""Tests for the DECOUPLED alpha variant (alpha_emb_dim).

The decoupled variant gives discrimination its OWN wide item key -- a separate
``alpha_item_emb`` table feeding ONLY the state-conditioned alpha head -- while
the theta-encoder (LSTM input, theta_proj) and beta (fc_b on item_emb) stay at
the cheap config.  This buys alpha capacity without widening the encoder that
produces theta.

Contracts pinned here
---------------------
1. ``alpha_emb_dim=None`` is bit-for-bit identical to plain ``state_alpha`` (no
   extra table, no RNG drift, same NLL + tracked theta).
2. The wide alpha key is built and wired: the encoder grows an ``alpha_item_emb``
   table of the requested width and the decoder's alpha head input is
   ``hidden_dim + alpha_emb_dim`` (not ``hidden_dim + emb_dim``).
3. Decoupling is real: the LSTM input embedding (``item_emb``) and theta are
   driven by the CHEAP table only; perturbing the wide alpha table never moves
   theta but DOES move the recovered discrimination.
4. ``alpha_emb_dim`` requires ``state_alpha`` (the key only feeds that head).
5. Recovery works: occurrence-averaged alpha over the wide key, shapes correct.
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


def _model(num_items=10, n_cats=4, state_alpha=True, alpha_emb_dim=None,
           emb_dim=4, hidden_dim=8, seed=_SEED):
    return DeepIRTModel(
        num_items=num_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
        n_cats=n_cats, decoder="gpcm", state_alpha=state_alpha,
        alpha_emb_dim=alpha_emb_dim, device=_DEVICE, seed=seed,
    )


# ---------------------------------------------------------------------------
# Contract 1: alpha_emb_dim=None is bit-for-bit identical to plain state_alpha
# ---------------------------------------------------------------------------

def test_alpha_emb_none_is_bit_identical():
    it, rp = _data()
    m_plain = _model(state_alpha=True, alpha_emb_dim=None, seed=_SEED)
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
    assert not hasattr(m_plain.encoder, "alpha_item_emb")


# ---------------------------------------------------------------------------
# Contract 2: the wide alpha key is built and wired
# ---------------------------------------------------------------------------

def test_wide_alpha_key_is_built_and_wired():
    m = _model(num_items=12, emb_dim=4, hidden_dim=8, alpha_emb_dim=16)
    assert hasattr(m.encoder, "alpha_item_emb")
    assert m.encoder.alpha_item_emb.weight.shape == (12, 16)
    # alpha head reads [state(hidden), alpha_key(16)] not [state, item_emb(4)].
    assert m.decoder.fc_a_state.in_features == 8 + 16
    # beta still reads the cheap item_emb.
    assert m.decoder.fc_b.in_features == 4
    # the LSTM input is still the cheap 2*emb_dim.
    assert m.encoder.lstm.input_size == 2 * 4


# ---------------------------------------------------------------------------
# Contract 3: decoupling is real -- the wide key drives alpha, never theta
# ---------------------------------------------------------------------------

def test_wide_key_moves_alpha_not_theta():
    it, rp = _data(num_items=10)
    m = _model(num_items=10, alpha_emb_dim=16)
    m.fit(it, rp, n_epochs=20, verbose=False)

    theta_before = m.track(it, rp).clone()
    rec_before = m.recover_item_params(it, rp)

    # Perturb ONLY the wide alpha table.  Theta (driven by the cheap item_emb +
    # LSTM) must be invariant; recovered discrimination must move.
    with torch.no_grad():
        m.encoder.alpha_item_emb.weight.add_(
            torch.randn_like(m.encoder.alpha_item_emb.weight))
    theta_after = m.track(it, rp)
    rec_after = m.recover_item_params(it, rp)

    assert torch.allclose(theta_before, theta_after, atol=1e-7), \
        "theta moved when only the alpha key changed"
    assert not torch.allclose(
        torch.tensor(rec_before["a"]), torch.tensor(rec_after["a"]), atol=1e-4), \
        "discrimination did not move when the alpha key changed"


def test_cheap_item_emb_drives_theta():
    """Perturbing the cheap item_emb moves theta; this is the encoder table."""
    it, rp = _data(num_items=10)
    m = _model(num_items=10, alpha_emb_dim=16)
    m.fit(it, rp, n_epochs=15, verbose=False)
    theta_before = m.track(it, rp).clone()
    with torch.no_grad():
        m.encoder.item_emb.weight.add_(
            torch.randn_like(m.encoder.item_emb.weight))
    theta_after = m.track(it, rp)
    assert not torch.allclose(theta_before, theta_after, atol=1e-5)


# ---------------------------------------------------------------------------
# Contract 4: alpha_emb_dim requires state_alpha
# ---------------------------------------------------------------------------

def test_alpha_emb_dim_requires_state_alpha():
    try:
        DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                       decoder="gpcm", state_alpha=False, alpha_emb_dim=16)
    except ValueError as e:
        assert "state_alpha" in str(e)
    else:
        raise AssertionError("expected ValueError for alpha_emb_dim w/o state_alpha")


# ---------------------------------------------------------------------------
# Contract 5: recovery works (occurrence-averaged over the wide key)
# ---------------------------------------------------------------------------

def test_decoupled_recovery_shapes_and_variation():
    it, rp = _data(num_items=10, n_cats=4, n=24, t=20)
    m = _model(num_items=10, n_cats=4, alpha_emb_dim=16)
    m.fit(it, rp, n_epochs=25, verbose=False)
    rec = m.recover_item_params(it, rp)
    assert rec["a"].shape == (10,)
    assert rec["b"].shape == (10, 3)
    assert rec["seen"].shape == (10,) and rec["seen"].all()
    assert float(rec["a"].std()) > 0.0


def test_decoupled_trains_and_predicts():
    it, rp = _data(num_items=10)
    m = _model(num_items=10, alpha_emb_dim=16)
    r = m.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isfinite(r["final_nll"])
    # different objective from the cheap baseline (a genuinely different model)
    m_cheap = _model(num_items=10, alpha_emb_dim=None)
    r_cheap = m_cheap.fit(it, rp, n_epochs=15, verbose=False)
    assert not math.isclose(r["final_nll"], r_cheap["final_nll"], rel_tol=1e-6)


# ---------------------------------------------------------------------------
# beta_wide: step thresholds read the wide alpha key (ma-irt's shared-key beta)
# ---------------------------------------------------------------------------

def _model_bw(num_items=10, n_cats=4, alpha_emb_dim=16, beta_wide=False,
              emb_dim=4, hidden_dim=8, seed=_SEED):
    return DeepIRTModel(
        num_items=num_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
        n_cats=n_cats, decoder="gpcm", state_alpha=True,
        alpha_emb_dim=alpha_emb_dim, beta_wide=beta_wide,
        device=_DEVICE, seed=seed,
    )


def test_beta_wide_requires_alpha_emb_dim():
    try:
        DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                     decoder="gpcm", state_alpha=True, alpha_emb_dim=None,
                     beta_wide=True)
    except ValueError as e:
        assert "beta_wide" in str(e) and "alpha_emb_dim" in str(e)
    else:
        raise AssertionError("expected ValueError for beta_wide w/o alpha_emb_dim")


def test_beta_wide_wires_fc_b_to_wide_key():
    m = _model_bw(num_items=12, emb_dim=4, hidden_dim=8, alpha_emb_dim=16,
                  beta_wide=True)
    # fc_b now reads the wide alpha key (16), not the narrow item_emb (4).
    assert m.decoder.fc_b.in_features == 16
    # alpha head still reads [state(8), alpha_key(16)].
    assert m.decoder.fc_a_state.in_features == 8 + 16
    # LSTM input still cheap.
    assert m.encoder.lstm.input_size == 2 * 4


def test_beta_wide_false_keeps_narrow_fc_b():
    m = _model_bw(num_items=12, emb_dim=4, hidden_dim=8, alpha_emb_dim=16,
                  beta_wide=False)
    assert m.decoder.fc_b.in_features == 4   # narrow item_emb read, unchanged


def test_beta_wide_key_moves_beta():
    """Perturbing the wide alpha key moves recovered beta under beta_wide."""
    it, rp = _data(num_items=10, n=24, t=20)
    m = _model_bw(num_items=10, alpha_emb_dim=16, beta_wide=True)
    m.fit(it, rp, n_epochs=20, verbose=False)
    rec_before = m.recover_item_params(it, rp)
    with torch.no_grad():
        m.encoder.alpha_item_emb.weight.add_(
            torch.randn_like(m.encoder.alpha_item_emb.weight))
    rec_after = m.recover_item_params(it, rp)
    assert not torch.allclose(
        torch.tensor(rec_before["b"]), torch.tensor(rec_after["b"]), atol=1e-4), \
        "beta did not move when the wide alpha key changed (beta_wide off?)"


def test_beta_wide_recovery_shapes():
    it, rp = _data(num_items=10, n_cats=4, n=24, t=20)
    m = _model_bw(num_items=10, n_cats=4, alpha_emb_dim=16, beta_wide=True)
    m.fit(it, rp, n_epochs=20, verbose=False)
    rec = m.recover_item_params(it, rp)
    assert rec["a"].shape == (10,)
    assert rec["b"].shape == (10, 3)
    assert rec["seen"].shape == (10,) and rec["seen"].all()


def test_beta_wide_changes_objective():
    """beta_wide is a genuinely different model from narrow-beta decoupled."""
    it, rp = _data(num_items=10, n=24, t=20)
    m_wide = _model_bw(num_items=10, alpha_emb_dim=16, beta_wide=True)
    m_narrow = _model_bw(num_items=10, alpha_emb_dim=16, beta_wide=False)
    r_wide = m_wide.fit(it, rp, n_epochs=15, verbose=False)
    r_narrow = m_narrow.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isfinite(r_wide["final_nll"])
    assert not math.isclose(r_wide["final_nll"], r_narrow["final_nll"],
                            rel_tol=1e-6)


def test_beta_wide_binary_decoder():
    """beta_wide threads through the binary (K=2) decoder too."""
    it, rp = _data(num_items=10, n_cats=2, n=20, t=15)
    m = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=2, decoder="binary",
        state_alpha=True, alpha_emb_dim=16, beta_wide=True,
        device=_DEVICE, seed=_SEED,
    )
    assert m.decoder._gpcm.fc_b.in_features == 16
    r = m.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isfinite(r["final_nll"])
    rec = m.recover_item_params(it, rp)
    assert rec["b"].shape == (10, 1)
