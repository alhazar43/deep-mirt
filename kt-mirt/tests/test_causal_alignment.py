"""Tests for the single-shift causal alignment contract and the organic
state-conditioned alpha engine.

ONE convention, pinned here
---------------------------
The theta/state used to predict the response at step t is a function of the
interaction history STRICTLY BEFORE t -- (q_{<t}, r_{<t}) INCLUDING the
immediately preceding interaction (q_{t-1}, r_{t-1}) -- and never of r_t.
Exactly one shift, total.  This is the alignment ma-irt's encoders use.

Contracts
---------
1. ``theta_for_prediction`` / ``state_for_prediction`` are exactly one shift
   (equal to shifting the raw responsive theta once).
2. The step-t prediction theta does NOT see r_t (causal) but DOES depend on
   interaction t-1.
3. ``state_alpha=False`` is bit-for-bit identical to the static decoder.
4. ``state_alpha=True`` builds the state head, trains, reports a responsive
   (direct) theta track, reads an occurrence-averaged state-conditioned alpha,
   and costs only the small state head over the static model.
"""

from __future__ import annotations

import math

import torch

from kt_mirt.core import DeepIRTModel
from kt_mirt.core.encoder import LSTMEncoder


_SEED = 42
_DEVICE = torch.device("cpu")


def _data(num_items=10, n_cats=4, n=16, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    it = torch.randint(0, num_items, (n, t), generator=g)
    rp = torch.randint(0, n_cats, (n, t), generator=g)
    return it, rp


def _enc(num_items=10, n_cats=4, seed=_SEED):
    torch.manual_seed(seed)
    return LSTMEncoder(num_items=num_items, emb_dim=4, hidden_dim=8,
                       n_cats=n_cats)


# ---------------------------------------------------------------------------
# Contract 1: exactly one shift, realised correctly per path
# ---------------------------------------------------------------------------

def test_direct_prediction_is_one_shift_of_raw_theta():
    """DIRECT path: theta_for_prediction == one right-shift of the raw theta.

    This is the legacy convention (literal 0 prior at t=0), kept exactly so the
    static engine stays bit-identical.
    """
    enc = _enc()
    it, rp = _data()
    with torch.no_grad():
        raw = enc.encode(it, rp)                       # (B, T) responsive
        pred = enc.theta_for_prediction(it, rp)        # (B, T) aligned
        manual = LSTMEncoder._shift(raw)
    assert torch.allclose(pred, manual, atol=1e-7)


# ---------------------------------------------------------------------------
# Contract 2: causal + keeps interaction t-1
# ---------------------------------------------------------------------------

def test_prediction_theta_does_not_see_current_response():
    """The step-t prediction theta must be invariant to r_t (strict causality)."""
    enc = _enc()
    it, rp = _data()
    rp2 = rp.clone()
    rp2[:, -1] = (rp2[:, -1] + 1) % enc.n_cats     # perturb ONLY r_T
    with torch.no_grad():
        a = enc.theta_for_prediction(it, rp)[:, -1]
        b = enc.theta_for_prediction(it, rp2)[:, -1]
    assert torch.allclose(a, b, atol=1e-7)


def test_direct_prediction_keeps_previous_interaction():
    """The step-T prediction theta depends on (q_{T-1}, r_{T-1}) but not on r_T."""
    enc = _enc()
    it, rp = _data()
    rp_prev = rp.clone()
    rp_prev[:, -2] = (rp_prev[:, -2] + 1) % enc.n_cats
    with torch.no_grad():
        base = enc.theta_for_prediction(it, rp)[:, -1]
        moved = enc.theta_for_prediction(it, rp_prev)[:, -1]
    assert not torch.allclose(base, moved, atol=1e-6)


def test_state_and_theta_share_the_same_aligned_hidden():
    """theta and state come from ONE pass and agree at t>=1: theta[t] ==
    theta_proj(state[t]).  At t=0 the direct path keeps the legacy theta prior
    (0) while the state prior is the zero vector, so they may differ only there.
    """
    enc = _enc()
    it, rp = _data()
    with torch.no_grad():
        theta, state = enc.aligned_theta_and_state(it, rp)
        from_state = enc.theta_proj(state).squeeze(-1)
    # t >= 1 must agree exactly (same shifted hidden).
    assert torch.allclose(theta[:, 1:], from_state[:, 1:], atol=1e-7)
    # Direct: theta[0] == 0 (legacy prior); state[0] == zero vector.
    assert torch.allclose(theta[:, 0], torch.zeros_like(theta[:, 0]))
    assert torch.allclose(state[:, 0], torch.zeros_like(state[:, 0]))


# ---------------------------------------------------------------------------
# Contract 3: state_alpha OFF is bit-for-bit identical
# ---------------------------------------------------------------------------

def _model(num_items=10, n_cats=4, state_alpha=False, decoder="gpcm", seed=_SEED):
    return DeepIRTModel(
        num_items=num_items, emb_dim=4, hidden_dim=8, n_cats=n_cats,
        decoder=decoder, state_alpha=state_alpha, device=_DEVICE, seed=seed,
    )


def test_state_alpha_off_is_bit_identical():
    """state_alpha=False must match the plain path (decouple=False): init
    weights, training NLL, and the tracked theta trajectory.  (Decoupling is now
    the DEFAULT, so the plain baseline is requested explicitly.)"""
    it, rp = _data()
    m_default = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=4, decoder="gpcm",
        decouple=False, device=_DEVICE, seed=_SEED,
    )
    m_false = _model(state_alpha=False, seed=_SEED)

    for (n1, p1), (n2, p2) in zip(
        m_default.encoder.named_parameters(), m_false.encoder.named_parameters()
    ):
        assert n1 == n2 and torch.allclose(p1, p2), n1
    for (n1, p1), (n2, p2) in zip(
        m_default.decoder.named_parameters(), m_false.decoder.named_parameters()
    ):
        assert n1 == n2 and torch.allclose(p1, p2), n1

    r_def = m_default.fit(it, rp, n_epochs=15, verbose=False)
    r_false = m_false.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isclose(r_def["final_nll"], r_false["final_nll"], rel_tol=1e-12)
    assert torch.allclose(m_default.track(it, rp), m_false.track(it, rp), atol=1e-12)


def test_state_alpha_off_has_no_state_head():
    m_off = _model(state_alpha=False)
    assert not hasattr(m_off.encoder, "interaction_start")
    assert getattr(m_off.decoder, "state_dim", None) is None
    assert not hasattr(m_off.decoder, "fc_a_state")

    m_on = _model(state_alpha=True)
    # state_alpha alone does NOT need the item-blind start token (it rides the
    # direct stream); the decoder gets the state head.
    assert m_on.decoder.state_dim == m_on.hidden_dim
    assert hasattr(m_on.decoder, "fc_a_state")


# ---------------------------------------------------------------------------
# Contract 4: the organic state_alpha engine
# ---------------------------------------------------------------------------

def test_state_alpha_theta_track_is_responsive():
    """The reported dynamic theta is the direct responsive stream: perturbing
    the final item id moves the final tracked theta."""
    it, rp = _data()
    num_items = 10
    m = _model(num_items=num_items, state_alpha=True)
    m.fit(it, rp, n_epochs=10, verbose=False)
    it2 = it.clone()
    it2[:, -1] = (it2[:, -1] + 1) % num_items
    with torch.no_grad():
        a = m.track(it, rp)[:, -1]
        b = m.track(it2, rp)[:, -1]
    assert not torch.allclose(a, b, atol=1e-6)


def test_state_alpha_alpha_state_is_item_blind_at_current_step():
    """The discrimination state (state_for_prediction) is item-blind for the
    CURRENT step: perturbing only q_T does not change the state at T (it
    conditions on history before answering T)."""
    it, rp = _data()
    num_items = 10
    m = _model(num_items=num_items, state_alpha=True)
    m.fit(it, rp, n_epochs=10, verbose=False)
    it2 = it.clone()
    it2[:, -1] = (it2[:, -1] + 1) % num_items
    with torch.no_grad():
        sa = m.encoder.state_for_prediction(it, rp)[:, -1]
        sb = m.encoder.state_for_prediction(it2, rp)[:, -1]
    assert torch.allclose(sa, sb, atol=1e-7)


def test_state_alpha_recovery_shapes_and_variation():
    it, rp = _data(num_items=10, n_cats=4, n=24, t=20)
    m = _model(num_items=10, n_cats=4, state_alpha=True)
    m.fit(it, rp, n_epochs=20, verbose=False)
    rec = m.recover_item_params(it, rp)
    assert rec["alpha"].shape == (10,)
    assert rec["beta"].shape == (10, 3)
    assert rec["seen"].shape == (10,)
    assert rec["seen"].all()
    assert float(rec["alpha"].std()) > 0.0


def test_state_alpha_recovery_requires_sequences():
    m = _model(state_alpha=True)
    try:
        m.recover_item_params()          # no sequences
    except ValueError as e:
        assert "recover_item_params(item_ids, responses)" in str(e)
    else:
        raise AssertionError("expected ValueError without sequences")


def test_state_alpha_rejects_non_gpcm_decoders():
    for dec in ("bt", "nrm"):
        try:
            DeepIRTModel(num_items=10, emb_dim=4, hidden_dim=8, n_cats=4,
                           decoder=dec, state_alpha=True)
        except ValueError as e:
            assert "state_alpha" in str(e)
        else:
            raise AssertionError(f"expected ValueError for state_alpha + {dec}")


def test_state_alpha_param_cost_is_small():
    """state_alpha adds only the decoder's fc_a_state head over the static
    model; the encoder is unchanged (no start token, single pass)."""
    def npar(m):
        return (sum(p.numel() for p in m.encoder.parameters())
                + sum(p.numel() for p in m.decoder.parameters()))
    m_static = _model(state_alpha=False)
    m_state = _model(state_alpha=True)
    extra = npar(m_state) - npar(m_static)
    emb_dim, hidden_dim = m_static.emb_dim, m_static.hidden_dim
    # fc_a_state: (hidden+emb)->1 weight + bias.  No encoder start token.
    expected = (hidden_dim + emb_dim) * 1 + 1
    assert extra == expected, (extra, expected)
