"""
Tests for the encoder's separate_theta toggle (item-blind double pass).

The toggle ports ma-irt's separated ability pathway into the deep_irt's LSTM
encoder.  Two contracts must hold:

1. separate_theta=False is BIT-FOR-BIT identical to the pre-toggle encoder
   (no new parameters constructed, no RNG drift, same NLL + theta).
2. separate_theta=True reads theta from an item-blind stream: the per-step
   theta is invariant to the identity of the CURRENT step's item, so it cannot
   absorb the item it is about to be scored against.  It must still depend on
   the response history (it is not a constant).
"""

from __future__ import annotations

import math

import torch

from deep_irt.core import DeepIRTModel


_SEED = 42
_DEVICE = torch.device("cpu")


def _data(num_items=10, n_cats=4, n=16, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    it = torch.randint(0, num_items, (n, t), generator=g)
    rp = torch.randint(0, n_cats, (n, t), generator=g)
    return it, rp


def _model(num_items=10, n_cats=4, separate_theta=False, seed=_SEED):
    return DeepIRTModel(
        num_items=num_items, emb_dim=4, hidden_dim=8, n_cats=n_cats,
        decoder="gpcm", separate_theta=separate_theta, device=_DEVICE, seed=seed,
    )


def test_direct_path_is_bit_identical():
    """separate_theta=False must match a model that omits the flag entirely."""
    it, rp = _data()

    m_default = DeepIRTModel(
        num_items=10, emb_dim=4, hidden_dim=8, n_cats=4, decoder="gpcm",
        device=_DEVICE, seed=_SEED,
    )
    m_false = _model(separate_theta=False, seed=_SEED)

    # Identical init (the new modules are not even constructed when False).
    for (n1, p1), (n2, p2) in zip(
        m_default.encoder.named_parameters(), m_false.encoder.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2), n1

    r_def = m_default.fit(it, rp, n_epochs=15, verbose=False)
    r_false = m_false.fit(it, rp, n_epochs=15, verbose=False)
    assert math.isclose(r_def["final_nll"], r_false["final_nll"], rel_tol=1e-9)

    assert torch.allclose(
        m_default.track(it, rp), m_false.track(it, rp), atol=1e-9
    )


def test_separate_path_trains_and_has_distinct_objective():
    """The item-blind stream is a genuinely different model (different NLL)."""
    it, rp = _data()
    m_dir = _model(separate_theta=False, seed=_SEED)
    m_sep = _model(separate_theta=True, seed=_SEED)
    nll_dir = m_dir.fit(it, rp, n_epochs=15, verbose=False)["final_nll"]
    nll_sep = m_sep.fit(it, rp, n_epochs=15, verbose=False)["final_nll"]
    assert not math.isclose(nll_dir, nll_sep, rel_tol=1e-6)
    assert m_sep.track(it, rp).shape == (it.size(0), it.size(1))


def test_separate_theta_is_item_blind_at_current_step():
    """Changing only the current step's item id must NOT move the item-blind
    theta at that step, but MUST move the direct theta (which sees q_t)."""
    it, rp = _data()
    num_items = 10
    m_dir = _model(num_items=num_items, separate_theta=False, seed=_SEED)
    m_sep = _model(num_items=num_items, separate_theta=True, seed=_SEED)
    m_dir.fit(it, rp, n_epochs=10, verbose=False)
    m_sep.fit(it, rp, n_epochs=10, verbose=False)

    it2 = it.clone()
    it2[:, -1] = (it2[:, -1] + 1) % num_items  # perturb only the last item id

    with torch.no_grad():
        d_a = m_dir.encoder.encode(it, rp)[:, -1]
        d_b = m_dir.encoder.encode(it2, rp)[:, -1]
        s_a = m_sep.encoder.encode(it, rp)[:, -1]
        s_b = m_sep.encoder.encode(it2, rp)[:, -1]

    # Direct theta at the final step depends on the current item id.
    assert not torch.allclose(d_a, d_b, atol=1e-6)
    # Item-blind theta at the final step is invariant to the current item id.
    assert torch.allclose(s_a, s_b, atol=1e-7)


def test_separate_theta_depends_on_response_history():
    """The item-blind theta is not a constant: changing past responses moves it."""
    it, rp = _data()
    m_sep = _model(separate_theta=True, seed=_SEED)
    m_sep.fit(it, rp, n_epochs=10, verbose=False)

    rp2 = rp.clone()
    rp2[:, 0] = (rp2[:, 0] + 1) % m_sep.n_cats  # perturb the first response

    with torch.no_grad():
        t_a = m_sep.encoder.encode(it, rp)[:, -1]
        t_b = m_sep.encoder.encode(it, rp2)[:, -1]
    assert not torch.allclose(t_a, t_b, atol=1e-6)
