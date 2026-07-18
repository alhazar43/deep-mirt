"""Direct unit and integration tests for kt_mirt.core.anchor (anchored
item-bank extension).

None of the ported kt-irt tests call anchored_extend / build_extended_encoder
directly, or exercise DeepIRTModel.extend() end-to-end -- the source suite
does not unit-test this module at all. This file closes that gap with small
synthetic fixtures.
"""

from __future__ import annotations

import math

import torch

from kt_mirt.core import DeepIRTModel
from kt_mirt.core.anchor import anchored_extend, build_extended_encoder
from kt_mirt.core.decoders import GPCMDecoder
from kt_mirt.core.encoder import LSTMEncoder


_DEVICE = torch.device("cpu")


def _base_encoder(num_items=6, emb_dim=4, hidden_dim=8, n_cats=4, seed=0):
    torch.manual_seed(seed)
    return LSTMEncoder(num_items=num_items, emb_dim=emb_dim, hidden_dim=hidden_dim,
                       n_cats=n_cats)


def _ext_data(n_anchor=5, E=3, n_cats=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    anchor_theta = torch.randn(n_anchor, generator=g)
    ext_item_ids = torch.randint(0, E, (n_anchor, E), generator=g)
    ext_responses = torch.randint(0, n_cats, (n_anchor, E), generator=g)
    return anchor_theta, ext_item_ids, ext_responses


# ---------------------------------------------------------------------------
# anchored_extend
# ---------------------------------------------------------------------------

def test_anchored_extend_shapes_and_finiteness():
    enc = _base_encoder(emb_dim=4, n_cats=4)
    dec = GPCMDecoder(emb_dim=4, n_cats=4)
    anchor_theta, ext_item_ids, ext_responses = _ext_data(n_anchor=5, E=3)

    result = anchored_extend(
        encoder=enc, decoder=dec, anchor_theta=anchor_theta,
        ext_item_ids=ext_item_ids, ext_responses=ext_responses,
        n_epochs=5, lr=1e-2, device=_DEVICE, verbose=False,
    )
    assert result["ext_emb_weight"].shape == (3, 4)
    assert result["n_params"] == 3 * 4
    assert math.isfinite(result["final_nll"])
    assert result["train_time"] >= 0.0


def test_anchored_extend_freezes_encoder_and_decoder():
    enc = _base_encoder(emb_dim=4, n_cats=4)
    dec = GPCMDecoder(emb_dim=4, n_cats=4)
    anchor_theta, ext_item_ids, ext_responses = _ext_data(n_anchor=5, E=3)

    anchored_extend(
        encoder=enc, decoder=dec, anchor_theta=anchor_theta,
        ext_item_ids=ext_item_ids, ext_responses=ext_responses,
        n_epochs=3, lr=1e-2, device=_DEVICE, verbose=False,
    )
    assert all(not p.requires_grad for p in enc.parameters())
    assert all(not p.requires_grad for p in dec.parameters())


def test_anchored_extend_touches_only_new_embeddings():
    """The base encoder's item_val_emb table must be untouched by extension."""
    enc = _base_encoder(emb_dim=4, n_cats=4)
    dec = GPCMDecoder(emb_dim=4, n_cats=4)
    before = enc.item_val_emb.weight.clone()
    anchor_theta, ext_item_ids, ext_responses = _ext_data(n_anchor=5, E=3)

    anchored_extend(
        encoder=enc, decoder=dec, anchor_theta=anchor_theta,
        ext_item_ids=ext_item_ids, ext_responses=ext_responses,
        n_epochs=5, lr=1e-2, device=_DEVICE, verbose=False,
    )
    assert torch.equal(before, enc.item_val_emb.weight)


# ---------------------------------------------------------------------------
# build_extended_encoder
# ---------------------------------------------------------------------------

def test_build_extended_encoder_shapes_and_weight_copy():
    B, E, emb_dim, hidden_dim, n_cats = 6, 3, 4, 8, 4
    base = _base_encoder(num_items=B, emb_dim=emb_dim, hidden_dim=hidden_dim,
                          n_cats=n_cats)
    ext_emb_weight = torch.randn(E, emb_dim)

    ext_enc = build_extended_encoder(base, ext_emb_weight, device=_DEVICE)

    assert ext_enc.num_items == B + E
    assert ext_enc.item_val_emb.weight.shape == (B + E, emb_dim)
    assert torch.allclose(ext_enc.item_val_emb.weight[:B], base.item_val_emb.weight)
    assert torch.allclose(ext_enc.item_val_emb.weight[B:], ext_emb_weight)
    assert torch.allclose(ext_enc.resp_emb.weight, base.resp_emb.weight)
    assert all(not p.requires_grad for p in ext_enc.parameters())


def test_build_extended_encoder_reproduces_base_theta_on_base_items():
    """On a sequence using only base-item ids, the extended encoder's theta
    must match the base encoder's (identical LSTM/theta_proj/item_val rows)."""
    B, E, emb_dim, hidden_dim, n_cats = 6, 3, 4, 8, 4
    base = _base_encoder(num_items=B, emb_dim=emb_dim, hidden_dim=hidden_dim,
                          n_cats=n_cats)
    ext_emb_weight = torch.randn(E, emb_dim)
    ext_enc = build_extended_encoder(base, ext_emb_weight, device=_DEVICE)

    g = torch.Generator().manual_seed(1)
    item_ids = torch.randint(0, B, (4, 10), generator=g)
    responses = torch.randint(0, n_cats, (4, 10), generator=g)

    base.eval()
    with torch.no_grad():
        theta_base = base.encode(item_ids, responses)
        theta_ext = ext_enc.encode(item_ids, responses)
    assert torch.allclose(theta_base, theta_ext, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: DeepIRTModel.extend()
# ---------------------------------------------------------------------------

def test_deepirtmodel_extend_end_to_end():
    num_items, n_cats = 8, 4
    g = torch.Generator().manual_seed(7)
    item_ids = torch.randint(0, num_items, (12, 10), generator=g)
    responses = torch.randint(0, n_cats, (12, 10), generator=g)

    model = DeepIRTModel(num_items=num_items, emb_dim=4, hidden_dim=8,
                          n_cats=n_cats, decoder="gpcm", decouple=False,
                          device=_DEVICE, seed=3)
    model.fit(item_ids, responses, n_epochs=5, verbose=False)

    anchor_theta = model.track(item_ids, responses)[:, -1]
    E = 3
    ext_item_ids = torch.randint(0, E, (12, E), generator=g)
    ext_responses = torch.randint(0, n_cats, (12, E), generator=g)

    result = model.extend(E, anchor_theta, ext_item_ids, ext_responses,
                           n_epochs=5, lr=1e-2, verbose=False)

    assert result["a_ext"].shape == (E,)
    assert result["b_ext"].shape == (E, n_cats - 1)
    assert (result["a_ext"] > 0).all()

    # use_extended=True now works and returns a (B+E)-item-aware theta track.
    theta_ext = model.track(item_ids, responses, use_extended=True)
    assert theta_ext.shape == item_ids.shape


def test_deepirtmodel_extend_requires_lstm_backbone():
    num_items, n_cats = 6, 4
    model = DeepIRTModel(num_items=num_items, emb_dim=4, hidden_dim=8,
                          n_cats=n_cats, decoder="gpcm", encoder="transformer",
                          decouple=False, device=_DEVICE, seed=0,
                          encoder_kwargs={"n_heads": 2, "n_layers": 1})
    try:
        model.extend(2, torch.zeros(3), torch.zeros(3, 2, dtype=torch.long),
                     torch.zeros(3, 2, dtype=torch.long), n_epochs=1, verbose=False)
    except NotImplementedError as e:
        assert "LSTM" in str(e)
    else:
        raise AssertionError("expected NotImplementedError for non-LSTM extend()")


def test_deepirtmodel_extend_requires_non_bt_decoder():
    model = DeepIRTModel(num_items=6, emb_dim=4, hidden_dim=8, n_cats=4,
                          decoder="bt", device=_DEVICE, seed=0)
    try:
        model.extend(2, torch.zeros(3), torch.zeros(3, 2, dtype=torch.long),
                     torch.zeros(3, 2, dtype=torch.long), n_epochs=1, verbose=False)
    except NotImplementedError as e:
        assert "pairwise" in str(e) or "BT" in str(e)
    else:
        raise AssertionError("expected NotImplementedError for bt decoder extend()")
