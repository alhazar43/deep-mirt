"""Tests for the E5 alpha positive-map bench controls."""

from __future__ import annotations

import math

import pytest
import torch

from deep_irt.bench.run_alpha_map_bench import (
    _MAP_SPECS,
    raw_alpha_value_for_map,
    set_matched_alpha_init,
)
from deep_irt.core import DeepIRTModel
from deep_irt.core.decoders import GPCMDecoder


def test_raw_inverse_hits_target_for_all_runner_maps():
    target = 0.5
    for name, kwargs in _MAP_SPECS.items():
        dec = GPCMDecoder(
            emb_dim=4,
            n_cats=4,
            alpha_pos_map=name,
            alpha_pos_kwargs=kwargs,
        )
        raw = raw_alpha_value_for_map(target, name, kwargs)
        actual = dec._alpha_pos(torch.tensor([[raw]], dtype=torch.float32)).item()
        assert math.isclose(actual, target, rel_tol=1e-5, abs_tol=1e-6), name


def test_matched_init_sets_static_and_state_alpha_heads():
    target = 0.5
    model = DeepIRTModel(
        num_items=8,
        emb_dim=4,
        hidden_dim=7,
        n_cats=4,
        decoder="gpcm",
        state_alpha=True,
        item_key_dim=6,
        alpha_pos_map="scaled_softplus",
        alpha_pos_kwargs={"scale": 1.5},
        decouple=False,
        device=torch.device("cpu"),
        seed=123,
    )

    info = set_matched_alpha_init(model, target)
    dec = model.decoder
    assert info["initialized_heads"] == ["fc_a", "fc_a_state"]
    assert torch.allclose(dec.fc_a.weight, torch.zeros_like(dec.fc_a.weight))
    assert torch.allclose(
        dec.fc_a_state.weight, torch.zeros_like(dec.fc_a_state.weight)
    )

    g = torch.Generator().manual_seed(0)
    emb = torch.randn(5, dec.emb_dim, generator=g)
    state = torch.randn(5, dec.state_dim, generator=g)
    item_key = torch.randn(5, dec.item_key_dim, generator=g)

    static_params = dec.item_params(emb, item_key=item_key)
    state_params = dec.item_params(emb, state=state, item_key=item_key)
    expected = torch.full((5, 1), target)
    assert torch.allclose(static_params["alpha"], expected, atol=1e-6)
    assert torch.allclose(state_params["alpha"], expected, atol=1e-6)


def test_raw_inverse_rejects_unreachable_targets():
    with pytest.raises(ValueError):
        raw_alpha_value_for_map(1.1, "sigmoid")

    with pytest.raises(ValueError):
        raw_alpha_value_for_map(
            math.exp(6.0),
            "clipped_exp",
            {"scale": 1.0, "clip_min": -5.0, "clip_max": 5.0},
        )

    with pytest.raises(ValueError):
        raw_alpha_value_for_map(
            2.0,
            "clipped_relu",
            {"clip_min": 0.0, "clip_max": 1.0},
        )


def test_clipped_raw_inverse_hits_reachable_target():
    target = 0.5
    cases = [
        ("clipped_identity", {"clip_min": -2.0, "clip_max": 2.0}),
        ("clipped_relu", {"clip_min": 0.0, "clip_max": 2.0}),
    ]
    for name, kwargs in cases:
        dec = GPCMDecoder(
            emb_dim=4,
            n_cats=4,
            alpha_pos_map=name,
            alpha_pos_kwargs=kwargs,
        )
        raw = raw_alpha_value_for_map(target, name, kwargs)
        actual = dec._alpha_pos(torch.tensor([[raw]], dtype=torch.float32)).item()
        assert math.isclose(actual, target, rel_tol=1e-5, abs_tol=1e-6), name
