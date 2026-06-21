"""Tests for the E7 neural map-isolation controls."""

from __future__ import annotations

import pytest
import torch

from deep_irt.bench.run_neural_map_isolation import (
    _MAP_SPECS,
    _resolve_map_specs,
    _resolve_representations,
    apply_representation_control,
)
from deep_irt.core import DeepIRTModel


def _model():
    return DeepIRTModel(
        num_items=10,
        emb_dim=4,
        hidden_dim=8,
        n_cats=4,
        decoder="gpcm",
        state_alpha=True,
        item_key_dim=6,
        alpha_pos_map="softplus",
        decouple=False,
        device=torch.device("cpu"),
        seed=123,
    )


def test_apply_representation_control_freezes_item_embeddings_only():
    model = _model()
    info = apply_representation_control(model, "frozen_item_embeddings")

    assert info["representation_control"] == "frozen_item_embeddings"
    assert "encoder.item_val_emb.weight" in info["frozen_parameter_names"]
    assert "encoder.item_key_emb.weight" in info["frozen_parameter_names"]
    assert model.encoder.item_val_emb.weight.requires_grad is False
    assert model.encoder.item_key_emb.weight.requires_grad is False
    assert model.encoder.lstm.weight_ih_l0.requires_grad is True
    assert model.decoder.fc_a_state.weight.requires_grad is True


def test_apply_representation_control_freezes_backbone_not_item_key():
    model = _model()
    info = apply_representation_control(model, "frozen_backbone")

    assert any(name.startswith("encoder.lstm.") for name in info["frozen_parameter_names"])
    assert "encoder.theta_proj.weight" in info["frozen_parameter_names"]
    assert model.encoder.lstm.weight_ih_l0.requires_grad is False
    assert model.encoder.theta_proj.weight.requires_grad is False
    assert model.encoder.item_key_emb.weight.requires_grad is True


def test_map_and_representation_resolvers():
    maps = _resolve_map_specs(["raw", "clipped-relu"], quick=False)
    assert list(maps) == ["identity", "clipped_relu"]
    assert maps["clipped_relu"].grad_clip_norm == 1.0
    assert _resolve_representations(["learned-all", "frozen_encoder"], quick=False) == [
        "learned_all",
        "frozen_encoder",
    ]


def test_bad_representation_raises():
    model = _model()
    with pytest.raises(ValueError):
        apply_representation_control(model, "unknown")


def test_clipped_map_specs_have_explicit_ranges():
    for name in ("clipped_identity", "clipped_relu"):
        spec = _MAP_SPECS[name]
        assert "clip_min" in spec.kwargs
        assert "clip_max" in spec.kwargs
        assert spec.kwargs["clip_min"] < spec.kwargs["clip_max"]
