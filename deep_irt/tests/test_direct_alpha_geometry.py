"""Tests for the direct alpha-space geometry control."""

from __future__ import annotations

import math

import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.run_direct_alpha_geometry import (
    direct_alpha_loss,
    flatten_training_observations,
    gpcm_logits_from_alpha,
    optimize_alpha_cell,
    torch_preconditioner,
)


def test_gpcm_logits_match_manual_binary_case():
    alpha = torch.tensor([2.0])
    item = torch.tensor([0])
    theta = torch.tensor([0.5])
    beta = torch.tensor([[0.1]])
    logits = gpcm_logits_from_alpha(alpha, item, theta, beta)
    assert torch.allclose(logits, torch.tensor([[0.0, 0.8]]), atol=1e-7)


def test_direct_alpha_loss_is_finite_and_differentiable():
    cfg = BenchDataConfig(
        name="tiny",
        kind="static",
        n_learners=20,
        n_items=5,
        seq_len=10,
        n_cats=4,
        seed=0,
    )
    ds = generate(cfg)
    obs = flatten_training_observations(ds, torch.device("cpu"))
    alpha = torch.full((cfg.n_items,), 0.5, requires_grad=True)
    loss = direct_alpha_loss(alpha, obs)
    loss.backward()
    assert math.isfinite(float(loss.item()))
    assert alpha.grad is not None
    assert alpha.grad.shape == alpha.shape


def test_torch_preconditioners_reference_values():
    alpha = torch.tensor([0.5])
    assert torch.allclose(
        torch_preconditioner(alpha, {"preconditioner": "identity"}),
        torch.tensor([1.0]),
    )
    assert torch.allclose(
        torch_preconditioner(alpha, {"preconditioner": "exp"}),
        torch.tensor([0.25]),
    )
    assert torch.allclose(
        torch_preconditioner(alpha, {"preconditioner": "softplus"}),
        torch.tensor([(1.0 - math.exp(-0.5)) ** 2]),
    )
    assert torch.allclose(
        torch_preconditioner(
            alpha, {"preconditioner": "scaled_sigmoid", "scale": 2.0}
        ),
        torch.tensor([(0.5 * (1.0 - 0.5 / 2.0)) ** 2]),
    )


def test_optimize_alpha_cell_smoke():
    cfg = BenchDataConfig(
        name="tiny",
        kind="static",
        n_learners=40,
        n_items=6,
        seq_len=12,
        n_cats=4,
        seed=1,
    )
    ds = generate(cfg)
    obs = flatten_training_observations(ds, torch.device("cpu"))
    cell = optimize_alpha_cell(
        obs=obs,
        condition="projected",
        lr=0.1,
        epochs=5,
        alpha_init=0.5,
        alpha_max=20.0,
        epsilon=1e-6,
        checkpoints=[0, 1, 5],
    )
    assert math.isfinite(cell["final_loss"])
    assert cell["history"][0]["epoch"] == 0
    assert cell["history"][-1]["epoch"] == 5
    assert cell["alpha_min"] >= 0.0
